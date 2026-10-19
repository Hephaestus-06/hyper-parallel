# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Unit tests for pure-text offline SFT conversion and reading."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hyper_parallel.data.indexed.indexed_sft_dataset import SFTIndexedDataset
from hyper_parallel.data.tools.offline_sft import (
    encode_sft_record,
    normalize_sft_record,
    pack_sft_samples,
    write_sft_dataset,
)


class _Tokenizer:
    """Small deterministic tokenizer for SFT contract tests."""

    chat_template = None
    eos_token = "<eos>"
    eos_token_id = 99
    pad_token_id = 0
    name_or_path = "test-tokenizer"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        tokenized = []
        segments = text.split(self.eos_token)
        for index, segment in enumerate(segments):
            tokenized.extend(ord(char) for char in segment)
            if index < len(segments) - 1:
                tokenized.append(self.eos_token_id)
        return tokenized

    def apply_chat_template(
            self,
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            eos_token=None,
    ):
        del add_generation_prompt
        eos = self.eos_token if eos_token is None else eos_token
        rendered = "".join(f"<{item['role']}>{item['content']}{eos}</{item['role']}>" for item in messages)
        ids = self.encode(rendered) if tokenize else rendered
        return {"input_ids": ids} if return_dict else ids


class TestOfflineSFT(unittest.TestCase):
    """Cover format normalization, masks, packing, and indexed roundtrip."""

    def test_source_adapters(self):
        self.assertEqual(normalize_sft_record({"instruction": "q", "output": "a"}).source_format, "alpaca")
        self.assertEqual(
            normalize_sft_record({"prompt": "q", "completion": "a"}).direct_segments,
            ("q", "a"),
        )
        sharegpt = normalize_sft_record({"conversations": [{"from": "human", "value": "q"},
                                                             {"from": "gpt", "value": "a"}]})
        self.assertEqual([message.role for message in sharegpt.messages], ["user", "assistant"])

    def test_invalid_source_records_are_rejected(self):
        invalid_records = (
            {},
            {"instruction": "q", "output": ""},
            {"messages": [{"role": "unknown", "content": "x"}]},
        )
        for record in invalid_records:
            with self.subTest(record=record), self.assertRaises(ValueError):
                normalize_sft_record(record)

    def test_assistant_labels_and_eos(self):
        tokenizer = _Tokenizer()
        record = normalize_sft_record({"messages": [{"role": "user", "content": "q"},
                                                     {"role": "assistant", "content": "a"}]})
        input_ids, labels = encode_sft_record(record, tokenizer, max_seq_len=32)
        self.assertEqual(len(input_ids), len(labels))
        self.assertTrue(np.all(labels[:3] == -100))
        self.assertIn(tokenizer.eos_token_id, labels)

    def test_no_eos_for_message_and_alpaca_formats(self):
        tokenizer = _Tokenizer()
        records = (
            {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
            {"instruction": "q", "output": "a"},
        )
        for record in records:
            with self.subTest(record=record):
                input_ids, labels = encode_sft_record(
                    normalize_sft_record(record), tokenizer, max_seq_len=64, append_eos=False
                )
                self.assertNotIn(tokenizer.eos_token_id, input_ids)
                self.assertNotIn(tokenizer.eos_token_id, labels)

    def test_no_eos_with_tokenizer_template(self):
        tokenizer = _Tokenizer()
        tokenizer.chat_template = "test-template"
        record = normalize_sft_record(
            {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
        )
        input_ids, labels = encode_sft_record(
            record, tokenizer, max_seq_len=64, chat_template="tokenizer", append_eos=False
        )
        self.assertNotIn(tokenizer.eos_token_id, input_ids)
        self.assertNotIn(tokenizer.eos_token_id, labels)

    def test_completion_only_masks_previous_assistant(self):
        tokenizer = _Tokenizer()
        record = normalize_sft_record({"messages": [{"role": "user", "content": "q"},
                                                     {"role": "assistant", "content": "a1"},
                                                     {"role": "user", "content": "q2"},
                                                     {"role": "assistant", "content": "a2"}]})
        _, labels = encode_sft_record(record, tokenizer, max_seq_len=64, loss_mode="completion_only")
        final_segment = tokenizer.encode("Assistant: a2<eos>\n", add_special_tokens=False)
        self.assertEqual(int(np.count_nonzero(labels != -100)), len(final_segment))

    def test_packing_masks_padding_and_boundaries(self):
        samples = [
            (np.asarray([1, 2]), np.asarray([-100, 2])),
            (np.asarray([3, 4]), np.asarray([-100, 4])),
        ]
        packed = pack_sft_samples(samples, sequence_length=6, pad_token_id=0)
        self.assertEqual(packed[0][0].tolist(), [1, 2, 3, 4, 0, 0])
        self.assertEqual(packed[0][1].tolist(), [-100, 2, -100, 4, -100, -100])
        self.assertEqual(packed[0][2], [0, 2, 4])

    def test_no_packing_and_truncation(self):
        samples = [
            (np.asarray([1, 2, 3]), np.asarray([-100, 2, 3])),
            (np.asarray([4, 5]), np.asarray([-100, 5])),
        ]
        unpacked = pack_sft_samples(samples, sequence_length=2, pad_token_id=0, packing=False)
        self.assertEqual(len(unpacked), 2)
        self.assertEqual(unpacked[0][0].tolist(), [2, 3])
        self.assertEqual(unpacked[0][1].tolist(), [2, 3])
        self.assertEqual(unpacked[1][0].tolist(), [4, 5])
        self.assertEqual(unpacked[1][1].tolist(), [-100, 5])

    def test_indexed_roundtrip(self):
        tokenizer = _Tokenizer()
        with self.subTest("write and read"):
            with tempfile.TemporaryDirectory() as directory:
                manifest = write_sft_dataset(
                    [{"prompt": "q", "completion": "a"}, {"prompt": "x", "completion": "y"}],
                    output_prefix=str(Path(directory) / "sft"),
                    tokenizer=tokenizer,
                    sequence_length=8,
                )
                self.assertEqual(manifest["labels_are_shifted"], False)
                dataset = SFTIndexedDataset(str(Path(directory) / "sft"), split="train")
                self.assertEqual(len(dataset), 1)
                self.assertEqual(dataset[0]["tokens"].shape[0], 8)
                self.assertEqual(
                    dataset[0]["labels"].tolist(),
                    [
                        -100,
                        ord("a"),
                        tokenizer.eos_token_id,
                        -100,
                        ord("y"),
                        tokenizer.eos_token_id,
                        -100,
                        -100,
                    ],
                )
                self.assertEqual(dataset[0]["loss_mask"].sum().item(), 4)

    def test_manifest_rejects_shifted_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "sft")
            write_sft_dataset(
                [{"prompt": "q", "completion": "a"}],
                output_prefix=prefix,
                tokenizer=_Tokenizer(),
                sequence_length=8,
            )
            manifest_path = Path(directory) / "sft_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["labels_are_shifted"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unshifted labels"):
                SFTIndexedDataset(prefix)

    def test_manifest_rejects_record_count_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "sft")
            write_sft_dataset(
                [{"prompt": "q", "completion": "a"}],
                output_prefix=prefix,
                tokenizer=_Tokenizer(),
                sequence_length=8,
            )
            manifest_path = Path(directory) / "sft_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["splits"]["train"]["num_records"] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "record count"):
                SFTIndexedDataset(prefix)

    def test_manifest_rejects_incompatible_split_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "sft")
            write_sft_dataset(
                [{"prompt": "q", "completion": "a"}],
                output_prefix=prefix,
                tokenizer=_Tokenizer(),
                sequence_length=8,
                split="train",
            )
            with self.assertRaisesRegex(ValueError, "append_eos"):
                write_sft_dataset(
                    [{"prompt": "q", "completion": "a"}],
                    output_prefix=prefix,
                    tokenizer=_Tokenizer(),
                    sequence_length=8,
                    split="valid",
                    append_eos=False,
                )


if __name__ == "__main__":
    unittest.main()
