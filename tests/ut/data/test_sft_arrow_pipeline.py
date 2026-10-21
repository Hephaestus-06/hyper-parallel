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
"""White-box tests for the variable-length packed SFT Parquet/Arrow contract."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hyper_parallel.data.tools.sft.offline import (
    PackedSFTRecord,
    collate_sft_batch,
    convert_parquet_to_arrow,
    pack_tokenized_samples,
    read_arrow_records,
    write_packed_parquet,
)


class TestSFTArrowPipeline(unittest.TestCase):
    """Check packing, boundaries, storage roundtrip, and dynamic padding."""

    def test_4096_retained_and_4097_filtered_without_truncation(self):
        records, stats = pack_tokenized_samples([
            (np.arange(4096), np.arange(4096)),
            (np.arange(4097), np.arange(4097)),
        ])
        self.assertEqual(stats["filtered_too_long"], 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].real_len, 4096)
        self.assertEqual(records[0].cu_seqlens, (0, 4096))

    def test_pack_preserves_masks_and_variable_length(self):
        records, _ = pack_tokenized_samples([
            ([1, 2], [-100, 2]),
            ([3, 4, 5], [-100, 4, 5]),
        ], sequence_length=8)
        packed = records[0]
        self.assertIsInstance(packed, PackedSFTRecord)
        self.assertEqual(packed.input_ids.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(packed.labels.tolist(), [-100, 4, 5, -100, 2])
        self.assertEqual(packed.supervision_mask.tolist(), [0, 1, 1, 0, 1])
        self.assertEqual(packed.cu_seqlens, (0, 3, 5))
        self.assertEqual(packed.real_len, len(packed.input_ids))

    def test_batch_padding_is_dynamic_and_masks_padding(self):
        records, _ = pack_tokenized_samples([
            ([1, 2], [-100, 2]),
            ([3, 4, 5], [-100, 4, 5]),
        ], sequence_length=8)
        batch = collate_sft_batch(
            [record.as_dict() for record in records], pad_token_id=99, sequence_multiple=4
        )
        self.assertEqual(batch["input_ids"].shape, (1, 8))
        self.assertEqual(batch["input_ids"][0, 5:].tolist(), [99, 99, 99])
        self.assertEqual(batch["attention_mask"][0].tolist(), [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(batch["labels"][0, 5:].tolist(), [-100, -100, -100])
        self.assertEqual(batch["supervision_mask"][0, 5:].tolist(), [0, 0, 0])
        self.assertEqual(batch["cu_seqlens"][0].tolist(), [0, 2, 5])

    def test_parquet_to_arrow_preserves_schema_values_and_rows(self):
        try:
            import pyarrow
        except ImportError:
            self.skipTest("pyarrow is not installed")
        records, _ = pack_tokenized_samples([([1, 2], [-100, 2])], sequence_length=8)
        with tempfile.TemporaryDirectory() as directory:
            parquet = Path(directory) / "train.parquet"
            arrow = Path(directory) / "train.arrow"
            self.assertEqual(write_packed_parquet(records, parquet, sequence_length=8), 1)
            self.assertEqual(convert_parquet_to_arrow(parquet, arrow, sequence_length=8), 1)
            rows = read_arrow_records(arrow)
            self.assertEqual(rows[0], records[0].as_dict())
            self.assertEqual(len(rows), 1)

    def test_empty_split_keeps_schema(self):
        try:
            import pyarrow
        except ImportError:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as directory:
            parquet = Path(directory) / "empty.parquet"
            arrow = Path(directory) / "empty.arrow"
            self.assertEqual(write_packed_parquet([], parquet), 0)
            self.assertEqual(convert_parquet_to_arrow(parquet, arrow), 0)
            self.assertEqual(read_arrow_records(arrow), [])

    def test_invalid_boundary_rejected(self):
        with self.assertRaisesRegex(ValueError, "cu_seqlens"):
            collate_sft_batch([{
                "input_ids": [1, 2], "labels": [-100, 2],
                "supervision_mask": [0, 1], "cu_seqlens": [0, 1], "real_len": 2,
            }], pad_token_id=0)


if __name__ == "__main__":
    unittest.main()
