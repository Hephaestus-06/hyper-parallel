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
"""Offline preparation for pure-text supervised fine-tuning datasets.

The converter deliberately keeps source normalization, tokenization, and
packing in one module.  This makes the loss boundary explicit before data is
written to the generic indexed-dataset storage format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.text.chat_template import build_chat_template
from hyper_parallel.data.tools import io as indexed_dataset
from hyper_parallel.data.text.sft.schema import CanonicalSFTRecord, SFTMessage

logger = get_dataset_logger(__name__)

_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}


def _require_mapping(record: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError(f"SFT record must be a mapping, got {type(record).__name__}")
    return record


def _text(value: Any, field_name: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"SFT field {field_name!r} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"SFT field {field_name!r} must not be empty")
    return value


def _normalize_role(role: Any, role_map: Mapping[str, str] | None = None) -> str:
    role_name = _text(role, "role", allow_empty=False).lower()
    mapped = (role_map or {}).get(role_name, _ROLE_ALIASES.get(role_name, role_name))
    if mapped not in _ALLOWED_ROLES:
        raise ValueError(f"Unsupported SFT role {role!r}; expected one of {sorted(_ALLOWED_ROLES)}")
    return mapped


def _messages(value: Any, *, role_map: Mapping[str, str] | None = None,
              default_trainable: bool | None = None) -> tuple[SFTMessage, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("SFT messages must be a non-empty list")
    normalized = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"SFT message {index} must be a mapping")
        role = _normalize_role(item.get("role"), role_map)
        content = _text(item.get("content"), "content")
        trainable = item.get("trainable", default_trainable)
        if trainable is None:
            trainable = role == "assistant"
        if not isinstance(trainable, bool):
            raise ValueError(f"SFT message {index} trainable flag must be boolean")
        normalized.append(SFTMessage(role, content, trainable))
    return tuple(normalized)


def _alpaca_record(record: Mapping[str, Any], user_template: str) -> CanonicalSFTRecord:
    instruction = _text(record.get("instruction"), "instruction", allow_empty=False)
    source_input = _text(record.get("input", ""), "input")
    output = _text(record.get("output"), "output", allow_empty=False)
    try:
        user_content = user_template.format(instruction=instruction, input=source_input)
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError("alpaca user_template must use valid {instruction} and {input} fields") from exc
    return CanonicalSFTRecord(
        (SFTMessage("user", user_content, False), SFTMessage("assistant", output, True)),
        "alpaca",
    )


def normalize_sft_record(
        record: Mapping[str, Any],
        *,
        source_format: str = "auto",
        role_map: Mapping[str, str] | None = None,
        user_template: str = "{instruction}\n{input}",
) -> CanonicalSFTRecord:
    """Normalize one supported source record.

    Args:
        record: Alpaca, prompt-completion, messages, or ShareGPT mapping.
        source_format: Explicit format or ``"auto"`` for field-based detection.
        role_map: Optional aliases used for ShareGPT and messages roles.
        user_template: Format string for Alpaca instruction/input fields.

    Returns:
        A canonical record retaining assistant/completion loss boundaries.

    Raises:
        ValueError: If the schema is missing fields or contains invalid roles.
    """
    record = _require_mapping(record)
    selected = source_format.lower()
    if selected == "auto":
        if "messages" in record:
            selected = "messages"
        elif "conversations" in record:
            selected = "sharegpt"
        elif "instruction" in record and "output" in record:
            selected = "alpaca"
        elif "prompt" in record and "completion" in record:
            selected = "prompt_completion"
        else:
            raise ValueError("Unable to infer SFT source format from record fields")

    if selected in {"alpaca", "instruction"}:
        return _alpaca_record(record, user_template)
    if selected in {"messages", "chat"}:
        return CanonicalSFTRecord(_messages(record.get("messages"), role_map=role_map), "messages")
    if selected in {"sharegpt", "share_gpt"}:
        conversations = record.get("conversations")
        if not isinstance(conversations, Sequence) or isinstance(conversations, (str, bytes)):
            raise ValueError("ShareGPT conversations must be a non-empty list")
        converted = []
        for index, item in enumerate(conversations):
            if not isinstance(item, Mapping):
                raise ValueError(f"ShareGPT conversation {index} must be a mapping")
            role = _normalize_role(item.get("from"), role_map)
            converted.append({"role": role, "content": item.get("value")})
        return CanonicalSFTRecord(_messages(converted), "sharegpt")
    if selected in {"prompt_completion", "prompt-completion", "completion"}:
        prompt = record.get("prompt")
        completion = record.get("completion")
        if isinstance(prompt, str) and isinstance(completion, str):
            _text(prompt, "prompt")
            _text(completion, "completion", allow_empty=False)
            return CanonicalSFTRecord(
                (SFTMessage("user", prompt, False), SFTMessage("assistant", completion, True)),
                "prompt_completion",
                direct_segments=(prompt, completion),
            )
        prompt_messages = _messages(prompt, role_map=role_map, default_trainable=False)
        completion_messages = _messages(completion, role_map=role_map, default_trainable=True)
        return CanonicalSFTRecord((*prompt_messages, *completion_messages), "prompt_completion")
    raise ValueError(f"Unsupported SFT source format: {source_format!r}")


def _apply_loss_mode(record: CanonicalSFTRecord, loss_mode: str) -> CanonicalSFTRecord:
    if loss_mode not in {"assistant_only", "completion_only"}:
        raise ValueError("loss_mode must be 'assistant_only' or 'completion_only'")
    if loss_mode == "assistant_only":
        return record
    assistant_indices = [index for index, message in enumerate(record.messages) if message.role == "assistant"]
    if not assistant_indices:
        raise ValueError("completion_only requires at least one assistant message")
    last_index = assistant_indices[-1]
    messages = tuple(
        SFTMessage(message.role, message.content, index == last_index)
        for index, message in enumerate(record.messages)
    )
    return CanonicalSFTRecord(messages, record.source_format, record.direct_segments)


def encode_sft_record(
        record: CanonicalSFTRecord,
        tokenizer: Any,
        *,
        max_seq_len: int,
        loss_mode: str = "assistant_only",
        chat_template: str | None = None,
        append_eos: bool = True,
        truncate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize one canonical record and return unshifted ids and labels."""
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    record = _apply_loss_mode(record, loss_mode)
    if record.direct_segments is not None:
        prompt, completion = record.direct_segments
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        completion_ids = list(tokenizer.encode(completion, add_special_tokens=False))
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if append_eos and eos_id is not None:
            completion_ids.append(int(eos_id))
        input_ids = prompt_ids + completion_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
    else:
        template_name = chat_template
        if template_name is None:
            template_name = "tokenizer" if getattr(tokenizer, "chat_template", None) else "default"
        template = build_chat_template(template_name, tokenizer)
        messages = [
            {"role": message.role, "content": message.content, "loss_mask": int(message.trainable)}
            for message in record.messages
        ]
        template_kwargs = (
            {"append_eos": append_eos}
            if template_name in {"default", "tokenizer", "gpt_oss", "llama2"}
            else {}
        )
        template_limit = max_seq_len if truncate else 2 ** 31 - 1
        encoded = template.encode_messages(messages, max_seq_len=template_limit, **template_kwargs)
        input_ids = list(encoded["input_ids"])
        labels = list(encoded["labels"])
    if not input_ids:
        raise ValueError("SFT record produced no tokens")
    if truncate:
        input_ids = input_ids[-max_seq_len:]
        labels = labels[-max_seq_len:]
    if len(input_ids) != len(labels):
        raise ValueError("Tokenizer produced input_ids and labels with different lengths")
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("SFT record contains no trainable completion tokens")
    return np.asarray(input_ids, dtype=np.int32), np.asarray(labels, dtype=np.int32)


def pack_sft_samples(
        samples: Iterable[tuple[np.ndarray, np.ndarray]],
        *,
        sequence_length: int,
        pad_token_id: int,
        packing: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, list[int]]]:
    """Pack encoded samples without creating cross-sample supervision targets."""
    return list(iter_pack_sft_samples(
        samples,
        sequence_length=sequence_length,
        pad_token_id=pad_token_id,
        packing=packing,
    ))


def iter_pack_sft_samples(
        samples: Iterable[tuple[np.ndarray, np.ndarray]],
        *,
        sequence_length: int,
        pad_token_id: int,
        packing: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray, list[int]]]:
    """Yield packed samples incrementally for large offline datasets."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    current_ids: list[int] = []
    current_labels: list[int] = []
    boundaries: list[int] = [0]

    def flush() -> tuple[np.ndarray, np.ndarray, list[int]] | None:
        if not current_ids:
            return None
        ids = current_ids + [pad_token_id] * (sequence_length - len(current_ids))
        labels = current_labels + [IGNORE_INDEX] * (sequence_length - len(current_labels))
        item = (np.asarray(ids, dtype=np.int32), np.asarray(labels, dtype=np.int32), boundaries.copy())
        current_ids.clear()
        current_labels.clear()
        boundaries[:] = [0]
        return item

    for input_ids, labels in samples:
        if len(input_ids) != len(labels) or len(input_ids) == 0:
            raise ValueError("Each SFT sample must have equal, non-empty input_ids and labels")
        if len(input_ids) > sequence_length:
            input_ids = input_ids[-sequence_length:]
            labels = labels[-sequence_length:]
        if not packing:
            pad_length = sequence_length - len(input_ids)
            ids = np.asarray(input_ids.tolist() + [pad_token_id] * pad_length, dtype=np.int32)
            target = np.asarray(labels.tolist() + [IGNORE_INDEX] * pad_length, dtype=np.int32)
            yield ids, target, [0, len(ids)]
            continue
        if current_ids and len(current_ids) + len(input_ids) > sequence_length:
            item = flush()
            if item is not None:
                yield item
        current_ids.extend(input_ids.tolist())
        current_labels.extend(labels.tolist())
        boundaries.append(len(current_ids))
        if len(current_ids) == sequence_length:
            item = flush()
            if item is not None:
                yield item
    item = flush()
    if item is not None:
        yield item


def _manifest_path(output_prefix: Path) -> Path:
    return output_prefix.with_name(f"{output_prefix.name}_manifest.json")


def write_sft_dataset(
        records: Iterable[Mapping[str, Any]],
        *,
        output_prefix: str,
        tokenizer: Any,
        sequence_length: int,
        split: str = "train",
        source_format: str = "auto",
        loss_mode: str = "assistant_only",
        chat_template: str | None = None,
        role_map: Mapping[str, str] | None = None,
        user_template: str = "{instruction}\n{input}",
        packing: bool = True,
        append_eos: bool = True,
        pad_token_id: int | None = None,
) -> dict[str, Any]:
    """Convert records and write the paired SFT Indexed Dataset files."""
    if not split or any(char in split for char in "/\\"):
        raise ValueError("split must be a simple non-empty name")
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("pad_token_id or tokenizer eos_token_id is required")
    source_formats = set()
    output = Path(output_prefix).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    split_prefix = output.with_name(f"{output.name}_{split}")
    tokenizer_name = getattr(tokenizer, "name_or_path", None)
    manifest_file = _manifest_path(output)
    previous = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.is_file() else None
    manifest_contract = {
        "format": "hyperparallel_sft_v1",
        "loss_mode": loss_mode,
        "packing": bool(packing),
        "labels_are_shifted": False,
        "ignore_index": IGNORE_INDEX,
        "sequence_length": sequence_length,
        "tokenizer": tokenizer_name,
        "chat_template": chat_template or (
            "tokenizer_default" if getattr(tokenizer, "chat_template", None) else "default"
        ),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": int(pad_token_id),
        "append_eos": bool(append_eos),
    }
    if previous is not None:
        for field, value in manifest_contract.items():
            previous_value = previous.get(field, True if field == "append_eos" else None)
            if previous_value != value:
                raise ValueError(f"Existing SFT manifest field {field!r} conflicts with the new split")

    ids_builder = indexed_dataset.IndexedDatasetBuilder(
        str(split_prefix) + "_input_ids.bin", dtype=np.int32
    )
    labels_builder = indexed_dataset.IndexedDatasetBuilder(
        str(split_prefix) + "_labels.bin", dtype=np.int32
    )
    record_count = 0

    def encoded_samples() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for source_record in records:
            canonical = normalize_sft_record(
                source_record,
                source_format=source_format,
                role_map=role_map,
                user_template=user_template,
            )
            source_formats.add(canonical.source_format)
            yield encode_sft_record(
                canonical,
                tokenizer,
                max_seq_len=sequence_length,
                loss_mode=loss_mode,
                chat_template=chat_template,
                append_eos=append_eos,
            )

    try:
        for input_ids, labels, _ in iter_pack_sft_samples(
                encoded_samples(), sequence_length=sequence_length,
                pad_token_id=int(pad_token_id), packing=packing):
            ids_builder.add_item(torch.as_tensor(input_ids))
            ids_builder.end_document()
            labels_builder.add_item(torch.as_tensor(labels))
            labels_builder.end_document()
            record_count += 1
        if not source_formats:
            raise ValueError("SFT input contains no records")
        ids_builder.finalize(str(split_prefix) + "_input_ids.idx")
        labels_builder.finalize(str(split_prefix) + "_labels.idx")
    except Exception:
        ids_builder.data_file.close()
        labels_builder.data_file.close()
        raise
    manifest = {
        **manifest_contract,
        "source_format": next(iter(source_formats)) if len(source_formats) == 1 else "mixed",
        "num_records": record_count,
        "splits": {split: {"num_records": record_count}},
    }
    if manifest_file.is_file():
        previous_splits = previous.get("splits", {})
        previous_splits.update(manifest["splits"])
        manifest["splits"] = previous_splits
        manifest["num_records"] = sum(int(item["num_records"]) for item in previous_splits.values())
        if previous.get("source_format") != manifest["source_format"]:
            manifest["source_format"] = "mixed"
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote %d SFT records to %s", record_count, split_prefix)
    return manifest


def _as_int_array(
        record: Mapping[str, Any], field: str, *, dtype: Any = np.int32, allow_scalar: bool = False
) -> np.ndarray:
    """Convert one tokenized HF field to a one-dimensional integer array."""
    if field not in record:
        raise ValueError(f"Tokenized SFT record is missing {field!r}")
    try:
        value = np.asarray(record[field], dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Tokenized SFT field {field!r} must be an integer sequence") from exc
    if value.ndim == 0 and allow_scalar:
        value = value.reshape(1)
    if value.ndim != 1:
        raise ValueError(f"Tokenized SFT field {field!r} must be one-dimensional")
    return value


def write_tokenized_sft_dataset(
        records: Iterable[Mapping[str, Any]],
        *,
        output_prefix: str,
        sequence_length: int,
        split: str = "train",
        pad_token_id: int = 0,
        ignore_index: int = IGNORE_INDEX,
) -> dict[str, Any]:
    """Write an already-tokenized/packed HF SFT dataset.

    This accepts the common generated Dataset schema with ``input_ids``,
    ``labels``, ``cu_seqlens`` and ``real_len``.  No tokenizer or chat
    template is applied; labels and packed boundaries are preserved.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not split or any(char in split for char in "/\\"):
        raise ValueError("split must be a simple non-empty name")
    output = Path(output_prefix).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    split_prefix = output.with_name(f"{output.name}_{split}")
    manifest_file = _manifest_path(output)
    previous = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.is_file() else None
    contract = {
        "format": "hyperparallel_sft_v1",
        "source_format": "tokenized",
        "loss_mode": "precomputed",
        "packing": True,
        "labels_are_shifted": False,
        "ignore_index": int(ignore_index),
        "sequence_length": int(sequence_length),
        "pad_token_id": int(pad_token_id),
        "tokenized_fields": ["input_ids", "labels", "cu_seqlens", "real_len"],
        "boundaries": True,
    }
    if previous is not None:
        for field, value in contract.items():
            if field in previous and previous[field] != value:
                raise ValueError(f"Existing SFT manifest field {field!r} conflicts with the new split")

    builders = {
        "input_ids": indexed_dataset.IndexedDatasetBuilder(str(split_prefix) + "_input_ids.bin", dtype=np.int32),
        "labels": indexed_dataset.IndexedDatasetBuilder(str(split_prefix) + "_labels.bin", dtype=np.int32),
        "cu_seqlens": indexed_dataset.IndexedDatasetBuilder(
            str(split_prefix) + "_cu_seqlens.bin", dtype=np.int32
        ),
        "real_len": indexed_dataset.IndexedDatasetBuilder(str(split_prefix) + "_real_len.bin", dtype=np.int32),
    }
    count = 0
    try:
        for record in records:
            input_ids = _as_int_array(record, "input_ids")
            labels = _as_int_array(record, "labels")
            boundaries = _as_int_array(record, "cu_seqlens")
            real_len_array = _as_int_array(record, "real_len", dtype=np.int64, allow_scalar=True)
            if input_ids.size != sequence_length or labels.size != sequence_length:
                raise ValueError("Tokenized SFT input_ids and labels must equal sequence_length")
            if real_len_array.size != 1:
                raise ValueError("Tokenized SFT real_len must contain one value")
            real_len = int(real_len_array[0])
            if real_len <= 0 or real_len > sequence_length:
                raise ValueError("Tokenized SFT real_len must be within input_ids length")
            if boundaries.size < 2 or int(boundaries[0]) != 0 or int(boundaries[-1]) != real_len:
                raise ValueError("Tokenized SFT cu_seqlens must start at 0 and end at real_len")
            if np.any(np.diff(boundaries) <= 0) or np.any(boundaries < 0) or np.any(boundaries > real_len):
                raise ValueError("Tokenized SFT cu_seqlens must be increasing within real_len")
            if np.any(input_ids[real_len:] != int(pad_token_id)):
                raise ValueError("Tokenized SFT input_ids after real_len must use pad_token_id")
            if np.any(labels[real_len:] != int(ignore_index)):
                raise ValueError("Tokenized SFT labels after real_len must use ignore_index")
            if not np.any(labels[:real_len] != int(ignore_index)):
                raise ValueError("Tokenized SFT record contains no trainable labels")
            for field, value in (("input_ids", input_ids), ("labels", labels),
                                 ("cu_seqlens", boundaries), ("real_len", real_len_array)):
                builders[field].add_item(torch.as_tensor(value))
                builders[field].end_document()
            count += 1
        if not count:
            raise ValueError("Tokenized SFT input contains no records")
        for field, builder in builders.items():
            builder.finalize(str(split_prefix) + f"_{field}.idx")
    except Exception:
        for builder in builders.values():
            builder.data_file.close()
        raise
    manifest = {**contract, "num_records": count, "splits": {split: {"num_records": count}}}
    if previous is not None:
        splits = previous.get("splits", {})
        splits.update(manifest["splits"])
        manifest["splits"] = splits
        manifest["num_records"] = sum(int(item["num_records"]) for item in splits.values())
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote %d tokenized SFT records to %s", count, split_prefix)
    return manifest


def read_hf_tokenized_dataset(path: str | Path, *, split: str = "train") -> Iterable[Mapping[str, Any]]:
    """Read a Hugging Face ``save_to_disk`` Dataset with the common SFT schema."""
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("Reading HF SFT datasets requires the 'datasets' package") from exc
    dataset = load_from_disk(str(Path(path).expanduser()))
    if not hasattr(dataset, "column_names"):
        if split not in dataset:
            raise ValueError(f"HF SFT DatasetDict does not contain split {split!r}")
        dataset = dataset[split]
    required = {"input_ids", "labels", "cu_seqlens", "real_len"}
    missing = sorted(required.difference(dataset.column_names))
    if missing:
        raise ValueError(f"HF SFT dataset is missing columns: {missing}")
    return (dataset[index] for index in range(len(dataset)))


def read_jsonl(path: str | Path) -> Iterable[Mapping[str, Any]]:
    """Yield JSON objects from an UTF-8 JSONL file."""
    with Path(path).expanduser().open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            yield _require_mapping(record)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare pure-text SFT indexed data")
    parser.add_argument("--input", required=True, help="Input JSONL file or HF save_to_disk directory")
    parser.add_argument("--input-format", choices=("jsonl", "hf_tokenized"), default="jsonl")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--tokenizer", required=False)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--source-format", default="auto")
    parser.add_argument("--loss-mode", default="assistant_only", choices=("assistant_only", "completion_only"))
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--user-template", default="{instruction}\n{input}")
    parser.add_argument("--no-packing", action="store_true")
    parser.add_argument("--no-eos", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    """Run the pure-text SFT preparation CLI."""
    args = _build_cli().parse_args()
    if args.input_format == "hf_tokenized":
        write_tokenized_sft_dataset(
            read_hf_tokenized_dataset(args.input, split=args.split),
            output_prefix=args.output_prefix,
            sequence_length=args.sequence_length,
            pad_token_id=args.pad_token_id,
        )
        return
    if not args.tokenizer:
        raise ValueError("--tokenizer is required for JSONL input")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    write_sft_dataset(
        read_jsonl(args.input), output_prefix=args.output_prefix, tokenizer=tokenizer,
        sequence_length=args.sequence_length, split=args.split, source_format=args.source_format,
        loss_mode=args.loss_mode, chat_template=args.chat_template, user_template=args.user_template,
        packing=not args.no_packing, append_eos=not args.no_eos,
    )


if __name__ == "__main__":
    main()
