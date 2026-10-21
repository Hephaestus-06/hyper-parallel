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

"""Offline SFT pipeline with variable-length Parquet and Arrow records.

The implementation intentionally keeps the offline artifact unpadded.  Padding
is performed by :func:`collate_sft_batch`, where the actual batch shape and
parallel alignment are known.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.text.sft.hf_dataset import iter_hf_records
from hyper_parallel.data.text.sft.schema import PackedSFTRecord

DEFAULT_SEQUENCE_LENGTH = 4096
_FIELDS = ("input_ids", "labels", "supervision_mask", "cu_seqlens", "real_len")


def _packed_schema(pa: Any) -> Any:
    """Build the exact physical schema shared by Parquet and Arrow."""
    return pa.schema([
        ("input_ids", pa.list_(pa.int32())),
        ("labels", pa.list_(pa.int32())),
        ("supervision_mask", pa.list_(pa.int8())),
        ("cu_seqlens", pa.list_(pa.int32())),
        ("real_len", pa.int32()),
    ])


def _require_pyarrow() -> Any:
    """Import PyArrow only for storage operations."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet/Arrow support requires the 'pyarrow' package") from exc
    return pa, pq


def _mapped_record(record: Mapping[str, Any], column_mapping: Mapping[str, str] | None) -> Mapping[str, Any]:
    """Apply explicit source-column mapping before schema detection."""
    if not column_mapping:
        return record
    mapped = dict(record)
    for canonical, source in column_mapping.items():
        if source not in record:
            raise ValueError(f"Mapped SFT source column {source!r} is missing")
        mapped[canonical] = record[source]
    return mapped


def _encode_records(
        records: Iterable[Mapping[str, Any]],
        tokenizer: Any,
        *,
        sequence_length: int,
        source_format: str,
        loss_mode: str,
        chat_template: str | None,
        role_map: Mapping[str, str] | None,
        user_template: str,
        append_eos: bool,
        column_mapping: Mapping[str, str] | None,
) -> tuple[Iterator[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Normalize and tokenize records while retaining filtering statistics."""
    stats: dict[str, Any] = {
        "input_records": 0, "encoded_records": 0, "filtered_too_long": 0, "source_formats": set()
    }

    def iterator() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        from hyper_parallel.data.tools.offline_sft import encode_sft_record, normalize_sft_record

        for raw in records:
            stats["input_records"] += 1
            canonical = normalize_sft_record(
                _mapped_record(raw, column_mapping),
                source_format=source_format,
                role_map=role_map,
                user_template=user_template,
            )
            stats["source_formats"].add(canonical.source_format)
            ids, labels = encode_sft_record(
                canonical,
                tokenizer,
                max_seq_len=sequence_length,
                loss_mode=loss_mode,
                chat_template=chat_template,
                append_eos=append_eos,
                truncate=False,
            )
            if len(ids) > sequence_length:
                stats["filtered_too_long"] += 1
                continue
            stats["encoded_records"] += 1
            yield ids, labels

    return iterator(), stats


def pack_tokenized_samples(
        samples: Iterable[tuple[Sequence[int], Sequence[int]]],
        *,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        ignore_index: int = IGNORE_INDEX,
) -> tuple[list[PackedSFTRecord], dict[str, int]]:
    """Pack samples with first-fit decreasing bins and no offline padding.

    A sample longer than ``sequence_length`` is discarded as a whole.  This
    function never truncates a sample and never creates a cross-sample label.
    """
    if sequence_length <= 0 or sequence_length > DEFAULT_SEQUENCE_LENGTH:
        raise ValueError(f"sequence_length must be within 1..{DEFAULT_SEQUENCE_LENGTH}")
    valid_samples: list[tuple[np.ndarray, np.ndarray]] = []
    stats = {"samples": 0, "filtered_too_long": 0, "packed_records": 0}
    for ids_value, labels_value in samples:
        stats["samples"] += 1
        ids = np.asarray(ids_value, dtype=np.int32).reshape(-1)
        labels = np.asarray(labels_value, dtype=np.int32).reshape(-1)
        if ids.size == 0 or ids.size != labels.size:
            raise ValueError("Each tokenized SFT sample must have equal, non-empty fields")
        if ids.size > sequence_length:
            stats["filtered_too_long"] += 1
            continue
        valid_samples.append((ids, labels))

    bins: list[list[tuple[np.ndarray, np.ndarray]]] = []
    bin_lengths: list[int] = []
    for sample in sorted(valid_samples, key=lambda item: -len(item[0])):
        sample_length = len(sample[0])
        target_bin = next(
            (index for index, current_length in enumerate(bin_lengths)
             if current_length + sample_length <= sequence_length),
            None,
        )
        if target_bin is None:
            bins.append([sample])
            bin_lengths.append(sample_length)
        else:
            bins[target_bin].append(sample)
            bin_lengths[target_bin] += sample_length

    output: list[PackedSFTRecord] = []
    for packed_samples in bins:
        ids_values: list[int] = []
        labels_values: list[int] = []
        boundaries = [0]
        for ids, labels in packed_samples:
            ids_values.extend(ids.tolist())
            sample_labels = labels.copy()
            sample_labels[0] = ignore_index
            labels_values.extend(sample_labels.tolist())
            boundaries.append(len(ids_values))
        labels_array = np.asarray(labels_values, dtype=np.int32)
        output.append(PackedSFTRecord(
            np.asarray(ids_values, dtype=np.int32), labels_array,
            (labels_array != ignore_index).astype(np.int8), tuple(boundaries),
        ))
    stats["packed_records"] = len(output)
    return output, stats


def _validate_record(record: Mapping[str, Any], sequence_length: int, ignore_index: int) -> dict[str, Any]:
    """Validate one packed record before storage or runtime consumption."""
    missing = [field for field in _FIELDS if field not in record]
    if missing:
        raise ValueError(f"packed SFT record is missing fields: {missing}")
    values = {field: np.asarray(record[field]) for field in _FIELDS[:-1]}
    if any(value.ndim != 1 or value.dtype.kind not in "iu" for value in values.values()):
        raise ValueError("packed token, label, mask, and boundary fields must be 1-D integer sequences")
    real_len_value = np.asarray(record["real_len"])
    if real_len_value.size != 1 or real_len_value.dtype.kind not in "iu":
        raise ValueError("packed real_len must be one integer value")
    ids = values["input_ids"].astype(np.int32, copy=False)
    labels = values["labels"].astype(np.int32, copy=False)
    mask = values["supervision_mask"].astype(np.int8, copy=False)
    bounds = values["cu_seqlens"].astype(np.int32, copy=False)
    real_len = int(real_len_value.reshape(-1)[0])
    if real_len <= 0 or real_len > sequence_length or len(ids) != real_len:
        raise ValueError("packed real_len and input_ids must satisfy 0 < real_len <= sequence_length")
    if len(labels) != real_len or len(mask) != real_len:
        raise ValueError("packed token, label, and supervision fields must have equal lengths")
    if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != real_len or np.any(np.diff(bounds) <= 0):
        raise ValueError("cu_seqlens must be strictly increasing from 0 to real_len")
    expected_mask = (labels != ignore_index).astype(np.int8)
    if not np.array_equal(mask, expected_mask):
        raise ValueError("supervision_mask must equal labels != ignore_index")
    if np.any((mask != 0) & (mask != 1)):
        raise ValueError("supervision_mask values must be binary")
    return {
        "input_ids": ids.tolist(), "labels": labels.tolist(), "supervision_mask": mask.tolist(),
        "cu_seqlens": bounds.tolist(), "real_len": real_len,
    }


def write_packed_parquet(
        records: Iterable[PackedSFTRecord | Mapping[str, Any]],
        path: str | Path,
        *,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        ignore_index: int = IGNORE_INDEX,
) -> int:
    """Write variable-length packed records to an intermediate Parquet file."""
    pa, pq = _require_pyarrow()
    rows = []
    for record in records:
        rows.append(_validate_record(record.as_dict() if isinstance(record, PackedSFTRecord) else record,
                                     sequence_length, ignore_index))
    schema = _packed_schema(pa)
    table = pa.Table.from_pylist(rows, schema=schema)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination)
    return len(rows)


def convert_parquet_to_arrow(
        parquet_path: str | Path,
        arrow_path: str | Path,
        *,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        ignore_index: int = IGNORE_INDEX,
) -> int:
    """Convert Parquet to Arrow IPC and verify schema and row contents."""
    pa, pq = _require_pyarrow()
    table = pq.read_table(parquet_path)
    rows = [_validate_record(row, sequence_length, ignore_index) for row in table.to_pylist()]
    output = Path(arrow_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    schema = _packed_schema(pa)
    arrow_table = pa.Table.from_pylist(rows, schema=schema)
    if not table.schema.equals(schema):
        raise ValueError(f"Parquet SFT schema mismatch: expected {schema}, got {table.schema}")
    with pa.OSFile(str(output), "wb") as sink:
        with pa.ipc.new_file(sink, arrow_table.schema) as writer:
            writer.write_table(arrow_table)
    converted_rows = read_arrow_records(output)
    if len(converted_rows) != len(rows):
        raise ValueError("Parquet-to-Arrow row count validation failed")
    if converted_rows != rows:
        raise ValueError("Parquet-to-Arrow value validation failed")
    with pa.memory_map(str(output), "r") as source:
        converted_schema = pa.ipc.open_file(source).schema
    if not converted_schema.equals(table.schema):
        raise ValueError("Parquet-to-Arrow schema validation failed")
    return len(rows)


def read_arrow_records(path: str | Path) -> list[dict[str, Any]]:
    """Read Arrow IPC records as Python dictionaries for inspection/runtime."""
    pa, _ = _require_pyarrow()
    with pa.memory_map(str(Path(path).expanduser()), "r") as source:
        return pa.ipc.open_file(source).read_all().to_pylist()


def collate_sft_batch(
        records: Sequence[Mapping[str, Any]],
        *,
        pad_token_id: int,
        sequence_multiple: int = 1,
        ignore_index: int = IGNORE_INDEX,
) -> dict[str, np.ndarray]:
    """Pad a batch to its own maximum length and preserve packed boundaries."""
    if not records:
        raise ValueError("SFT batch cannot be empty")
    if sequence_multiple <= 0:
        raise ValueError("sequence_multiple must be positive")
    normalized = [_validate_record(record, 2 ** 31 - 1, ignore_index) for record in records]
    max_len = max(int(record["real_len"]) for record in normalized)
    target_len = ((max_len + sequence_multiple - 1) // sequence_multiple) * sequence_multiple
    input_ids = np.full((len(normalized), target_len), pad_token_id, dtype=np.int32)
    labels = np.full((len(normalized), target_len), ignore_index, dtype=np.int32)
    mask = np.zeros((len(normalized), target_len), dtype=np.int8)
    attention_mask = np.zeros((len(normalized), target_len), dtype=np.int8)
    boundaries = np.full((len(normalized), max(len(row["cu_seqlens"]) for row in normalized)), -1, dtype=np.int32)
    real_lens = np.zeros(len(normalized), dtype=np.int32)
    for index, row in enumerate(normalized):
        length = int(row["real_len"])
        input_ids[index, :length] = row["input_ids"]
        labels[index, :length] = row["labels"]
        mask[index, :length] = row["supervision_mask"]
        attention_mask[index, :length] = 1
        boundaries[index, :len(row["cu_seqlens"])] = row["cu_seqlens"]
        real_lens[index] = length
    return {
        "input_ids": input_ids, "attention_mask": attention_mask,
        "labels": labels, "supervision_mask": mask,
        "cu_seqlens": boundaries, "real_len": real_lens,
    }


def build_sft_dataset(
        records: Iterable[Mapping[str, Any]],
        *,
        tokenizer: Any,
        output_dir: str | Path,
        split: str = "train",
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        source_format: str = "auto",
        loss_mode: str = "assistant_only",
        chat_template: str | None = None,
        role_map: Mapping[str, str] | None = None,
        user_template: str = "{instruction}\n{input}",
        append_eos: bool = True,
        column_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run normalization through Arrow conversion and write a manifest."""
    if not split or any(char in split for char in "/\\"):
        raise ValueError("split must be a simple non-empty name")
    if sequence_length <= 0 or sequence_length > DEFAULT_SEQUENCE_LENGTH:
        raise ValueError(f"sequence_length must be within 1..{DEFAULT_SEQUENCE_LENGTH}")
    encoded, stats = _encode_records(
        records, tokenizer, sequence_length=sequence_length, source_format=source_format,
        loss_mode=loss_mode, chat_template=chat_template, role_map=role_map,
        user_template=user_template, append_eos=append_eos, column_mapping=column_mapping,
    )
    packed, packing_stats = pack_tokenized_samples(encoded, sequence_length=sequence_length)
    destination = Path(output_dir).expanduser()
    parquet_path = destination / "parquet" / f"{split}.parquet"
    arrow_path = destination / "arrow" / f"{split}.arrow"
    write_packed_parquet(packed, parquet_path, sequence_length=sequence_length)
    convert_parquet_to_arrow(parquet_path, arrow_path, sequence_length=sequence_length)
    manifest_path = destination / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    tokenizer_name = getattr(tokenizer, "name_or_path", None)
    selected_template = chat_template or (
        "tokenizer_default" if getattr(tokenizer, "chat_template", None) else "default"
    )
    contract = {
        "format": "hyperparallel_sft_v1", "loss_mode": loss_mode,
        "labels_are_shifted": False, "ignore_index": IGNORE_INDEX,
        "sequence_length": sequence_length, "tokenizer": tokenizer_name,
        "chat_template": selected_template,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "append_eos": append_eos, "packing": True,
    }
    if previous is not None:
        for key, value in contract.items():
            if previous.get(key) != value:
                raise ValueError(f"Existing SFT manifest field {key!r} conflicts with this split")
    splits = previous.get("splits", {}) if previous else {}
    splits[split] = {
        "num_records": len(packed), "parquet": str(parquet_path), "arrow": str(arrow_path),
        "stats": {
            "input_records": stats["input_records"],
            "encoded_records": stats["encoded_records"],
            "filtered_too_long": stats["filtered_too_long"],
            "packed_records": packing_stats["packed_records"],
        },
    }
    source_formats = set(stats["source_formats"])
    if previous and previous.get("source_format") not in {None, "mixed"}:
        source_formats.add(previous["source_format"])
    manifest = {
        **contract,
        "source_format": next(iter(source_formats)) if len(source_formats) == 1 else "mixed",
        "num_records": sum(int(item["num_records"]) for item in splits.values()),
        "splits": splits,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _build_cli() -> argparse.ArgumentParser:
    """Build the offline SFT command-line interface."""
    parser = argparse.ArgumentParser(description="Build packed SFT Parquet and Arrow datasets")
    parser.add_argument("--input", required=True, help="HF dataset ID, save_to_disk directory, or JSONL file")
    parser.add_argument("--input-format", choices=("hf", "jsonl"), default="hf")
    parser.add_argument("--split", default="train")
    parser.add_argument("--config-name")
    parser.add_argument("--revision")
    parser.add_argument("--data-files")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--source-format", default="auto")
    parser.add_argument("--loss-mode", choices=("assistant_only", "completion_only"), default="assistant_only")
    parser.add_argument("--chat-template")
    parser.add_argument("--user-template", default="{instruction}\n{input}")
    parser.add_argument("--column-mapping", help="JSON object mapping canonical names to source column names")
    parser.add_argument("--no-eos", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    """Run the offline SFT converter from the command line."""
    args = _build_cli().parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.input_format == "jsonl":
        from hyper_parallel.data.tools.offline_sft import read_jsonl
        records = read_jsonl(args.input)
    else:
        records = iter_hf_records(
            args.input, split=args.split, config_name=args.config_name,
            revision=args.revision, data_files=args.data_files,
        )
    column_mapping = json.loads(args.column_mapping) if args.column_mapping else None
    if column_mapping is not None and not isinstance(column_mapping, dict):
        raise ValueError("--column-mapping must be a JSON object")
    build_sft_dataset(
        records, tokenizer=tokenizer, output_dir=args.output_dir, split=args.split,
        sequence_length=args.sequence_length, source_format=args.source_format,
        loss_mode=args.loss_mode, chat_template=args.chat_template,
        user_template=args.user_template, append_eos=not args.no_eos,
        column_mapping=column_mapping,
    )


if __name__ == "__main__":
    main()
