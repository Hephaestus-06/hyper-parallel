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
# ============================================================================
"""Arrow-backed packed SFT Dataset and dynamic batch collation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from hyper_parallel.data.constants import IGNORE_INDEX


def _arrow_table(path: str | Path) -> Any:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("Arrow SFT reading requires the 'pyarrow' package") from exc
    with pa.memory_map(str(Path(path).expanduser()), "r") as source:
        return pa.ipc.open_file(source).read_all()


def _validate_manifest(path: Path, row_count: int, sequence_length: int) -> None:
    """Validate the manifest contract for an Arrow split when available."""
    manifest_path = path.parent.parent / "manifest.json" if path.parent.name == "arrow" else None
    if manifest_path is None or not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("labels_are_shifted") is not False:
        raise ValueError("SFT Arrow manifest must declare unshifted labels")
    if int(manifest.get("sequence_length", 0)) != sequence_length:
        raise ValueError("SFT Arrow manifest sequence_length does not match the reader")
    split = path.stem
    split_info = manifest.get("splits", {}).get(split)
    if split_info is not None and int(split_info.get("num_records", -1)) != row_count:
        raise ValueError("SFT Arrow manifest record count does not match the Arrow file")


class SFTArrowDataset:
    """Read variable-length Arrow rows without applying offline padding."""

    REQUIRED_FIELDS = ("input_ids", "labels", "supervision_mask", "cu_seqlens", "real_len")

    def __init__(self, path: str | Path, *, sequence_length: int = 4096,
                 ignore_index: int = IGNORE_INDEX) -> None:
        arrow_path = Path(path).expanduser()
        if sequence_length <= 0 or sequence_length > 4096:
            raise ValueError("sequence_length must be within 1..4096")
        self.table = _arrow_table(arrow_path)
        self.sequence_length = sequence_length
        self.ignore_index = ignore_index
        missing = [field for field in self.REQUIRED_FIELDS if field not in self.table.column_names]
        if missing:
            raise ValueError(f"Arrow SFT dataset is missing columns: {missing}")
        try:
            import pyarrow as pa
        except ImportError as exc:  # pragma: no cover - already required by _arrow_table
            raise RuntimeError("Arrow SFT reading requires the 'pyarrow' package") from exc
        expected_schema = pa.schema([
            ("input_ids", pa.list_(pa.int32())), ("labels", pa.list_(pa.int32())),
            ("supervision_mask", pa.list_(pa.int8())), ("cu_seqlens", pa.list_(pa.int32())),
            ("real_len", pa.int32()),
        ])
        if not self.table.schema.equals(expected_schema):
            raise ValueError(f"Arrow SFT schema mismatch: expected {expected_schema}, got {self.table.schema}")
        self._rows = self.table.to_pylist()
        _validate_manifest(arrow_path, len(self._rows), sequence_length)
        for row in self._rows:
            _validate_row(row, sequence_length, ignore_index)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("SFT Arrow Dataset index out of range")
        row = self._rows[index]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.ones(int(row["real_len"]), dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.long),
            "supervision_mask": torch.tensor(row["supervision_mask"], dtype=torch.long),
            "cu_seqlens": torch.tensor(row["cu_seqlens"], dtype=torch.long),
            "real_len": torch.tensor(row["real_len"], dtype=torch.long),
        }


def _validate_row(row: Mapping[str, Any], sequence_length: int, ignore_index: int) -> None:
    """Validate runtime row contracts before exposing a sample."""
    lengths = {field: len(row[field]) for field in ("input_ids", "labels", "supervision_mask")}
    real_len = int(row["real_len"])
    boundaries = np.asarray(row["cu_seqlens"]).reshape(-1).tolist()
    if real_len <= 0 or real_len > sequence_length or any(length != real_len for length in lengths.values()):
        raise ValueError("Arrow SFT row has inconsistent real_len or token field lengths")
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != real_len:
        raise ValueError("Arrow SFT row cu_seqlens must start at 0 and end at real_len")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("Arrow SFT row cu_seqlens must be strictly increasing")
    expected = [int(label != ignore_index) for label in row["labels"]]
    if list(row["supervision_mask"]) != expected:
        raise ValueError("Arrow SFT row supervision_mask does not match labels")


def collate_sft_batch(
        records: Sequence[Mapping[str, Any]],
        *,
        pad_token_id: int,
        sequence_multiple: int = 1,
        ignore_index: int = IGNORE_INDEX,
) -> dict[str, np.ndarray]:
    """Dynamically pad a batch while keeping each row's packed boundaries."""
    if not records:
        raise ValueError("SFT batch cannot be empty")
    if sequence_multiple <= 0:
        raise ValueError("sequence_multiple must be positive")
    for row in records:
        _validate_row(row, 2 ** 31 - 1, ignore_index)
    max_len = max(int(row["real_len"]) for row in records)
    target_len = ((max_len + sequence_multiple - 1) // sequence_multiple) * sequence_multiple
    max_segments = max(len(row["cu_seqlens"]) for row in records)
    input_ids = np.full((len(records), target_len), pad_token_id, dtype=np.int32)
    labels = np.full((len(records), target_len), ignore_index, dtype=np.int32)
    supervision_mask = np.zeros((len(records), target_len), dtype=np.int8)
    attention_mask = np.zeros((len(records), target_len), dtype=np.int8)
    cu_seqlens = np.full((len(records), max_segments), -1, dtype=np.int32)
    real_lens = np.zeros(len(records), dtype=np.int32)
    for row_index, row in enumerate(records):
        length = int(row["real_len"])
        input_ids[row_index, :length] = row["input_ids"]
        labels[row_index, :length] = row["labels"]
        supervision_mask[row_index, :length] = row["supervision_mask"]
        attention_mask[row_index, :length] = 1
        cu_seqlens[row_index, :len(row["cu_seqlens"])] = row["cu_seqlens"]
        real_lens[row_index] = length
    return {
        "input_ids": torch.from_numpy(input_ids).long(),
        "attention_mask": torch.from_numpy(attention_mask).long(),
        "labels": torch.from_numpy(labels).long(),
        "supervision_mask": torch.from_numpy(supervision_mask).long(),
        "cu_seqlens": torch.from_numpy(cu_seqlens).long(),
        "real_len": torch.from_numpy(real_lens).long(),
    }
