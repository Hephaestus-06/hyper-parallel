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
"""Reader for the paired Indexed Dataset files emitted by offline SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.indexed.indexed_data_reader import IndexedDataReader


class SFTIndexedDataset(Dataset):
    """Read unshifted SFT input IDs and labels from paired indexed files.

    The reader validates the manifest at construction time so a normal
    pretraining ``.bin/.idx`` pair cannot accidentally be used as SFT data.
    """

    def __init__(
            self,
            data_prefix: str,
            *,
            split: str = "train",
            manifest_path: str | None = None,
            mmap: bool = True,
            num_samples: int | None = None,
    ) -> None:
        """Open and validate one SFT split.

        Args:
            data_prefix: Base prefix passed to :func:`write_sft_dataset`.
            split: Split suffix used in the input and label file names.
            manifest_path: Optional explicit manifest path.
            mmap: Whether to memory-map indexed payloads.
            num_samples: Optional upper bound used by a Trainer split.
        """
        if not split or any(char in split for char in "/\\"):
            raise ValueError("split must be a simple non-empty name")
        prefix = Path(data_prefix).expanduser().resolve()
        manifest_file = Path(manifest_path).expanduser() if manifest_path else prefix.with_name(
            f"{prefix.name}_manifest.json"
        )
        if not manifest_file.is_file():
            raise FileNotFoundError(f"SFT manifest not found: {manifest_file}")
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid SFT manifest: {manifest_file}") from exc
        self._validate_manifest(manifest)
        split_prefix = prefix.with_name(f"{prefix.name}_{split}")
        self.input_reader = IndexedDataReader(str(split_prefix) + "_input_ids", mmap=mmap)
        self.label_reader = IndexedDataReader(str(split_prefix) + "_labels", mmap=mmap)
        if len(self.input_reader) != len(self.label_reader):
            raise ValueError("SFT input_ids and labels contain different record counts")
        split_metadata = manifest.get("splits", {}).get(split)
        expected_records = int(split_metadata["num_records"]) if split_metadata is not None else int(
            manifest["num_records"]
        )
        if expected_records != len(self.input_reader):
            raise ValueError(
                f"SFT manifest record count for {split!r} does not match the indexed input_ids record count"
            )
        if not np.array_equal(self.input_reader.sequence_lengths, self.label_reader.sequence_lengths):
            raise ValueError("SFT input_ids and labels contain different record lengths")
        self.manifest = manifest
        self.split = split
        self.ignore_index = int(manifest["ignore_index"])
        self.sequence_length = int(manifest["sequence_length"])
        self.packing = bool(manifest["packing"])
        self.num_samples = len(self.input_reader) if num_samples is None else int(num_samples)
        if self.num_samples < 0 or self.num_samples > len(self.input_reader):
            raise ValueError(
                f"num_samples must be between 0 and {len(self.input_reader)}, got {self.num_samples}"
            )
        if self.packing:
            lengths = self.input_reader.sequence_lengths
            if not np.all(lengths == self.sequence_length):
                raise ValueError("Packed SFT records must all equal manifest sequence_length")

    @staticmethod
    def _validate_manifest(manifest: Mapping[str, Any]) -> None:
        if not isinstance(manifest, Mapping):
            raise ValueError("SFT manifest root must be a JSON object")
        required = {
            "format", "source_format", "loss_mode", "packing", "labels_are_shifted",
            "ignore_index", "sequence_length", "num_records",
        }
        missing = sorted(required.difference(manifest))
        if missing:
            raise ValueError(f"SFT manifest is missing fields: {missing}")
        if manifest["format"] != "hyperparallel_sft_v1":
            raise ValueError(f"Unsupported SFT manifest format: {manifest['format']!r}")
        if manifest["labels_are_shifted"]:
            raise ValueError("SFT Indexed Dataset requires unshifted labels")
        if manifest["loss_mode"] not in {"assistant_only", "completion_only"}:
            raise ValueError(f"Unsupported SFT loss_mode: {manifest['loss_mode']!r}")
        if int(manifest["sequence_length"]) <= 0:
            raise ValueError("SFT manifest sequence_length must be positive")
        if int(manifest["ignore_index"]) != IGNORE_INDEX:
            raise ValueError(
                f"SFT manifest ignore_index must be {IGNORE_INDEX}, got {manifest['ignore_index']!r}"
            )
        if "append_eos" in manifest and not isinstance(manifest["append_eos"], bool):
            raise ValueError("SFT manifest append_eos must be a boolean")

    def __len__(self) -> int:
        """Return the configured number of records."""
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Read one sample and derive its binary loss mask."""
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        input_ids = np.asarray(self.input_reader[index]).copy()
        labels = np.asarray(self.label_reader[index]).copy()
        input_tensor = torch.from_numpy(input_ids).to(dtype=torch.long)
        label_tensor = torch.from_numpy(labels).to(dtype=torch.long)
        return {
            "tokens": input_tensor,
            "labels": label_tensor,
            "loss_mask": label_tensor.ne(self.ignore_index).to(dtype=torch.long),
        }


def build_sft_dataset_splits(
        *,
        data_prefix: str,
        train_valid_test_num_samples: tuple[int, int, int] | list[int],
        **_: Any,
) -> tuple[SFTIndexedDataset | None, SFTIndexedDataset | None, SFTIndexedDataset | None]:
    """Build train/validation/test SFT readers from one output prefix."""
    if not isinstance(data_prefix, str):
        raise ValueError("SFT data_prefix must be one output prefix, not a weighted blend")
    if len(train_valid_test_num_samples) != 3:
        raise ValueError("train_valid_test_num_samples must contain three values")
    datasets = []
    for split, requested in zip(("train", "valid", "test"), train_valid_test_num_samples):
        if requested == 0:
            datasets.append(None)
            continue
        datasets.append(SFTIndexedDataset(data_prefix, split=split, num_samples=int(requested)))
    return datasets[0], datasets[1], datasets[2]


__all__ = ["SFTIndexedDataset", "build_sft_dataset_splits"]
