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
"""Hugging Face Dataset loading for offline SFT conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


def iter_hf_records(
        dataset_name_or_path: str,
        *,
        split: str = "train",
        config_name: str | None = None,
        revision: str | None = None,
        data_files: Any = None,
        streaming: bool = False,
        **load_kwargs: Any,
) -> Iterable[Mapping[str, Any]]:
    """Load a local or Hub Hugging Face split without changing its schema."""
    try:
        from datasets import DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError("HF dataset loading requires the 'datasets' package") from exc
    path = Path(dataset_name_or_path).expanduser()
    if path.exists():
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise ValueError(f"HF dataset does not contain split {split!r}")
            dataset = dataset[split]
    else:
        dataset = load_dataset(
            dataset_name_or_path,
            name=config_name,
            split=split,
            revision=revision,
            data_files=data_files,
            streaming=streaming,
            **load_kwargs,
        )
    return (dataset[index] for index in range(len(dataset))) if not streaming else iter(dataset)
