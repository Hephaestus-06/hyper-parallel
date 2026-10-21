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
"""Canonical source schema and packed SFT row types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SFTMessage:
    """One normalized conversation message and its loss participation."""

    role: str
    content: str
    trainable: bool = False


@dataclass(frozen=True)
class CanonicalSFTRecord:
    """Normalized SFT sample shared by all source-format adapters."""

    messages: tuple[SFTMessage, ...]
    source_format: str
    direct_segments: tuple[str, str] | None = None


@dataclass(frozen=True)
class PackedSFTRecord:
    """One unpadded packed record and its source-sample boundaries."""

    input_ids: np.ndarray
    labels: np.ndarray
    supervision_mask: np.ndarray
    cu_seqlens: tuple[int, ...]

    @property
    def real_len(self) -> int:
        """Return the number of valid tokens in this record."""
        return int(self.input_ids.size)

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""
        return {
            "input_ids": self.input_ids.astype(np.int32, copy=False).tolist(),
            "labels": self.labels.astype(np.int32, copy=False).tolist(),
            "supervision_mask": self.supervision_mask.astype(np.int8, copy=False).tolist(),
            "cu_seqlens": list(self.cu_seqlens),
            "real_len": self.real_len,
        }
