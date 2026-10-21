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
"""Runtime reader and collator for HyperParallel packed SFT Arrow data."""

from .dataset import SFTArrowDataset, collate_sft_batch
from .hf_dataset import iter_hf_records
from .schema import CanonicalSFTRecord, PackedSFTRecord, SFTMessage

__all__ = [
    "CanonicalSFTRecord", "PackedSFTRecord", "SFTArrowDataset", "SFTMessage",
    "collate_sft_batch", "iter_hf_records",
]
