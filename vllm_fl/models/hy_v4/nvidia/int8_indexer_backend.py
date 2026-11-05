# SPDX-License-Identifier: Apache-2.0
"""INT8 consumer capability; all native indexer metadata mathematics retained."""

from __future__ import annotations

from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadataBuilder,
)


class HYV4Int8IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    # FlagGems INT8 logits consumes row_batch/lengths/table, not a DeepGEMM
    # scheduler. The native schedule field/buffer and all other math remain.
    requires_paged_mqa_logits_metadata = False
    supports_common_decode_lens = True


class HYV4Int8IndexerBackend(DeepseekV32IndexerBackend):
    # Keep inherited get_name() for native prefill-kernel warmup classification.
    # Native grouping uses full_cls_name(), so FP8 and INT8 remain separate.
    @staticmethod
    def get_builder_cls() -> type[HYV4Int8IndexerMetadataBuilder]:
        return HYV4Int8IndexerMetadataBuilder


class HYV4Int8IndexerCache(DeepseekV32IndexerCache):
    def get_attn_backend(self) -> type[HYV4Int8IndexerBackend]:
        return HYV4Int8IndexerBackend
