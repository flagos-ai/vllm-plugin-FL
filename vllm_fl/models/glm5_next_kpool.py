# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.24 cache objects for the GLM5-Next kpool indexer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


@dataclass(frozen=True, kw_only=True)
class KpoolTailSpec(SlidingWindowSpec):
    """One per-request circular page containing raw K and gate-score tail."""

    def max_admission_blocks_per_request(
        self, max_num_batched_tokens: int, max_model_len: int
    ) -> int:
        del max_num_batched_tokens, max_model_len
        return 1

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(isinstance(spec, KpoolTailSpec) for spec in kv_cache_specs.values())

    @property
    def participates_in_prefix_caching(self) -> bool:
        return False


class KpoolTailManager(FullAttentionManager):
    """Reference no-hit/no-prune fixed-one-block manager for the tail ring."""

    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        del (
            block_hashes,
            max_length,
            block_pool,
            kv_cache_spec,
            drop_eagle_block,
            alignment_tokens,
            dcp_world_size,
            pcp_world_size,
        )
        return tuple([] for _ in kv_cache_group_ids)

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        del request, num_tokens, retention_interval

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        del running_request_id
        return 0

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        del num_computed_tokens
        return 0

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        del request_id, total_computed_tokens

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        del (
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        return max(1 - len(self.req_to_blocks.get(request_id, ())), 0)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        # The base implementation grows with sequence length. The tail instead
        # obtains one page on first admission and circularly reuses it forever.
        del num_tokens, num_tokens_main_model
        req_blocks = self.req_to_blocks[request_id]
        if req_blocks:
            return []
        new_blocks = self.block_pool.get_new_blocks(1)
        req_blocks.extend(new_blocks)
        return new_blocks


def compute_kpool_tail_slot_mapping(
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_actual_tokens: int,
    num_reqs: int,
    kpool: int,
) -> torch.Tensor:
    """Map every real token to its request-owned circular tail block.

    The generic vLLM slot mapper indexes ``block_table[request, pos //
    block_size]``. A ``KpoolTailManager`` owns exactly one block per request,
    so columns after zero are unset and positions beyond the first pool would
    collapse onto physical block zero. The tail cache instead uses the first
    block as a ring addressed by ``pos % kpool``.
    """
    out = slot_mapping.clone()
    if num_actual_tokens == 0:
        return out

    device = slot_mapping.device
    tokens = torch.arange(num_actual_tokens, device=device)
    request_ids = torch.searchsorted(query_start_loc, tokens, right=True) - 1
    request_ids.clamp_(min=0, max=num_reqs - 1)
    own_blocks = block_table[:num_reqs, 0].index_select(0, request_ids)
    own_blocks = own_blocks.to(torch.int64)
    token_positions = positions[:num_actual_tokens].to(torch.int64)
    out[:num_actual_tokens] = own_blocks * kpool + torch.remainder(
        token_positions, kpool
    )
    return out


class KpoolTailMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.ALWAYS
    supports_update_block_table = False
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        # Storage-only tail: the common builder state is sufficient. In
        # particular, do not allocate the normal indexer's paged-MQA buffers,
        # whose kernel block must be 32/64 rather than kpool (4).
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        del vllm_config, kv_cache_spec
        return AttentionCGSupport.ALWAYS

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        del common_prefix_len, fast_build
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(common_attn_metadata)
        )
        slot_mapping = common_attn_metadata.slot_mapping
        positions = common_attn_metadata.positions
        if positions is not None:
            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
            )
        return DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=None,
            decode=None,
        )


class KpoolTailBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "KPOOL_TAIL"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        assert num_kv_heads == 1 and head_size % 2 == 0
        return (num_blocks, 2, block_size, head_size // 2)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        del include_num_layers_dimension
        return (0, 1, 2, 3)

    @staticmethod
    def get_builder_cls():
        return KpoolTailMetadataBuilder


class Glm5NextIndexerCache(DeepseekV32IndexerCache):
    def __init__(self, *, index_kpool: int, **kwargs) -> None:
        super().__init__(**kwargs)
        assert index_kpool > 1
        cache_config = kwargs.get("cache_config")
        if cache_config is not None:
            assert cache_config.block_size % index_kpool == 0, (
                "Glm5NextIndexerCache: cache block_size "
                f"({cache_config.block_size}) must be divisible by index_kpool "
                f"({index_kpool}) so chunked-prefill boundaries stay aligned"
            )
        self.index_kpool = index_kpool

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        spec = super().get_kv_cache_spec(vllm_config)
        assert isinstance(spec, MLAAttentionSpec)
        spec = replace(spec, compress_ratio=self.index_kpool)
        storage_block_size = spec.block_size // self.index_kpool
        assert (
            spec.block_size % self.index_kpool == 0 and storage_block_size % 32 == 0
        ), (
            "GLM5-Next kpool requires logical block_size to be a multiple of "
            f"index_kpool*32 ({self.index_kpool * 32}); got {spec.block_size}."
        )
        return spec


class Glm5NextTailCache(DeepseekV32IndexerCache):
    def __init__(self, *, index_kpool: int, **kwargs) -> None:
        super().__init__(**kwargs)
        assert index_kpool > 1
        self.index_kpool = index_kpool

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        del vllm_config
        return KpoolTailSpec(
            block_size=self.index_kpool,
            num_kv_heads=1,
            head_size=2 * self.head_dim,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=self.index_kpool,
            indexes_kv_by_block_stride=True,
        )

    def get_attn_backend(self):
        return KpoolTailBackend


__all__ = [
    "Glm5NextIndexerCache",
    "Glm5NextTailCache",
    "KpoolTailBackend",
    "KpoolTailManager",
    "KpoolTailSpec",
]
