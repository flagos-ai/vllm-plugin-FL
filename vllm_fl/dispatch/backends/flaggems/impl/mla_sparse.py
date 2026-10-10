# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlagGems-backed sparse MLA for out-of-tree accelerators.

The metadata and cache contract match vLLM 0.24's sparse MLA backend.  Kernel
selection is backend-neutral: use FlagGems when its sparse MLA/cache writer is
available. Initialization rejects missing kernels before cache execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


def _flag_op(module: str, name: str):
    from vllm_fl.kernels.glm5_next.indexer_backend import _load_flaggems_op

    return _load_flaggems_op(module, name)


def _resolve_sparse_mla_kernels():
    from vllm_fl.dispatch.binding import OperatorBinding
    from vllm_fl.dispatch.manager import OpManager
    from vllm_fl.dispatch.types import BackendImplKind, OpImpl
    from vllm_fl.utils import use_flaggems_op

    manager = OpManager()
    bindings = []
    for name in ("concat_and_cache_mla", "flash_mla_sparse_fwd"):
        op = _flag_op(name, name)
        if op is None:
            raise RuntimeError(f"Sparse MLA requires FlagGems-vllm {name}")

        def call(*args, _op=op, **kwargs):
            return _op(*args, **kwargs)

        call._is_available = lambda _name=name: use_flaggems_op(_name)
        manager.registry.register_impl(
            OpImpl(name, "glm5.flaggems", BackendImplKind.DEFAULT, call)
        )
        binding = OperatorBinding(
            manager, name, graph_capabilities={"glm5.flaggems": False}
        )
        binding.describe()
        bindings.append(binding)
    return tuple(bindings)


def sparse_mla_cudagraph_support(head_size):
    return AttentionCGSupport.NEVER, "Sparse MLA runs at the eager attention boundary"


@dataclass
class FlagGemsSparseMLAMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048


class FlagGemsSparseMLAMetadataBuilder(
    AttentionMetadataBuilder[FlagGemsSparseMLAMetadata]
):
    # Sparse MLA remains an eager attention boundary.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        del vllm_config
        support, _ = sparse_mla_cudagraph_support(
            getattr(kv_cache_spec, "head_size", None)
        )
        return support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.req_id_per_token_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlagGemsSparseMLAMetadata:
        del common_prefix_len, fast_build
        metadata = common_attn_metadata
        starts = np.asarray(metadata.query_start_loc_cpu, dtype=np.int32)
        segment_lengths = np.diff(starts)
        request_ids = np.repeat(
            np.arange(segment_lengths.shape[0], dtype=np.int32), segment_lengths
        )
        self.req_id_per_token_buffer.fill_(0)
        if request_ids.size:
            request_ids_tensor = torch.from_numpy(request_ids)
            self.req_id_per_token_buffer[: request_ids.size].copy_(
                request_ids_tensor, non_blocking=True
            )
        return FlagGemsSparseMLAMetadata(
            num_reqs=metadata.num_reqs,
            max_query_len=metadata.max_query_len,
            max_seq_len=metadata.max_seq_len,
            num_actual_tokens=metadata.num_actual_tokens,
            query_start_loc=metadata.query_start_loc,
            slot_mapping=metadata.slot_mapping,
            block_table=metadata.block_table_tensor,
            req_id_per_token=self.req_id_per_token_buffer[: metadata.num_actual_tokens],
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )


class FlagGemsSparseMLABackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]
    # vLLM 0.24 resolves cudagraph support through the metadata builder, but
    # newer runners also inspect the backend class.  Keep both declarations in
    # sync: NEVER while a correctness fallback with host scalar reads is
    # reachable.  See FlagGemsSparseMLAMetadataBuilder for the rationale.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return FlagGemsSparseMLAMetadataBuilder.get_cudagraph_support(
            vllm_config, kv_cache_spec
        )

    @staticmethod
    def get_name() -> str:
        return "FLAGGEMS_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type[FlagGemsSparseMLAMetadataBuilder]:
        return FlagGemsSparseMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[FlagGemsSparseMLAImpl]:
        return FlagGemsSparseMLAImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # GLM5-Next is 512-NoPE; DeepSeek-V3.2 is 512-NoPE + 64-RoPE.
        return [512, 576]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        del capability
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        if num_kv_heads != 1:
            raise ValueError("Sparse MLA requires one latent KV head")
        return (num_blocks, block_size, head_size)


def _convert_request_to_physical_indices(
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_indices_i64 = token_indices.to(torch.int64)
    request_ids_i64 = request_ids.to(torch.int64).reshape(-1, 1)
    block_ids = torch.div(token_indices_i64, block_size, rounding_mode="floor")
    valid = (token_indices_i64 >= 0) & (block_ids < block_table.shape[1])
    safe_blocks = block_ids.clamp(min=0, max=block_table.shape[1] - 1)
    rows = request_ids_i64.expand_as(safe_blocks)
    physical_blocks = block_table[rows, safe_blocks].to(torch.int64)
    valid = valid & (physical_blocks >= 0)
    physical = physical_blocks * block_size + torch.remainder(
        token_indices_i64, block_size
    )
    valid_lengths = valid.sum(dim=-1).to(torch.int32)

    # ``flash_mla_sparse_fwd`` consumes only ``indices[:topk_length]``.  A
    # missing logical block can create an invalid entry in the middle of the
    # indexer's result, so merely replacing invalid entries with -1 would both
    # expose -1 to the kernel and drop later valid tokens.  Compact valid
    # physical slots to a contiguous prefix, matching vLLM's reference
    # ``COMPACT_TO_FRONT`` conversion contract.  The extra column is a sink for
    # every invalid entry and keeps this path free of host synchronization.
    width = physical.shape[-1]
    compact_dst = valid.to(torch.int64).cumsum(dim=-1) - 1
    compact_dst = torch.where(valid, compact_dst, width)
    compact = torch.full(
        (*physical.shape[:-1], width + 1),
        -1,
        dtype=physical.dtype,
        device=physical.device,
    )
    compact.scatter_(dim=-1, index=compact_dst, src=physical)
    return compact[..., :width].to(torch.int32), valid_lengths


class FlagGemsSparseMLAImpl(SparseMLAAttentionImpl[FlagGemsSparseMLAMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Any | None = None,
        **mla_args,
    ) -> None:
        del (
            alibi_slopes,
            sliding_window,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
        )
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "FlagGems sparse MLA portable path currently requires BF16 KV cache"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.softmax_scale = float(scale)
        self.kv_lora_rank = int(mla_args["kv_lora_rank"])
        self.topk_indices_buffer = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )

        self._cache_writer, self._sparse_attn = _resolve_sparse_mla_kernels()

    def do_kv_cache_update(
        self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
    ) -> None:
        if kv_cache.numel() == 0:
            return
        self._cache_writer(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype="auto" if kv_cache_dtype == "bfloat16" else kv_cache_dtype,
            scale=k_scale,
        )

    @staticmethod
    def _concat_query(q_nope, q_pe):
        from flaggems_vllm import concat_query

        return concat_query(q_nope, q_pe)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlagGemsSparseMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del layer
        if isinstance(q, tuple):
            q_nope, q_pe = q
            q = q_nope if q_pe.shape[-1] == 0 else self._concat_query(q_nope, q_pe)
        num_tokens = q.shape[0]
        if self.topk_indices_buffer is None:
            raise RuntimeError("Sparse MLA requires the indexer's top-k buffer")
        request_topk = self.topk_indices_buffer[:num_tokens]
        physical_topk, valid_lengths = _convert_request_to_physical_indices(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            request_topk,
            attn_metadata.block_size,
        )
        from flaggems_vllm import contiguous_copy, pad_attention_heads

        cache = contiguous_copy(kv_c_and_k_pe_cache).view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        actual_heads = q.shape[1]
        padded_heads = 64 if actual_heads <= 64 else 128
        q_padded = pad_attention_heads(q, padded_heads)
        output = self._sparse_attn(
            q_padded,
            cache,
            physical_topk.unsqueeze(1).contiguous(),
            self.softmax_scale,
            d_v=self.kv_lora_rank,
            topk_length=valid_lengths.contiguous(),
        )[0]
        return output[:, :actual_heads], None


__all__ = [
    "FlagGemsSparseMLABackend",
    "FlagGemsSparseMLAImpl",
    "FlagGemsSparseMLAMetadata",
    "FlagGemsSparseMLAMetadataBuilder",
]
