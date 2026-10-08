# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vllm/v1/attention/backends/mla/flashmla_sparse.py

"""
FlagGems-based FlashMLA Sparse Attention Backend.

This module provides a FlagGems-accelerated implementation of the FlashMLA Sparse
attention backend. It reuses the metadata/builder from vLLM's native
FlashMLASparseBackend and overrides only the kernel dispatch methods to call
FlagGems Triton kernels instead of the native CUDA FlashMLA kernels.
"""

from typing import TYPE_CHECKING, ClassVar

import torch
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionCGSupport,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
    FlashMLASparseMetadataBuilder,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)


class MLASparseFLMetadataBuilder(FlashMLASparseMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class MLASparseFLBackend(FlashMLASparseBackend):
    """FlagGems-based FlashMLA Sparse attention backend."""

    @staticmethod
    def get_name() -> str:
        return "FLAGGEMS_FLASHMLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["MLASparseFLImpl"]:
        return MLASparseFLImpl

    @staticmethod
    def get_builder_cls() -> type["MLASparseFLMetadataBuilder"]:
        return MLASparseFLMetadataBuilder

    @classmethod
    def supports_sink(cls) -> bool:
        return True


class MLASparseFLImpl(FlashMLASparseImpl):
    """FlagGems implementation of FlashMLA Sparse attention.

    Overrides the kernel dispatch methods to use FlagGems Triton kernels
    while reusing all control flow (padding, metadata handling, prefill/decode
    routing) from the parent class.
    """

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
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        # SparseMLACommonImpl accepts explicit keywords only. Consume the
        # architecture-level sink here and forward only standard MLA args.
        sinks: torch.Tensor | None = mla_args.pop("sinks", None)
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            topk_indices_buffer=topk_indices_buffer,
            indexer=indexer,
            **mla_args,
        )
        self._validate_sinks(sinks, num_heads)
        self.sinks = sinks

    def do_kv_cache_update(
        self,
        kv_c_normed,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
    ) -> None:
        """Override the backend hook, never the global vLLM custom-op table."""
        from flag_gems import concat_and_cache_mla

        if kv_cache.numel() == 0:
            return
        slots = slot_mapping.flatten()
        concat_and_cache_mla(
            kv_c_normed[: slots.shape[0]],
            k_pe.squeeze(1)[: slots.shape[0]],
            kv_cache,
            slots,
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        """Reuse vLLM metadata/control flow with FlagGems query concat."""
        if isinstance(q, tuple):
            from flag_gems import cat_out

            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            cat_out((ql_nope, q_pe), dim=-1, out=q)
        return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)

    @staticmethod
    def _validate_sinks(sinks: torch.Tensor | None, num_heads: int) -> None:
        if sinks is None:
            return
        if sinks.dtype != torch.float32:
            raise ValueError(
                "FlagGems Sparse MLA sinks must have dtype torch.float32, "
                f"but got {sinks.dtype}."
            )
        if sinks.ndim != 1 or sinks.shape[0] != num_heads:
            raise ValueError(
                "FlagGems Sparse MLA sinks must have shape "
                f"({num_heads},), but got {tuple(sinks.shape)}."
            )

    def _sinks_for_query(
        self,
        q: torch.Tensor,
        head_dim: int,
        kernel_heads: int,
    ) -> torch.Tensor | None:
        sinks = self.sinks
        if sinks is None:
            return None

        query_heads = q.shape[head_dim]
        if sinks.shape[0] != query_heads:
            raise ValueError(
                "FlagGems Sparse MLA sink head count must match the runtime "
                f"query layout: sinks={sinks.shape[0]}, "
                f"query_heads={query_heads}."
            )
        if sinks.device != q.device:
            raise ValueError(
                "FlagGems Sparse MLA sinks and query must share a device, "
                f"but got sinks={sinks.device}, query={q.device}."
            )
        if kernel_heads < query_heads:
            raise ValueError(
                "FlagGems Sparse MLA kernel head count cannot be smaller "
                f"than query head count: kernel_heads={kernel_heads}, "
                f"query_heads={query_heads}."
            )
        if kernel_heads == query_heads:
            return sinks

        # -inf makes padded sink lanes a no-op in the softmax denominator.
        padded_sinks = sinks.new_full((kernel_heads,), float("-inf"))
        padded_sinks[:query_heads] = sinks
        return padded_sinks

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FP8 decode kernel using FlagGems flash_mla_with_kvcache."""
        if self.sinks is not None:
            raise NotImplementedError("Sparse MLA sinks require BF16 KV cache")
        from flag_gems.fused.flash_mla_with_kvcache import (
            FlashMLASchedMeta,
            flash_mla_with_kvcache,
        )

        # q shape: (batch, seq_len, num_heads, head_dim)
        actual_num_heads = q.size(2)
        padded_num_heads = self.fp8_decode_padded_heads

        # Pad query if needed (kernel only supports h_q = 64 or 128)
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for FP8 sparse decode kernel"
            )
            q_padded = q.new_zeros((q.size(0), q.size(1), padded_num_heads, q.size(3)))
            q_padded[:, :, :actual_num_heads, :] = q
            q = q_padded

        # Adapt vLLM's FlashMLASchedMeta to FlagGems' FlashMLASchedMeta
        original_meta = kernel_metadata.scheduler_metadata
        if isinstance(original_meta, FlashMLASchedMeta):
            flaggems_meta = original_meta
        else:
            flaggems_meta = FlashMLASchedMeta(
                have_initialized=original_meta.have_initialized,
                config=original_meta.config,
                tile_scheduler_metadata=original_meta.tile_scheduler_metadata,
                num_splits=original_meta.num_splits,
            )

        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            cache_seqlens=kernel_metadata.cache_lens,
            head_dim_v=512,
            tile_scheduler_metadata=flaggems_meta,
            is_fp8_kvcache=True,
            indices=topk_indices,
            softmax_scale=self.softmax_scale,
        )

        # Sync metadata back if we created a new object
        if flaggems_meta is not original_meta:
            original_meta.have_initialized = flaggems_meta.have_initialized
            original_meta.config = flaggems_meta.config
            original_meta.tile_scheduler_metadata = (
                flaggems_meta.tile_scheduler_metadata
            )
            original_meta.num_splits = flaggems_meta.num_splits

        # Slice output back to actual head count if we padded
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]

        return out, lse

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """BF16 prefill/decode kernel using FlagGems flash_mla_sparse_fwd."""
        import flag_gems

        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        attn_sink = self._sinks_for_query(
            q, head_dim=1, kernel_heads=q.shape[1]
        )

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output, _, _ = flag_gems.flash_mla_sparse_fwd(
            q=q,
            kv=kv_c_and_k_pe_cache,
            indices=topk_indices,
            sm_scale=self.softmax_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )

        return output
