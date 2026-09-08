# Copyright 2026 BAAI. All rights reserved.

"""FlagGems-backed sparse MLA integration for vLLM Plugin FL.

This module adapts vLLM's ``FlashMLASparseBackend`` to the Hygon/BW1000
runtime used by GLM-5.2.  It has three deliberately narrow responsibilities:

1. expose sparse MLA through FL's ``CachedOp`` dispatch layer;
2. bridge the GLM full/shared Indexer semantics from vLLM PR #45895 to the
   vLLM 0.20.x constructor contract; and
3. replace FlagGems' default sparse FlashMLA Triton candidates with a tile
   that fits BW1000's per-workgroup shared-memory limit.

The GLM sparse Indexer runs before this backend.  By the time
``_bf16_flash_mla_kernel`` is called, ``topk_indices`` has already been
converted from per-request logical token positions to physical KV-cache
slots.  The kernel consumes BF16 query/KV tensors and returns the latent-value
attention output; this module does not calculate Indexer logits or TopK.

The FlagGems autotuner configuration is process-global.  It is changed lazily,
only on Hygon, and only once per worker process.  Other platforms retain their
existing FlagGems candidate list.
"""

from __future__ import annotations

from threading import Lock
from typing import ClassVar

import torch
import triton

from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
)

from vllm_fl.dispatch import CachedOp


_flash_mla_sparse_fwd = CachedOp(
    "flash_mla_sparse_fwd"
)

_hygon_sparse_config_lock = Lock()
_hygon_sparse_configured = False


def _configure_hygon_flashmla_sparse() -> None:
    """Install the BW1000-safe sparse FlashMLA Triton configuration.

    FlagGems currently provides BK=64/BH=64 candidates.  For GLM-5.2's BF16
    DQK=576 path, that configuration requests 81,920 bytes of shared memory,
    exceeding BW1000's 65,536-byte limit.

    The platform guard prevents a Hygon-specific tuning decision from changing
    sparse MLA behavior on other vendors.

    The fast path avoids locking after initialization; the second check prevents
    two threads from replacing the global candidate list concurrently during the
    first invocation.
    """

    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "hygon":
        return

    global _hygon_sparse_configured
    if _hygon_sparse_configured:
        return

    with _hygon_sparse_config_lock:
        if _hygon_sparse_configured:
            return

        from flag_gems.fused import flashmla_sparse

        flashmla_sparse.triton_flash_mla_sparse_fwd.configs = [
            triton.Config(
                {"BK": 32, "BH": 32},
                num_warps=8,
                num_stages=2,
            )
        ]
        _hygon_sparse_configured = True


class _TopKBufferRef:
    """Construction-only adapter for vLLM 0.20.x.

    vLLM PR #45895 makes FlashMLASparseImpl accept topk_indices_buffer directly
    when indexer is None.

    vLLM 0.20.x still requires indexer.topk_indices_buffer.
    This object exists only while calling the upstream constructor;
    it is not installed into the model and owns no parameters/cache.
    """

    __slots__ = ("topk_indices_buffer",)

    def __init__(
        self,
        topk_indices_buffer: torch.Tensor,
    ) -> None:
        self.topk_indices_buffer = topk_indices_buffer


class SparseMLAFLImpl(FlashMLASparseImpl):
    """Sparse MLA implementation with GLM sharing and BW1000 kernel support.

    All metadata construction, logical-to-physical index conversion, and
    higher-level sparse-attention control flow remain in the upstream
    ``FlashMLASparseImpl``.  This subclass changes only the constructor bridge
    needed by shared Indexer layers and the final BF16 kernel invocation.
    """
    def __init__(
        self,
        *args,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer=None,
        **kwargs,
    ) -> None:
        # Backport the semantic change from vLLM PR #45895:
        #
        #     indexer.topk_indices_buffer
        #         if indexer is not None
        #         else topk_indices_buffer
        #
        # Reuse the complete vLLM 0.20.x constructor instead of
        # copying its implementation.
        if indexer is None:
            if topk_indices_buffer is None:
                raise RuntimeError(
                    "Sparse MLA requires either a physical "
                    "Indexer or topk_indices_buffer."
                )

            upstream_indexer_arg = _TopKBufferRef(
                topk_indices_buffer
            )
        else:
            upstream_indexer_arg = indexer

        super().__init__(
            *args,
            topk_indices_buffer=topk_indices_buffer,
            indexer=upstream_indexer_arg,
            **kwargs,
        )

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        _configure_hygon_flashmla_sparse()
        num_tokens = q.shape[0]

        kv = kv_c_and_k_pe_cache.view(
            -1,
            1,
            kv_c_and_k_pe_cache.shape[-1],
        )

        if self.num_heads % self.prefill_padding != 0:
            assert (
                self.prefill_padding % self.num_heads == 0
            )

            q_padded = q.new_zeros(
                (
                    q.shape[0],
                    self.prefill_padding,
                    q.shape[2],
                )
            )

            q_padded[
                :, : self.num_heads, :
            ] = q

            q = q_padded

        topk_indices = topk_indices.view(
            num_tokens,
            1,
            -1,
        )

        out = torch.empty(
            (
                num_tokens,
                q.shape[1],
                self.kv_lora_rank,
            ),
            dtype=q.dtype,
            device=q.device,
        )

        out, _, _ = _flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=topk_indices,
            sm_scale=self.softmax_scale,
            attn_sink=None,
            topk_length=None,
            out=out,
        )

        return out[:, : self.num_heads, :]


class SparseMLAFLBackend(FlashMLASparseBackend):
    supported_kv_cache_dtypes: ClassVar[
        list[CacheDType]
    ] = [
        "auto",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "FL_SPARSE_MLA"

    @staticmethod
    def get_impl_cls():
        return SparseMLAFLImpl

    @classmethod
    def supports_compute_capability(
        cls,
        capability: DeviceCapability,
    ) -> bool:
        return True
