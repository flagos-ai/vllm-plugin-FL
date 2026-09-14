# Copyright (c) 2026 BAAI. All rights reserved.
"""Sparse MLA (DSA) attention backend backed by FlagGems Triton kernels.

GLM-5-Next's MLA layers run sparse attention: a kpool indexer selects
``index_topk`` KV tokens per query and MLA attends over just those. vLLM ships
five sparse-MLA backends and every one of them needs a compiled extension that
does not exist on this platform -- ``FLASHMLA_SPARSE`` and ``FLASHMLA`` want
``vllm._flashmla_C``, the FlashInfer/FA variants want their own CUDA libs, and
the ROCm one wants AITER. This vLLM build is ``VLLM_TARGET_DEVICE=empty`` (pure
Python, zero ``.so``), so on DCU the dispatch chain fell all the way through to
``NotImplementedError``.

Everything about ``FlashMLASparseBackend`` other than the kernel call itself is
portable: the metadata builder is plain torch, and
``triton_convert_req_index_to_global_index`` (``mla/sparse_utils.py``) is
already Triton. So this backend subclasses the upstream classes and overrides
exactly one method -- ``_bf16_flash_mla_kernel`` -- to call
``flag_gems.flash_mla_sparse_fwd`` instead of the FlashMLA CUDA kernel. The
FlagGems entry point takes the same ``(q, kv, indices, sm_scale, ...)``
arguments and returns the same ``(out, lse, ...)`` tuple.

BF16 KV cache only. The upstream FP8 (``fp8_ds_mla``) paths call
``get_mla_metadata`` / ``cp_gather_and_upconvert_fp8_kv_cache``, which are
FlashMLA-specific; ``supported_kv_cache_dtypes`` below excludes them so the
engine rejects that config up front instead of failing inside a kernel.
"""

from typing import ClassVar

import torch

from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadataBuilder,
)

logger = init_logger(__name__)


class SparseMLAFLImpl(FlashMLASparseImpl):
    """FlashMLA sparse impl with the BF16 kernel swapped for FlagGems Triton."""

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None,
    ) -> torch.Tensor:
        from flag_gems import flash_mla_sparse_fwd

        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # The kernel wants num_heads to be a multiple of the tile width; pad
        # and slice back afterwards.
        #
        # MUST be new_zeros, not new_empty. Upstream's flashmla_sparse.py:823
        # uses new_empty here, which leaves the padded heads reading whatever
        # the caching allocator last left in that block. On a freshly-started
        # process those bytes happen to be zero, so the bug hides; once the
        # allocator starts recycling, the padded heads carry live garbage
        # (measured absmax ~1.2e4 against a real-q rms of ~1.0). At TP16 this
        # model has num_heads = 64/16 = 4 padded to 64, i.e. 93.75% of the q
        # tensor handed to the kernel would be garbage. Zeros are safe: they
        # produce a uniform attention row whose output we slice away below.
        # (Upstream itself uses new_zeros for the same purpose at line 781.)
        #
        # Read the head count from the tensor rather than self.num_heads so a
        # mismatch pads the tensor that is actually passed to the kernel.
        actual_heads = q.shape[1]
        if actual_heads % self.prefill_padding != 0:
            assert self.prefill_padding % actual_heads == 0
            logger.warning_once(
                "Padding num_heads from %d to %d for the BF16 sparse MLA kernel",
                actual_heads,
                self.prefill_padding,
            )
            q_padded = q.new_zeros((q.shape[0], self.prefill_padding, q.shape[2]))
            q_padded[:, :actual_heads, :] = q
            q = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        # GLM-5-Next is NoPE (qk_rope_head_dim == 0): the kernel scores only the
        # NoPE part (d_qk == d_v == kv_lora_rank), so the softmax scale must be
        # over kv_lora_rank rather than the with-rope head size.
        sm_scale = (
            self.kv_lora_rank**-0.5
            if self.qk_rope_head_dim == 0
            else self.softmax_scale
        )
        output = flash_mla_sparse_fwd(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            sm_scale,
            topk_length=topk_length,
        )[0]

        return output[:, :actual_heads, :]


class SparseMLAFLBackend(FlashMLASparseBackend):
    """Sparse MLA backend for FlagOS platforms (no FlashMLA extension)."""

    # BF16 only -- see module docstring for why fp8_ds_mla is excluded.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "SPARSE_MLA_FL"

    @staticmethod
    def get_impl_cls() -> type[SparseMLAFLImpl]:
        return SparseMLAFLImpl

    @staticmethod
    def get_builder_cls() -> type[FlashMLASparseMetadataBuilder]:
        return FlashMLASparseMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Upstream restricts to SM90/SM100 because the FlashMLA kernel is
        # hardware-specific. The FlagGems kernel is Triton, so drop the gate.
        return True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # Upstream pins 64 (FlashMLA's tile). GLM-5-Next's kpool indexer needs
        # block_size to be a multiple of index_kpool * 32 = 128, and the Triton
        # kernel has no such tile constraint, so accept both.
        return [64, 128]
