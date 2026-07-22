# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference backend implementation using PyTorch.

This backend provides reference operator implementations using native PyTorch
operations. These implementations are always available when PyTorch is installed
and serve as fallback implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm.logger import init_logger
from vllm_fl.dispatch.backends.base import Backend

logger = init_logger(__name__)

_metax_mla_patched = False


def _patch_flash_attn_for_metax():
    """
    On MetaX platform, patch vLLM's MLA modules for MACA compatibility:

    1. flash_attn_varlen_func: vLLM imports from vllm.vllm_flash_attn (CUDA C
       extension) which is unavailable on MetaX. Patch in the MACA-adapted
       flash_attn package version for prefill.

    2. decode_attention_fwd: vLLM's native Triton decode kernel uses block sizes
       that exceed MetaX's 64KB shared memory limit. Patch in the MetaX-adapted
       version with reduced block sizes.
    """
    global _metax_mla_patched
    if _metax_mla_patched:
        return

    from vllm.platforms import current_platform
    if current_platform.vendor_name != "metax":
        _metax_mla_patched = True
        return

    # --- Patch 1: flash_attn_varlen_func for prefill ---
    import vllm.model_executor.layers.attention.mla_attention as mla_mod

    if mla_mod.flash_attn_varlen_func is None:
        try:
            from flash_attn import flash_attn_varlen_func
        except ImportError as e:
            raise RuntimeError(
                "MetaX platform requires flash_attn package for MLA prefill. "
                "Please install the MACA-adapted flash_attn."
            ) from e

        mla_mod.flash_attn_varlen_func = flash_attn_varlen_func
        mla_mod.is_vllm_fa = False
        logger.info("Patched flash_attn_varlen_func from flash_attn package "
                    "for MetaX MLA prefill support")

    # --- Patch 2: decode_attention_fwd for decode (shmem limit) ---
    import vllm.v1.attention.backends.mla.triton_mla as triton_mla_mod

    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.ops.triton_decode_attention import (
        decode_attention_fwd as metax_decode_attention_fwd,
    )
    triton_mla_mod.decode_attention_fwd = metax_decode_attention_fwd
    logger.info("Patched decode_attention_fwd with MetaX version "
                "(reduced block size for 64KB shmem limit)")

    _metax_mla_patched = True


class ReferenceBackend(Backend):
    """
    Reference backend for operator implementations.

    This backend uses native PyTorch operations to provide reference
    implementations that are always available as fallbacks.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "reference"

    def is_available(self) -> bool:
        """Check if PyTorch is available."""
        if ReferenceBackend._available is None:
            try:
                import torch

                ReferenceBackend._available = True
            except ImportError:
                ReferenceBackend._available = False
        return ReferenceBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_torch

        return silu_and_mul_torch(obj, x)

    def gelu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        GELU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import gelu_and_mul_torch

        return gelu_and_mul_torch(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization.

        Args:
            obj: The calling obj (e.g., RMSNorm layer)
            x: Input tensor
            residual: Optional residual tensor

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        from .impl.normalization import rms_norm_torch

        return rms_norm_torch(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding.

        Args:
            obj: The calling obj (for interface consistency)
            query: Query tensor
            key: Key tensor
            cos: Cosine cache
            sin: Sine cache
            position_ids: Position indices
            rotary_interleaved: Whether to use interleaved rotary
            inplace: Whether to modify tensors in-place (ignored in reference impl)

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        from .impl.rotary import rotary_embedding_torch

        return rotary_embedding_torch(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for reference (vLLM native).

        This method returns the vLLM native flash attention backend path,
        which serves as a fallback implementation.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string (vLLM native backend)
        """
        # Return vLLM's native attention backend as reference
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if use_mla:
            _patch_flash_attn_for_metax()
            logger.info("attention backend reference dispatch: "
                        "using TritonMLA for MLA attention")
            return AttentionBackendEnum.TRITON_MLA.get_path()
        return AttentionBackendEnum.FLASH_ATTN.get_path()

    def invoke_fused_moe_triton_kernel(
        self,
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape=None,
        B_bias=None,
    ):
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            invoke_fused_moe_triton_kernel,
        )

        invoke_fused_moe_triton_kernel(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape=block_shape,
            B_bias=B_bias,
        )

    def moe_align_block_size(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map=None,
        pad_sorted_ids: bool = False,
        ignore_invalid_experts: bool = False,
    ):
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )

        return moe_align_block_size(
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            pad_sorted_ids,
            ignore_invalid_experts,
        )

    def moe_sum(self, inp, out):
        from vllm._custom_ops import moe_sum

        moe_sum(inp, out)

    def topk_softmax(
        self,
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
    ):
        from vllm._custom_ops import topk_softmax

        return topk_softmax(
            topk_weights, topk_indices, token_expert_indices, gating_output, renormalize
        )

    def grouped_topk(
        self,
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func=0,
    ):
        from vllm._custom_ops import grouped_topk

        return grouped_topk(
            scores, n_group, topk_group, topk,
            renormalize, routed_scaling_factor, bias, scoring_func,
        )
