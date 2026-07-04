# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX backend implementation.

This backend provides operator implementations for METAX GPUs.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend

from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend


# Register attention backends for MACA
def register_attention_backends():
    register_backend(
        AttentionBackendEnum.FLASHMLA,
        class_path="vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla.flashmla.MacaFlashMLABackend",
    )
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        class_path="vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn.MacaFlashAttentionBackend",
    )


class MacaBackend(Backend):
    """
    METAX backend for operator implementations.

    This backend uses MACA libraries to provide high-performance
    operator implementations for METAX GPUs.
    """

    _available: bool | None = None

    @property
    def name(self) -> str:
        return "maca"

    @property
    def vendor(self) -> Optional[str]:
        return "metax"

    def is_available(self) -> bool:
        """Check if Metax hardware and libraries are available."""
        if MacaBackend._available is None:
            try:
                # Check if Metax device is available
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    MacaBackend._available = True
                else:
                    MacaBackend._available = False
            except Exception:
                MacaBackend._available = False
        return MacaBackend._available

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
        from .impl.activation import silu_and_mul_maca

        return silu_and_mul_maca(obj, x)

    def gelu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        GELU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import gelu_and_mul_maca

        return gelu_and_mul_maca(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization using Maca's CUDA implementation.
        """
        from .impl.layernorm import rms_norm_maca

        return rms_norm_maca(obj, x, residual)

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
        Apply rotary position embedding using vLLM's CUDA implementation.
        """
        from .impl.rotary_embedding import rotary_embedding_maca

        return rotary_embedding_maca(
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
        Get the attention backend class path for CUDA.

        Supports:
        - FLASH_ATTN (default)
        - TRITON_ATTN (when use_flaggems_op("triton_attn") is True)
        - FLASHMLA_SPARSE (when use_mla and use_sparse are both True)

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        # register before selection
        register_attention_backends()

        if use_mla:
            if use_sparse:
                return AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
            return AttentionBackendEnum.FLASHMLA.get_path()

        # Default to FLASH_ATTN
        return AttentionBackendEnum.FLASH_ATTN.get_path()

    def topk_softmax(
        self,
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
    ):
        from .impl.fused_moe import topk_softmax_maca

        return topk_softmax_maca(
            topk_weights, topk_indices, token_expert_indices, gating_output, renormalize
        )

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
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
        B_bias=None,
    ):
        from .impl.fused_moe import invoke_fused_moe_triton_kernel_maca

        invoke_fused_moe_triton_kernel_maca(
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
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=B_bias,
        )
