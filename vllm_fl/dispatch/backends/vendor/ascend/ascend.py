# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend backend implementation.

This backend provides operator implementations for Huawei Ascend NPUs.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class AscendBackend(Backend):
    """
    Ascend backend for operator implementations.

    This backend uses Ascend CANN libraries to provide high-performance
    operator implementations for Huawei Ascend NPUs.
    """

    _available: Optional[bool] = None

    def __init__(self) -> None:
        super().__init__()

    @property
    def name(self) -> str:
        return "ascend"

    @property
    def vendor(self) -> Optional[str]:
        return "ascend"

    def is_available(self) -> bool:
        """Check if Ascend hardware and libraries are available."""
        if AscendBackend._available is None:
            # Check if NPU device is available
            if torch.npu.is_available() and torch.npu.device_count() > 0:
                AscendBackend._available = True
            else:
                AscendBackend._available = False
        return AscendBackend._available

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
        from .impl.activation import silu_and_mul_ascend

        return silu_and_mul_ascend(obj, x)

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
        from .impl.normalization import rms_norm_ascend

        return rms_norm_ascend(obj, x, residual)

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
            inplace: Whether to modify tensors in-place

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        from .impl.rotary import rotary_embedding_ascend

        return rotary_embedding_ascend(
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
        Get the attention backend class path for Ascend NPU.

        This method returns the native Ascend attention backend that uses
        torch_npu operators (npu_fused_infer_attention_score, etc.)
        instead of flag_gems operators.

        Uses vllm_fl's native Ascend implementation which directly calls
        torch_npu operators without depending on vllm-ascend package.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string
        """
        if use_mla:
            if use_sparse:
                raise NotImplementedError("MLA with sparse attention is not implemented for Ascend yet.")
            return "vllm_fl.dispatch.backends.vendor.ascend.impl.attention.AscendMLABackend"
        return "vllm_fl.dispatch.backends.vendor.ascend.impl.attention.AscendAttentionBackend"

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
        compute_type=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        block_shape=None,
        B_bias=None,
    ):
        """Ascend NPU fused MoE kernel using torch.mm.

        Replaces the FlagGems Triton kernel which overflows the NPU's
        unified buffer on certain model shapes.
        """
        from .impl.fused_moe_kernel import invoke_fused_moe_torch
        invoke_fused_moe_torch(
            A, B, C, A_scale, B_scale, topk_weights,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            mul_routed_weight, top_k, config,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            B_bias=B_bias,
        )

    def moe_align_block_size(
        self,
        topk_ids,
        block_size,
        num_experts,
        expert_map=None,
        pad_sorted_ids=False,
        ignore_invalid_experts=False,
    ):
        """Pure-torch moe_align_block_size for Ascend NPU.

        Replaces the FlagGems Triton kernel which causes DDR address OOB
        errors on Ascend NPU hardware.
        """
        from .impl.fused_moe_kernel import moe_align_block_size_torch
        return moe_align_block_size_torch(
            topk_ids, block_size, num_experts, expert_map,
            pad_sorted_ids, ignore_invalid_experts,
        )

    def moe_sum(self, inp, out):
        """Pure-torch moe_sum: sum over top_k dimension."""
        # inp is (M, top_k, N), out is (M, N)
        # Avoid out= parameter which can cause NPU issues
        result = inp.sum(dim=1)
        out.copy_(result)

    def topk_softmax(
        self, topk_weights, topk_indices, token_expert_indices, gating_output,
        renormalize=False,
    ):
        """Pure-torch topk_softmax for Ascend NPU."""
        scores = torch.softmax(gating_output.float(), dim=-1)
        topk = topk_weights.shape[1]
        tk_weights, tk_indices = torch.topk(scores, k=topk, dim=-1)
        topk_weights.copy_(tk_weights.to(topk_weights.dtype))
        topk_indices.copy_(tk_indices.to(topk_indices.dtype))
        if renormalize:
            s = topk_weights.sum(dim=-1, keepdim=True)
            topk_weights.div_(s.clamp(min=1e-8))
        return topk_weights, topk_indices
