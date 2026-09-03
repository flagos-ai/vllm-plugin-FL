# Copyright (c) 2026 BAAI. All rights reserved.

"""
Hygon backend implementation.

Hygon devices in this environment are exposed through a ROCm/HIP-compatible
runtime stack. Some Python layers may still use CUDA-style compatibility entry
points such as ``torch.cuda`` on ROCm builds, but this backend should be treated
as Hygon/ROCm rather than NVIDIA CUDA.

This backend only performs platform detection for now; concrete operator methods
should be added once profiling identifies which FlagGems replacements need a
Hygon-specific fallback or override.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class HygonBackend(Backend):
    """Hygon backend for vendor-specific operator implementations."""

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "hygon"

    @property
    def vendor(self) -> Optional[str]:
        return "hygon"

    def is_available(self) -> bool:
        """
        Check whether the current runtime appears to be a Hygon environment.

        Detection follows the same loose style as the other vendor backends:
        prefer vLLM platform metadata when available, then fall back to the
        environment hints already used by ``vllm_fl.utils``.
        """
        if HygonBackend._available is not None:
            return HygonBackend._available

        try:
            from vllm.platforms import current_platform

            vendor_name = getattr(current_platform, "vendor_name", None)
            if isinstance(vendor_name, str) and vendor_name.lower() == "hygon":
                HygonBackend._available = True
                return True
        except Exception:
            pass

        gems_vendor = os.environ.get("GEMS_VENDOR", "").strip().lower()
        if gems_vendor == "hygon":
            HygonBackend._available = True
            return True

        # Hygon deployments in this project expose both management tools.
        if shutil.which("hy-smi") and shutil.which("rocm-smi"):
            HygonBackend._available = True
            return True

        HygonBackend._available = False
        return HygonBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        from .impl.activation import silu_and_mul_hygon

        return silu_and_mul_hygon(obj, x)

    def gelu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        from .impl.activation import gelu_and_mul_hygon

        return gelu_and_mul_hygon(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.normalization import rms_norm_hygon

        return rms_norm_hygon(obj, x, residual)

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
        from .impl.rotary import rotary_embedding_hygon

        return rotary_embedding_hygon(
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
        Get the vLLM native attention backend path for Hygon.

        Hygon deployments use a HIP/ROCm-compatible stack, so the vendor
        fallback should follow the adapted vLLM ROCm platform priorities rather
        than CUDA FlashAttention paths.
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if use_mla:
            if use_sparse:
                return AttentionBackendEnum.ROCM_AITER_MLA_SPARSE.get_path()

            from vllm._aiter_ops import rocm_aiter_ops

            if rocm_aiter_ops.is_mla_enabled():
                return AttentionBackendEnum.ROCM_AITER_MLA.get_path()
            return AttentionBackendEnum.TRITON_MLA.get_path()

        if use_sparse:
            raise ValueError("use_sparse=True requires use_mla=True.")

        return AttentionBackendEnum.ROCM_ATTN.get_path()

    def moe_align_block_size(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: Optional[torch.Tensor] = None,
        pad_sorted_ids: bool = False,
        ignore_invalid_experts: bool = False,
    ):
        from .impl.fused_moe import moe_align_block_size_hygon

        return moe_align_block_size_hygon(
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            pad_sorted_ids,
            ignore_invalid_experts,
        )

    def moe_sum(self, inp, out):
        from .impl.fused_moe import moe_sum_hygon

        moe_sum_hygon(inp, out)

    def topk_softmax(
        self,
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
    ):
        from .impl.fused_moe import topk_softmax_hygon

        return topk_softmax_hygon(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
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
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape=None,
        B_bias=None,
    ):
        from .impl.fused_moe import invoke_fused_moe_triton_kernel_hygon

        invoke_fused_moe_triton_kernel_hygon(
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
        from .impl.fused_moe import grouped_topk_hygon

        return grouped_topk_hygon(
            scores,
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            bias,
            scoring_func,
        )
