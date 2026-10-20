# Copyright (c) 2026 BAAI. All rights reserved.

"""
Sunrise backend implementation.

This backend provides operator implementations for Sunrise GPUs.

"""

from __future__ import annotations

from typing import Optional

import torch

from vllm_fl.dispatch.backends.base import Backend


class SunriseBackend(Backend):
    """
    Sunrise backend for operator implementations.

    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "sunrise"

    @property
    def vendor(self) -> Optional[str]:
        return "sunrise"

    def is_available(self) -> bool:
        """
        Check if the sunrise hardware and libraries are available.

        Implement this method to detect if your vendor's hardware/software
        is present and functional.
        """
        if SunriseBackend._available is None:
            # Check if Sunrise device is available
            if torch.ptpu.is_available() and torch.ptpu.device_count() > 0:
                SunriseBackend._available = True
            else:
                SunriseBackend._available = False
        return SunriseBackend._available

    # ==================== Operator Implementations ====================
    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """SiLU + element-wise multiply backed by ``torch_ptpu.sgl_kernel``."""
        from .impl.activation import silu_and_mul_sunrise

        return silu_and_mul_sunrise(obj, x)

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
        """Apply RoPE using ``torch_ptpu.sgl_kernel.apply_rope_with_cos_sin_cache``."""
        from .impl.rotary import rotary_embedding_sunrise

        return rotary_embedding_sunrise(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def topk_softmax(
        self,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Router top-k over ``softmax(gating_output)`` (PyTorch fallback)."""
        from .impl.fused_moe import topk_softmax_sunrise

        return topk_softmax_sunrise(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
        )

    def moe_sum(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        """Sum across the top-k expert axis (PyTorch fallback)."""
        from .impl.fused_moe import moe_sum_sunrise

        moe_sum_sunrise(inp, out)

    def moe_align_block_size(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: Optional[torch.Tensor] = None,
        pad_sorted_ids: bool = False,
        ignore_invalid_experts: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align MoE token blocks (PyTorch fallback; FlagGems stays default)."""
        from .impl.fused_moe import moe_align_block_size_sunrise

        return moe_align_block_size_sunrise(
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            pad_sorted_ids,
            ignore_invalid_experts,
        )

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for Sunrise.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string
        """
        if use_sparse:
            raise NotImplementedError(
                "Sunrise does not support sparse MLA (DSA) yet; "
                "use_sparse=True requires a dedicated backend."
            )

        if use_mla:
            return (
                "vllm_fl.dispatch.backends.vendor.sunrise.impl.mla.SunriseMLABackend"
            )

        return "vllm_fl.dispatch.backends.vendor.sunrise.impl.attention.AttentionFLBackend"
