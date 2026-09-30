# Copyright (c) 2026 BAAI. All rights reserved.

"""
Thead backend implementation.

This backend provides operator implementations for T-Head PPU accelerators.
For attention, it uses the flash_attn_3 wheel (FA3) for better performance
on CC 8.0 devices.
"""

from __future__ import annotations

import torch

from vllm_fl.dispatch.backends.base import Backend


class TheadBackend(Backend):
    """
    Thead (PPU) backend for operator implementations.

    This backend uses the PPU FA3 kernel (flash_attn_3 wheel) for attention,
    and PyTorch implementations for other ops (silu_and_mul, rms_norm,
    rotary_embedding) without requiring NVIDIA C extensions.
    """

    _available: bool | None = None

    @property
    def name(self) -> str:
        return "thead"

    @property
    def vendor(self) -> str | None:
        return "thead"

    def is_available(self) -> bool:
        """
        Check if thead (PPU) hardware is available.

        Detection is based on the PPU_SDK environment variable
        (same logic as FlagGems DeviceDetector).
        """
        if TheadBackend._available is None:
            try:
                if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                    TheadBackend._available = False
                    return False

                from vllm.platforms import current_platform

                vendor_name = getattr(current_platform, "vendor_name", None)
                if vendor_name == "thead":
                    TheadBackend._available = True
                else:
                    # Fallback: check PPU_SDK env var
                    import os

                    TheadBackend._available = "PPU_SDK" in os.environ
            except Exception:
                TheadBackend._available = False
        return TheadBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """SiLU activation followed by element-wise multiplication."""
        from vllm_fl.dispatch.backends.reference.impl.activation import (
            silu_and_mul_torch,
        )

        return silu_and_mul_torch(obj, x)

    def rms_norm(self, obj, x: torch.Tensor, residual: torch.Tensor | None = None):
        """RMS normalization with the FL layer's epsilon and residual contract."""
        input_dtype = x.dtype
        x_float = x.float()
        if residual is not None:
            x_float = x_float + residual.float()
            residual = x_float.to(input_dtype)

        variance_size = getattr(obj, "variance_size_override", None)
        variance_input = (
            x_float if variance_size is None else x_float[..., :variance_size]
        )
        variance = variance_input.square().mean(dim=-1, keepdim=True)
        output = (x_float * torch.rsqrt(variance + obj.variance_epsilon)).to(
            input_dtype
        )
        pass_weight = getattr(
            obj, "pass_weight_add" if residual is not None else "pass_weight", True
        )
        if pass_weight:
            output = output * obj.weight

        if residual is not None:
            return output, residual
        return output

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
        """Apply rotary embedding using the caches supplied by RotaryEmbeddingFL."""
        del obj
        cos_selected = cos[position_ids]
        sin_selected = sin[position_ids]
        if query.ndim in (3, 4):
            cos_selected = cos_selected.unsqueeze(1)
            sin_selected = sin_selected.unsqueeze(1)

        if cos_selected.shape[-1] * 2 == query.shape[-1]:
            if rotary_interleaved:
                cos_selected = cos_selected.repeat_interleave(2, dim=-1)
                sin_selected = sin_selected.repeat_interleave(2, dim=-1)
            else:
                cos_selected = torch.cat((cos_selected, cos_selected), dim=-1)
                sin_selected = torch.cat((sin_selected, sin_selected), dim=-1)

        def rotate(x):
            if rotary_interleaved:
                return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
            x_first, x_second = x.chunk(2, dim=-1)
            return torch.cat((-x_second, x_first), dim=-1)

        q_embed = query * cos_selected + rotate(query) * sin_selected
        k_embed = key * cos_selected + rotate(key) * sin_selected
        if inplace:
            query.copy_(q_embed)
            key.copy_(k_embed)
            return query, key
        return q_embed, k_embed

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for PPU.

        Returns the TheadFlashAttentionBackend which uses the FA3 wheel.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string
        """
        if use_mla or use_sparse:
            # Fall back to standard FLASH_ATTN for MLA/sparse
            from vllm.v1.attention.backends.registry import AttentionBackendEnum

            return AttentionBackendEnum.FLASH_ATTN.get_path()

        return (
            "vllm_fl.dispatch.backends.vendor.thead.impl.attention."
            "TheadFlashAttentionBackend"
        )
