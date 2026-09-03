# Copyright (c) 2026 BAAI. All rights reserved.

"""
Thead backend implementation.

This backend provides operator implementations for T-Head PPU accelerators.
For attention, it uses the flash_attn_3 wheel (FA3) for better performance
on CC 8.0 devices.
"""

from __future__ import annotations

from typing import Optional

import torch

from vllm_fl.dispatch.backends.base import Backend


class TheadBackend(Backend):
    """
    Thead (PPU) backend for operator implementations.

    This backend uses the PPU FA3 kernel (flash_attn_3 wheel) for attention,
    and vLLM native CUDA implementations for other ops (silu_and_mul, rms_norm,
    rotary_embedding).
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "thead"

    @property
    def vendor(self) -> Optional[str]:
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
        from vllm.model_executor.layers.activations import silu_and_mul
        return silu_and_mul(x)

    def rms_norm(self, obj, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        """RMS normalization."""
        if residual is not None:
            from vllm.model_executor.layers.layernorm import rms_norm
            return rms_norm(x, obj.weight, residual)
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=obj.weight)

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
        """Apply rotary position embedding."""
        from vllm.model_executor.layers.rotary_embedding import apply_rotary_emb
        return apply_rotary_emb(
            query, key, cos, sin,
            position_ids=position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(
        self, use_mla: bool = False, use_sparse: bool = False
    ) -> str:
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
