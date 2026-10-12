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

    This backend provides the PPU FA3 attention bridge. Shared operators use
    the FlagGems and PyTorch reference paths selected by the dispatch policy.
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
