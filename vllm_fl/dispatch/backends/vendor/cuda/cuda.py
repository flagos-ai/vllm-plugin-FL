# Copyright (c) 2026 BAAI. All rights reserved.

"""
CUDA backend implementation.

This backend provides operator implementations for NVIDIA CUDA GPUs.
"""

from __future__ import annotations

from typing import Optional

import torch

from vllm_fl.dispatch.backends.base import Backend


class CudaBackend(Backend):
    """
    CUDA backend for operator implementations.

    This backend uses CUDA libraries to provide high-performance
    operator implementations for NVIDIA GPUs.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def vendor(self) -> Optional[str]:
        return "nvidia"

    def is_available(self) -> bool:
        """
        Check if CUDA hardware and libraries are available.

        This method uses the platform's vendor information from FlagGems
        to determine if the device is a real NVIDIA GPU, decoupling from
        CUDA-alike devices (MACA, MUSA, etc.) which have their own vendor names.
        """
        if CudaBackend._available is None:
            try:
                # Check if CUDA device is available
                if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                    CudaBackend._available = False
                    return False

                from vllm.platforms import current_platform
                if hasattr(current_platform, 'vendor_name') and current_platform.vendor_name == "nvidia":
                    CudaBackend._available = True
                else:
                    CudaBackend._available = False
            except Exception:
                CudaBackend._available = False
        return CudaBackend._available

    # ==================== Operator Implementations ====================

    def attention_backend(self, use_mla: bool = False) -> str:
        """
        Get the attention backend class path for CUDA.

        Supports:
        - FLASH_ATTN (default)
        - TRITON_ATTN (when use_flaggems_op("triton_attn") is True)

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string
        """
        from vllm.attention.backends.registry import AttentionBackendEnum
        from vllm_fl.utils import use_flaggems_op

        if use_mla:
            return AttentionBackendEnum.FLASHMLA.get_path()

        # Default to FLASH_ATTN
        return AttentionBackendEnum.FLASH_ATTN.get_path()
