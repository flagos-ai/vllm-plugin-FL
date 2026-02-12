# Copyright (c) 2026 BAAI. All rights reserved.

"""
CUDA backend implementation.

This backend provides operator implementations for NVIDIA CUDA GPUs.
"""

from __future__ import annotations

from typing import Optional, Union

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

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Uses vLLM's native CUDA implementation.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_cuda

        return silu_and_mul_cuda(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization using vLLM's CUDA implementation.

        Args:
            obj: The calling obj (e.g., RMSNorm layer)
            x: Input tensor
            residual: Optional residual tensor

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        from .impl.normalization import rms_norm_cuda

        return rms_norm_cuda(obj, x, residual)

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
        from .impl.rotary import rotary_embedding_cuda

        return rotary_embedding_cuda(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

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
