# Copyright (c) 2026 BAAI. All rights reserved.

"""
ILUVATAR backend implementation.

This backend provides operator implementations for Iluvatar GPUs.
Iluvatar uses a CUDA-compatible architecture.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class IluvatarBackend(Backend):
    """
    Iluvatar backend for operator implementations.

    This backend uses Iluvatar libraries to provide high-performance
    operator implementations for Iluvatar GPUs.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "iluvatar"

    @property
    def vendor(self) -> Optional[str]:
        return "iluvatar"

    def is_available(self) -> bool:
        """
        Check if Iluvatar hardware and libraries are available.

        This method uses the platform's vendor information to determine
        if the device is an Iluvatar GPU.
        """
        if IluvatarBackend._available is None:
            try:
                from vllm.platforms import current_platform
                # Iluvatar GPUs should be detected via vendor_name
                if hasattr(current_platform, 'vendor_name') and current_platform.vendor_name == "iluvatar":
                    IluvatarBackend._available = True
                else:
                    # Fallback: check if CUDA is available with iluvatar device
                    if torch.cuda.is_available():
                        # Try to detect Iluvatar GPU
                        # Iluvatar GPUs typically expose CUDA-compatible interface
                        # We can check device name if available
                        device_name = torch.cuda.get_device_name(0)
                        if "iluvatar" in device_name.lower():
                            IluvatarBackend._available = True
                        else:
                            IluvatarBackend._available = False
    
                    else:
                        IluvatarBackend._available = False
            except Exception:
                IluvatarBackend._available = False
        return IluvatarBackend._available

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
        from .impl.activation import silu_and_mul_iluvatar

        return silu_and_mul_iluvatar(obj, x)

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
        from .impl.normalization import rms_norm_iluvatar

        return rms_norm_iluvatar(obj, x, residual)

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
        from .impl.rotary import rotary_embedding_iluvatar

        return rotary_embedding_iluvatar(
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
        Get the attention backend class path for Iluvatar.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string
        """
        from vllm.attention.backends.registry import AttentionBackendEnum

        if use_mla:
            return AttentionBackendEnum.FLASHMLA.get_path()

        return AttentionBackendEnum.FLASH_ATTN.get_path()