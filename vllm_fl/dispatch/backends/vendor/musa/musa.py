# Copyright (c) 2026 BAAI. All rights reserved.

"""
MUSA backend implementation.

This backend provides operator implementations for Musa GPUs.
Musa uses a CUDA-compatible architecture.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class MusaBackend(Backend):
    """
    Musa backend for operator implementations.

    This backend uses Musa libraries to provide high-performance
    operator implementations for Musa GPUs.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "musa"

    @property
    def vendor(self) -> Optional[str]:
        return "musa"

    def is_available(self) -> bool:
        """
        Check if Musa hardware and libraries are available.

        This method uses the platform's vendor information to determine
        if the device is an Musa GPU.
        """
        return torch.musa.is_available()

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
        from .impl.activation import silu_and_mul_musa

        return silu_and_mul_musa(obj, x)

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
        from .impl.normalization import rms_norm_musa

        return rms_norm_musa(obj, x, residual)

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
        from .impl.rotary import rotary_embedding_musa

        return rotary_embedding_musa(
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
        Get the attention backend class path for Musa.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use sparse attention

        Returns:
            Fully qualified class path string
        """
        from vllm.attention.backends.registry import AttentionBackendEnum

        from vllm_fl.dispatch.backends.flaggems.impl.custom_attention import (
            register_attention,
        )
        register_attention()

        if use_mla:
            return AttentionBackendEnum.TRITON_MLA.get_path()

        return AttentionBackendEnum.TRITON_ATTN.get_path()
