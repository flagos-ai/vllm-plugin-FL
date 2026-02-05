# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference backend implementation using PyTorch.

This backend provides reference operator implementations using native PyTorch
operations. These implementations are always available when PyTorch is installed
and serve as fallback implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class ReferenceBackend(Backend):
    """
    Reference backend for operator implementations.

    This backend uses native PyTorch operations to provide reference
    implementations that are always available as fallbacks.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "reference"

    def is_available(self) -> bool:
        """Check if PyTorch is available."""
        if ReferenceBackend._available is None:
            try:
                import torch

                ReferenceBackend._available = True
            except ImportError:
                ReferenceBackend._available = False
        return ReferenceBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_torch

        return silu_and_mul_torch(x)

    def rms_norm(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        weight: torch.Tensor,
        epsilon: float,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization.

        Args:
            x: Input tensor
            residual: Optional residual tensor
            weight: Normalization weight
            epsilon: Small constant for numerical stability

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        from .impl.normalization import rms_norm_torch

        return rms_norm_torch(x, residual, weight, epsilon)

    def rotary_embedding(
        self,
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
            query: Query tensor
            key: Key tensor
            cos: Cosine cache
            sin: Sine cache
            position_ids: Position indices
            rotary_interleaved: Whether to use interleaved rotary
            inplace: Whether to modify tensors in-place (ignored in reference impl)

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        from .impl.rotary import rotary_embedding_torch

        return rotary_embedding_torch(
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
        Get the attention backend class path for reference (vLLM native).

        This method returns the vLLM native flash attention backend path,
        which serves as a fallback implementation.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string (vLLM native backend)
        """
        # Return vLLM's native flash attention backend as reference
        from vllm.attention.backends.registry import AttentionBackendEnum

        if use_mla:
            # vLLM native MLA backend
            return AttentionBackendEnum.FLASHMLA.get_path()
        return AttentionBackendEnum.FLASH_ATTN.get_path()
