# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems backend implementation.

This backend provides operator implementations using the FlagGems library.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class FlagGemsBackend(Backend):
    """
    FlagGems backend for operator implementations.

    This backend uses the flag_gems library to provide high-performance
    operator implementations.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "flaggems"

    def is_available(self) -> bool:
        """Check if FlagGems is available."""
        if FlagGemsBackend._available is None:
            try:
                import flag_gems

                FlagGemsBackend._available = True
            except ImportError:
                FlagGemsBackend._available = False
        return FlagGemsBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_flaggems

        return silu_and_mul_flaggems(x)

    def rmsnorm(
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
        from .impl.normalization import rmsnorm_flaggems

        return rmsnorm_flaggems(x, residual, weight, epsilon)

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
            inplace: Whether to modify tensors in-place

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        from .impl.rotary import rotary_embedding_flaggems

        return rotary_embedding_flaggems(
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
        Get the attention backend class path for FlagGems.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string
        """
        if use_mla:
            return "vllm_fl.dispatch.backends.flaggems.impl.mla.MLAFLBackend"
        return "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"
