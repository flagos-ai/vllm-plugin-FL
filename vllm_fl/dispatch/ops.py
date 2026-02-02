# Copyright (c) 2026 BAAI. All rights reserved.

"""
Backend base interface definitions for vllm-plugin-FL dispatch.

This module defines the abstract base class that all backends must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch


class VLLMFLBackendBase(ABC):
    """
    Abstract base class for vllm-plugin-FL operator backends.

    Each backend provides implementations for a set of operators.
    Backends should implement is_available() to indicate whether
    the backend can be used in the current environment.

    All operator methods should be implemented by concrete backend classes.
    Methods that are not supported should raise NotImplementedError.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend is available in the current environment.

        Returns:
            True if the backend can be used, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of this backend.

        Returns:
            Backend name string.
        """
        pass

    @property
    def vendor(self) -> Optional[str]:
        """
        Get the vendor name for this backend (if applicable).

        Returns:
            Vendor name string, or None for non-vendor backends.
        """
        return None

    # ==================== Activation Operators ====================

    @abstractmethod
    def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        pass

    # ==================== Normalization Operators ====================

    @abstractmethod
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
        pass

    # ==================== Position Embedding Operators ====================

    @abstractmethod
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
        pass

    # ==================== Attention Backend ====================

    @abstractmethod
    def attention_backend(self, use_mla: bool = False) -> str:
        """
        Get the attention backend class path for this platform.

        This method returns the fully qualified class path of the attention
        backend implementation suitable for the current hardware platform.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string, e.g.:
            - "vllm_fl.dispatch.backends.vendor.ascend.impl.attention.AscendAttentionBackend"
            - "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"
        """
        pass
