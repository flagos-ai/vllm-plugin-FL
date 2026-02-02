# Copyright (c) 2026 BAAI. All rights reserved.

"""
Attention mask builder for Ascend NPU backend.

This module provides utilities for creating and caching attention masks
required by Ascend NPU attention operators.
"""

from typing import ClassVar, Optional

import torch


class AttentionMaskBuilder:
    """
    Builder for creating and caching attention masks.

    This class manages attention mask creation with caching to avoid
    redundant tensor allocations. The masks are created lazily and
    cached at the class level for reuse across instances.

    Attributes:
        device: The device to create masks on.
    """

    # Class-level cache for masks (shared across all instances)
    _chunked_prefill_mask: ClassVar[Optional[torch.Tensor]] = None
    _chunked_prefill_mask_device: ClassVar[Optional[torch.device]] = None
    _mla_mask: ClassVar[Optional[torch.Tensor]] = None
    _mla_mask_dtype: ClassVar[Optional[torch.dtype]] = None
    _pcp_mla_mask: ClassVar[Optional[torch.Tensor]] = None
    _pcp_mla_mask_dtype: ClassVar[Optional[torch.dtype]] = None

    def __init__(self, device: torch.device):
        """
        Initialize the attention mask builder.

        Args:
            device: The device to create masks on.
        """
        self.device = device

    def get_splitfuse_attn_mask(self) -> torch.Tensor:
        """
        Get attention mask for split-fuse (chunked prefill) attention.

        Creates a 2048x2048 upper triangular mask with int8 dtype.
        The mask is cached and reused for subsequent calls.

        Returns:
            Upper triangular attention mask tensor.
        """
        cls = AttentionMaskBuilder
        if (cls._chunked_prefill_mask is None or
            cls._chunked_prefill_mask_device != self.device):
            cls._chunked_prefill_mask = torch.triu(
                torch.ones(2048, 2048), diagonal=1
            ).to(torch.int8).to(self.device)
            cls._chunked_prefill_mask_device = self.device
        return cls._chunked_prefill_mask

    def get_mla_mask(self, dtype: torch.dtype) -> torch.Tensor:
        """
        Get attention mask for MLA (Multi-head Latent Attention).

        Creates a 512x512 upper triangular mask. For fp16, uses
        float32 min value as mask; otherwise uses 1.

        Args:
            dtype: The dtype for the mask tensor.

        Returns:
            MLA attention mask tensor.
        """
        cls = AttentionMaskBuilder
        if cls._mla_mask is None or cls._mla_mask_dtype != dtype:
            if dtype == torch.float16:
                mask_value = torch.finfo(torch.float32).min
            else:
                mask_value = 1
            prefill_mask = torch.triu(
                torch.ones(512, 512, device=self.device, dtype=dtype), 1
            )
            cls._mla_mask = torch.where(prefill_mask == 1, mask_value, 0).to(dtype)
            cls._mla_mask_dtype = dtype
        return cls._mla_mask

    def get_pcp_mla_mask(self, dtype: torch.dtype) -> torch.Tensor:
        """
        Get attention mask for PCP (Prefill Context Parallel) MLA.

        Creates a 512x512 upper triangular mask.

        Args:
            dtype: The dtype for the mask tensor.

        Returns:
            PCP MLA attention mask tensor.
        """
        cls = AttentionMaskBuilder
        if cls._pcp_mla_mask is None or cls._pcp_mla_mask_dtype != dtype:
            cls._pcp_mla_mask = torch.triu(
                torch.ones(512, 512, device=self.device, dtype=dtype), 1
            )
            cls._pcp_mla_mask_dtype = dtype
        return cls._pcp_mla_mask

    def get_attn_mask(self, max_seq_len: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Get a general attention mask for given sequence length.

        Creates a causal mask (lower triangular) for the given sequence length.

        Args:
            max_seq_len: Maximum sequence length.
            dtype: The dtype for the mask tensor.

        Returns:
            Causal attention mask tensor.
        """
        # Create lower triangle matrix (True for valid positions)
        mask_flag = torch.ones(
            (max_seq_len, max_seq_len), dtype=torch.bool
        ).tril_()
        # Invert to get mask positions (True for masked positions)
        mask_flag = ~mask_flag
        # For fp16, use -inf; otherwise use 1
        mask_value = float('-inf') if dtype == torch.float16 else 1
        attn_mask = torch.zeros(
            size=(max_seq_len, max_seq_len), dtype=dtype
        ).masked_fill_(mask_flag, mask_value)
        return attn_mask.to(self.device)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached masks. Useful for testing or memory cleanup."""
        cls._chunked_prefill_mask = None
        cls._chunked_prefill_mask_device = None
        cls._mla_mask = None
        cls._mla_mask_dtype = None
        cls._pcp_mla_mask = None
        cls._pcp_mla_mask_dtype = None


# Global instance cache for convenience
_builder_instance: Optional[AttentionMaskBuilder] = None
_builder_device: Optional[torch.device] = None


def get_attention_mask_builder(device: torch.device) -> AttentionMaskBuilder:
    """
    Get or create a global AttentionMaskBuilder instance.

    This function provides a convenient way to access the mask builder
    without managing instance lifecycle.

    Args:
        device: The device for the mask builder.

    Returns:
        AttentionMaskBuilder instance.
    """
    global _builder_instance, _builder_device
    if _builder_instance is None or _builder_device != device:
        _builder_instance = AttentionMaskBuilder(device)
        _builder_device = device
    return _builder_instance


__all__ = [
    "AttentionMaskBuilder",
    "get_attention_mask_builder",
]
