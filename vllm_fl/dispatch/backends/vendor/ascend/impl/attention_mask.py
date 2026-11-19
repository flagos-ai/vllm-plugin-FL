# Copyright (c) 2026 BAAI. All rights reserved.

"""
Attention mask builder for Ascend NPU backend.

This module provides utilities for creating and caching attention masks
required by Ascend NPU attention operators.
"""

import os
from typing import ClassVar, Optional

import torch

_RSWA_MASK_CHUNK_ELEMENTS = 1 << 24


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

    def __init__(self, device: torch.device, *, ocr_mode: bool = False):
        """
        Initialize the attention mask builder.

        Args:
            device: The device to create masks on.
        """
        self.device = device
        self.ocr_mode = ocr_mode

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

    def get_rswa_attn_mask(
        self,
        prefix_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        query_lens: torch.Tensor,
        max_kv_len: int,
        max_query_len: int,
        rswa_window: int,
    ) -> torch.Tensor:
        for name, size in (
            ("rswa_window", rswa_window),
            ("max_kv_len", max_kv_len),
            ("max_query_len", max_query_len),
        ):
            if type(size) is not int or size <= 0:
                raise ValueError(f"{name} must be a positive integer, got {size!r}")

        num_reqs = prefix_lens.numel()
        if seq_lens.numel() != num_reqs or query_lens.numel() != num_reqs:
            raise ValueError("R-SWA lengths must have the same number of requests")
        limit = os.environ.get(
            "VLLM_FL_RSWA_MASK_MAX_MIB", "2048" if self.ocr_mode else "512"
        )
        if not limit.isascii() or not limit.isdecimal() or not 1 <= int(limit) <= 8192:
            raise ValueError(
                "VLLM_FL_RSWA_MASK_MAX_MIB must be an integer from 1 to 8192"
            )
        if num_reqs * max_query_len * max_kv_len > int(limit) * (1 << 20):
            raise ValueError(f"R-SWA mask is too large: configured limit={limit} MiB")

        prefix_lens, seq_lens, query_lens = (
            lens.to(device=self.device, dtype=torch.int64, non_blocking=True).reshape(
                -1
            )
            for lens in (prefix_lens, seq_lens, query_lens)
        )
        # A full prompt prefix may exceed the KV processed by an early chunk.
        valid = (
            (prefix_lens >= 0)
            & (prefix_lens <= max_kv_len)
            & (query_lens >= 0)
            & (query_lens <= max_query_len)
            & (query_lens <= seq_lens)
            & (seq_lens <= max_kv_len)
        )
        torch._assert(valid.all(), "R-SWA lengths exceed mask bounds")

        # FIA needs the full KV width. Bound the temporary tensors by tiling rows.
        mask = torch.empty(
            (num_reqs, 1, max_query_len, max_kv_len),
            dtype=torch.int8,
            device=self.device,
        )
        output = mask.view(num_reqs * max_query_len, max_kv_len)
        keys = torch.arange(max_kv_len, device=self.device)
        chunk = max(1, _RSWA_MASK_CHUNK_ELEMENTS // max_kv_len)
        for start in range(0, len(output), chunk):
            rows = torch.arange(
                start, min(start + chunk, len(output)), device=self.device
            )
            requests = rows // max_query_len
            offsets = (rows % max_query_len).unsqueeze(1)
            prefixes = prefix_lens[requests].unsqueeze(1)
            sequences = seq_lens[requests].unsqueeze(1)
            queries = query_lens[requests].unsqueeze(1)
            positions = sequences - queries + offsets
            keep = (
                (offsets < queries)
                & (keys < sequences)
                & (keys <= positions)
                & ((keys < prefixes) | (keys > positions - rswa_window))
            )
            output[start : start + chunk].copy_(~keep)
        return mask

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
        mask_flag = torch.ones((max_seq_len, max_seq_len), dtype=torch.bool).tril_()
        # Invert to get mask positions (True for masked positions)
        mask_flag = ~mask_flag
        # For fp16, use -inf; otherwise use 1
        mask_value = float("-inf") if dtype == torch.float16 else 1
        attn_mask = torch.zeros(size=(max_seq_len, max_seq_len), dtype=dtype).masked_fill_(
            mask_flag, mask_value
        )
        return attn_mask.contiguous().to(self.device)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached masks. Useful for testing or memory cleanup."""
        cls._chunked_prefill_mask = None
        cls._chunked_prefill_mask_device = None
        cls._mla_mask = None
        cls._mla_mask_dtype = None
        cls._pcp_mla_mask = None
        cls._pcp_mla_mask_dtype = None


# Global obj cache for convenience
_builder_instance: Optional[AttentionMaskBuilder] = None
_builder_device: Optional[torch.device] = None


def get_attention_mask_builder(device: torch.device) -> AttentionMaskBuilder:
    """
    Get or create a global AttentionMaskBuilder obj.

    This function provides a convenient way to access the mask builder
    without managing obj lifecycle.

    Args:
        device: The device for the mask builder.

    Returns:
        AttentionMaskBuilder obj.
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
