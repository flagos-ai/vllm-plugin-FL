# Copyright (c) 2026 BAAI. All rights reserved.

"""
CUDA rotary embedding operator implementations.
"""

from __future__ import annotations

import torch


def rotary_embedding_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding using CUDA.

    Uses vLLM's optimized CUDA kernel when available.

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
    from vllm._custom_ops import rotary_embedding as vllm_rotary_embedding

    vllm_rotary_embedding(
        position_ids,
        query,
        key,
        cos,
        sin,
        rotary_interleaved,
    )
    return query, key
