# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend rotary embedding operator implementations.
"""

from __future__ import annotations

import torch


def rotary_embedding_ascend(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding using Ascend NPU.

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
    import torch_npu

    # Get cos/sin for the positions
    if position_ids.dim() == 1:
        cos_selected = cos[position_ids]
        sin_selected = sin[position_ids]
    else:
        cos_selected = cos[position_ids]
        sin_selected = sin[position_ids]

    # Prepare cos/sin shape for npu_rotary_mul: [1, seq_len, 1, head_dim]
    head_dim = query.shape[-1]
    rotary_dim = cos_selected.shape[-1]

    # Duplicate cos/sin if needed to match head_dim
    if rotary_dim != head_dim:
        cos_selected = torch.cat([cos_selected, cos_selected], dim=-1)
        sin_selected = torch.cat([sin_selected, sin_selected], dim=-1)

    # Reshape cos/sin to [1, seq_len, 1, head_dim]
    cos_selected = cos_selected.reshape(1, -1, 1, head_dim)
    sin_selected = sin_selected.reshape(1, -1, 1, head_dim)

    # Reshape query/key to [1, seq_len, num_heads, head_dim]
    query_shape = query.shape
    key_shape = key.shape

    if query.dim() == 3:
        query = query.unsqueeze(0)
    if key.dim() == 3:
        key = key.unsqueeze(0)

    # Apply rotary embedding using NPU kernel
    q_embed = torch_npu.npu_rotary_mul(query, cos_selected, sin_selected)
    k_embed = torch_npu.npu_rotary_mul(key, cos_selected, sin_selected)

    # Restore original shape
    q_embed = q_embed.view(query_shape)
    k_embed = k_embed.view(key_shape)

    return q_embed, k_embed
