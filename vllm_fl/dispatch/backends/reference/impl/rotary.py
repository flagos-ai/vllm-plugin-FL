# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference rotary embedding operator implementations using PyTorch.
"""

from __future__ import annotations

import torch


def rotary_embedding_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding using PyTorch.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim] or [seq_len, num_heads, head_dim]
        key: Key tensor [batch, num_heads, seq_len, head_dim] or [seq_len, num_heads, head_dim]
        cos: Cosine cache [max_seq_len, rotary_dim] where rotary_dim = head_dim or head_dim // 2
        sin: Sine cache [max_seq_len, rotary_dim] where rotary_dim = head_dim or head_dim // 2
        position_ids: Position indices [batch, seq_len] or [seq_len]
        rotary_interleaved: Whether to use interleaved rotary
        inplace: Whether to modify tensors in-place (ignored in reference impl)

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    # Get cos/sin for the positions
    # position_ids can be [batch, seq_len] or [seq_len]
    if position_ids.dim() == 1:
        # [seq_len] -> [seq_len, rotary_dim]
        cos_selected = cos[position_ids]
        sin_selected = sin[position_ids]
    else:
        # [batch, seq_len] -> [batch, seq_len, rotary_dim]
        cos_selected = cos[position_ids]
        sin_selected = sin[position_ids]

    # Expand dimensions to match query/key shape
    # query/key: [batch, num_heads, seq_len, head_dim] or [seq_len, num_heads, head_dim]
    if query.dim() == 4:
        # [batch, num_heads, seq_len, head_dim]
        # cos_selected: [batch, seq_len, rotary_dim] -> [batch, 1, seq_len, rotary_dim]
        cos_selected = cos_selected.unsqueeze(1)
        sin_selected = sin_selected.unsqueeze(1)
    elif query.dim() == 3:
        # [seq_len, num_heads, head_dim]
        # cos_selected: [seq_len, rotary_dim] -> [seq_len, 1, rotary_dim]
        cos_selected = cos_selected.unsqueeze(1)
        sin_selected = sin_selected.unsqueeze(1)

    # Check if we need to repeat cos/sin to match head_dim
    rotary_dim = cos_selected.shape[-1]
    head_dim = query.shape[-1]

    if rotary_dim != head_dim:
        # cos/sin only covers half of head_dim, need to repeat
        # This handles the case where rotary is only applied to part of the dimensions
        cos_selected = torch.cat([cos_selected, cos_selected], dim=-1)
        sin_selected = torch.cat([sin_selected, sin_selected], dim=-1)

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    if rotary_interleaved:
        # Interleaved rotary
        def rotate_interleaved(x):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).flatten(-2)

        q_embed = (query * cos_selected) + (rotate_interleaved(query) * sin_selected)
        k_embed = (key * cos_selected) + (rotate_interleaved(key) * sin_selected)
    else:
        # Standard rotary (neox style)
        q_embed = (query * cos_selected) + (rotate_half(query) * sin_selected)
        k_embed = (key * cos_selected) + (rotate_half(key) * sin_selected)

    return q_embed, k_embed
