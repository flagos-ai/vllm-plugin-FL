# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference rotary embedding operator implementations using PyTorch.
"""

from __future__ import annotations

import torch


def apply_rotary_emb_torch(
    obj, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply partial RoPE with vLLM's native math on the rotary slice."""
    rotary_dim = cos.shape[-1] * 2
    rotated = obj.forward_native(x[..., :rotary_dim], cos, sin)
    if rotary_dim == x.shape[-1]:
        return rotated
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


def rotary_embedding_torch(
    obj,
    query: torch.Tensor,
    key: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Apply rotary position embedding using PyTorch.

    Args:
        obj: The calling obj (for interface consistency)
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
    positions = position_ids.long()
    cos_selected = cos[positions].to(query.dtype)
    sin_selected = sin[positions].to(query.dtype)

    # Expand dimensions to match query/key shape
    # query/key: [batch, num_heads, seq_len, head_dim] or [seq_len, num_heads, head_dim]
    if query.dim() == 4:
        # [batch, num_heads, seq_len, head_dim]
        # cos_selected: [batch, seq_len, rotary_dim] -> [batch, 1, seq_len, rotary_dim]
        cos_selected = cos_selected.unsqueeze(-3)
        sin_selected = sin_selected.unsqueeze(-3)
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
        if rotary_interleaved:
            cos_selected = cos_selected.repeat_interleave(2, dim=-1)
            sin_selected = sin_selected.repeat_interleave(2, dim=-1)
        else:
            cos_selected = torch.cat([cos_selected, cos_selected], dim=-1)
            sin_selected = torch.cat([sin_selected, sin_selected], dim=-1)

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def rotate_interleaved(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    rotate = rotate_interleaved if rotary_interleaved else rotate_half

    def apply(tensor):
        return tensor * cos_selected + rotate(tensor) * sin_selected

    return apply(query), None if key is None else apply(key)
