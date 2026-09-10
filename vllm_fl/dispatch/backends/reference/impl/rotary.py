# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference rotary embedding operator implementations using PyTorch.
"""

from __future__ import annotations

import torch


def rotary_embedding_torch(
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
    Apply rotary position embedding using PyTorch.

    Args:
        obj: The calling obj (for interface consistency)
        query: Query tensor [batch, num_heads, seq_len, head_dim] or [seq_len, num_heads, head_dim]
        key: Key tensor [batch, num_heads, seq_len, head_dim] or [seq_len, num_heads, head_dim]
        cos: Cosine cache [max_seq_len, rotary_dim] where rotary_dim = head_dim or head_dim // 2
        sin: Sine cache [max_seq_len, rotary_dim] where rotary_dim = head_dim or head_dim // 2
        position_ids: Position indices [batch, seq_len] or [seq_len]
        rotary_interleaved: Whether to use interleaved rotary
        inplace: Whether to modify tensors in-place

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    if query.device != key.device or query.dtype != key.dtype:
        raise ValueError("query and key must have the same device and dtype")
    if query.dim() not in (3, 4) or key.dim() != query.dim():
        raise ValueError("query and key must both be 3D or both be 4D")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must have the same head dimension")
    if query.shape[-1] <= 0 or query.shape[-1] % 2 != 0:
        raise ValueError("the rotary head dimension must be a positive even number")
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("position_ids must have dtype int32 or int64")
    if query.dim() == 3:
        if position_ids.dim() != 1:
            raise ValueError("3D query and key require 1D position_ids")
        if query.shape[0] != key.shape[0] or query.shape[0] != position_ids.numel():
            raise ValueError(
                "query, key, and position_ids must have equal token counts"
            )
    elif (
        position_ids.dim() != 2
        or query.shape[0] != key.shape[0]
        or query.shape[2] != key.shape[2]
        or position_ids.shape != (query.shape[0], query.shape[2])
    ):
        raise ValueError(
            "4D BCHD query and key require matching batch/sequence dimensions "
            "and [batch, sequence] position_ids"
        )
    if cos.dim() != 2 or sin.shape != cos.shape:
        raise ValueError("cos and sin must be same-shaped 2D caches")
    if cos.shape[-1] not in (query.shape[-1], query.shape[-1] // 2):
        raise ValueError("cos and sin width must equal the head dimension or its half")

    # vLLM normally performs this match at the layer boundary. Keep the
    # standalone reference backend deterministic for direct callers and for
    # guarded vendor fallbacks as well.
    cos = cos.to(device=query.device, dtype=query.dtype)
    sin = sin.to(device=query.device, dtype=query.dtype)
    position_ids = position_ids.to(device=query.device, dtype=torch.long)

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
        # Half-width caches follow the coordinate layout used by each rotary
        # style: adjacent pairs for interleaved and two contiguous halves for
        # NeoX.
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

    if inplace:
        query.copy_(q_embed)
        key.copy_(k_embed)
        return query, key

    return q_embed, k_embed
