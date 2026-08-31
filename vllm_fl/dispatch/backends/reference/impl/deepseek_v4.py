# Copyright (c) 2026 BAAI. All rights reserved.

"""PyTorch reference implementations of DeepSeek-V4-specific operators."""

from __future__ import annotations

import torch


def deepseek_v4_inv_rope_quant_int8_torch(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply inverse RoPE and symmetric per-group-row INT8 quantization."""
    if o.ndim != 3:
        raise ValueError("o must be a 3D [tokens, heads, head_dim] tensor")
    tokens, heads, head_dim = o.shape
    if heads != n_groups * heads_per_group:
        raise ValueError("heads must equal n_groups * heads_per_group")
    if head_dim != nope_dim + rope_dim:
        raise ValueError("head_dim must equal nope_dim + rope_dim")
    if rope_dim % 2 != 0:
        raise ValueError("rope_dim must be even")

    selected_cache = cos_sin_cache.index_select(0, positions.to(torch.long))
    half_rope = rope_dim // 2
    cos = selected_cache[:, :half_rope].unsqueeze(1).to(torch.float32)
    sin = selected_cache[:, half_rope:rope_dim].unsqueeze(1).to(torch.float32)

    values = o.to(torch.float32)
    nope = values[..., :nope_dim]
    rope = values[..., nope_dim:]
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    inv_rope = torch.stack(
        (even * cos + odd * sin, odd * cos - even * sin),
        dim=-1,
    ).flatten(-2)
    restored = torch.cat((nope, inv_rope), dim=-1).to(torch.bfloat16)

    group_dim = heads_per_group * head_dim
    grouped = (
        restored.reshape(tokens, n_groups, group_dim)
        .permute(1, 0, 2)
        .contiguous()
        .to(torch.float32)
    )
    absmax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-4)
    scales = absmax / 127.0
    normalized = grouped / scales
    rounded = torch.where(
        normalized >= 0,
        torch.floor(normalized + 0.5),
        torch.ceil(normalized - 0.5),
    )
    quantized = rounded.clamp(-127, 127).to(torch.int8)
    return quantized, scales
