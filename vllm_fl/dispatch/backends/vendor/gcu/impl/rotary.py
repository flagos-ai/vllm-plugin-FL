# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import torch


def rotary_embedding_gcu(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if position_ids.dim() == 1:
        cos_selected = cos[position_ids]
        sin_selected = sin[position_ids]
    else:
        cos_selected = cos[position_ids]
        sin_selected = sin[position_ids]

    if query.dim() == 4:
        cos_selected = cos_selected.unsqueeze(1)
        sin_selected = sin_selected.unsqueeze(1)
    elif query.dim() == 3:
        cos_selected = cos_selected.unsqueeze(1)
        sin_selected = sin_selected.unsqueeze(1)

    rotary_dim = cos_selected.shape[-1]
    head_dim = query.shape[-1]

    if rotary_dim != head_dim:
        cos_selected = torch.cat([cos_selected, cos_selected], dim=-1)
        sin_selected = torch.cat([sin_selected, sin_selected], dim=-1)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    if rotary_interleaved:
        def rotate_interleaved(x):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).flatten(-2)

        q_embed = (query * cos_selected) + (rotate_interleaved(query) * sin_selected)
        k_embed = (key * cos_selected) + (rotate_interleaved(key) * sin_selected)
    else:
        q_embed = (query * cos_selected) + (rotate_half(query) * sin_selected)
        k_embed = (key * cos_selected) + (rotate_half(key) * sin_selected)

    return q_embed, k_embed
