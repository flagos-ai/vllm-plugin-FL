"""Hygon rotary embedding routed through the backend wrapper patch."""

from __future__ import annotations

from typing import Optional

import torch


def rotary_embedding_hygon(
    obj,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
):
    from vllm._custom_ops import rotary_embedding

    if not inplace:
        query = query.clone()
        if key is not None:
            key = key.clone()
    rotary_embedding(
        position_ids,
        query,
        key,
        obj.head_size,
        torch.cat((cos, sin), dim=-1),
        not rotary_interleaved,
    )
    return query, key
