# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon rotary embedding fallback backed by vLLM native ops."""

from __future__ import annotations

import torch


def rotary_embedding_hygon(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding using vLLM custom ops."""
    from vllm._custom_ops import rotary_embedding as vllm_rotary_embedding

    if not inplace:
        query = query.clone()
        key = key.clone()

    vllm_rotary_embedding(
        position_ids,
        query,
        key,
        cos,
        sin,
        rotary_interleaved,
    )
    return query, key
