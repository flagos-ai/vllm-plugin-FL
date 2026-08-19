# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems rotary embedding operator implementations.
"""

from __future__ import annotations

import torch

import vllm_fl.envs as fl_envs


def rotary_embedding_flaggems(
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
    Apply rotary position embedding using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
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
    if fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        from flaggems_vllm.ops.rotary_embedding import apply_rotary_pos_emb as gems_rope_forward
    else:
        from flag_gems import apply_rotary_pos_emb as gems_rope_forward

    return gems_rope_forward(
        query,
        key,
        cos,
        sin,
        position_ids=position_ids,
        rotary_interleaved=rotary_interleaved,
        inplace=inplace,
    )
