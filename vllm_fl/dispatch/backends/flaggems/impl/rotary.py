# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems rotary embedding operator implementations.
"""

from __future__ import annotations

import functools
import inspect

import torch


@functools.cache
def _supports_inplace_argument(fn) -> bool:
    try:
        return "inplace" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


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
    import flag_gems
    from flag_gems.config import use_c_extension
    from flag_gems.modules.rotary_embedding import gems_rope_forward

    # The official Ascend implementation in the qwen-vllm_for_ascend branch
    # exposes a six-argument apply_rotary_pos_emb without ``inplace``.  The
    # common Gems wrapper assumes the seven-argument API when the C extension
    # is disabled, so call the backend implementation directly in that case.
    if not use_c_extension and not _supports_inplace_argument(
        flag_gems.apply_rotary_pos_emb
    ):
        q_embed, k_embed = flag_gems.apply_rotary_pos_emb(
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved,
        )
        if inplace:
            if q_embed is not query:
                query.copy_(q_embed)
            if k_embed is not key:
                key.copy_(k_embed)
            return query, key
        return q_embed, k_embed

    return gems_rope_forward(
        query,
        key,
        cos,
        sin,
        position_ids=position_ids,
        rotary_interleaved=rotary_interleaved,
        inplace=inplace,
    )
