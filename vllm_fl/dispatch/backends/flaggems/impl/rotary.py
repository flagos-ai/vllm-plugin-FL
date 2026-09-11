# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems rotary embedding operator implementations.
"""

from __future__ import annotations

import torch
from flag_gems.pt2 import rotary_embedding_inplace


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
    if inplace:
        if position_ids is None:
            raise ValueError(
                "The compiler-integrated FlagGems RoPE path requires position_ids"
            )
        rotary_embedding_inplace(
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved,
        )
        return query, key

    # vLLM uses the in-place path above.  The generic out-of-place API has
    # no manifest entry yet, so it stays on FlagGems' original implementation.
    from flag_gems.modules.rotary_embedding import gems_rope_forward

    return gems_rope_forward(
        query,
        key,
        cos,
        sin,
        position_ids=position_ids,
        rotary_interleaved=rotary_interleaved,
        inplace=False,
    )
