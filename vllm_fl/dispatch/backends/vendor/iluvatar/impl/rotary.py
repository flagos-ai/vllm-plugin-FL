# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Iluvatar rotary embedding delegating to FlagGems."""

from __future__ import annotations

import torch
from flag_gems.modules.rotary_embedding import gems_rope_forward


def rotary_embedding_iluvatar(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Delegate NeoX or interleaved rotary position embedding to FlagGems `gems_rope_forward`."""
    return gems_rope_forward(
        query,
        key,
        cos,
        sin,
        position_ids=position_ids,
        rotary_interleaved=rotary_interleaved,
        inplace=inplace,
    )
