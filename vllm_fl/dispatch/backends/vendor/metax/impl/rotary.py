# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX rotary embedding operator implementations.
"""

from __future__ import annotations

import torch


def rotary_embedding_metax(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:

