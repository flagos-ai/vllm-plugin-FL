# Copyright (c) 2026 BAAI. All rights reserved.
"""Call-site repair for a dual-die descriptor inherited by a view.

A view keeps its parent's ``split_dim`` descriptor, so a row slice of a
row-split buffer still reports ``split_dim() == 0`` while its rows sit at the
parent width: ``unzip`` then parts the parent, a read sees another tensor's
rows and a write lands outside the view — and no view can be faithful there,
because the slice straddles the two die halves.  ``repair_view`` copies the
slice into a row-split tensor of its own shape; a write target copies the
result back.

"""

from __future__ import annotations

import torch

from ...local.double_die import to_split_dim_logical

__all__ = ["repair_view"]


def repair_view(tensor: torch.Tensor, split_dim: int = 0) -> torch.Tensor:
    """Return ``tensor`` with a descriptor faithful to its logical shape.

    The argument itself is returned when it is already faithful, so callers
    detect a repair with ``work is not original`` and write back only then.
    """
    if tensor.device.type != "lpu":
        # Dual-die layouts exist only on LPU; on any other device the
        # descriptor is trivially faithful and the lpu ops behind the repair
        # are not registered.
        return tensor
    return to_split_dim_logical(tensor, split_dim)
