# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Build element-offset copy plans for shared, strided GDN cache views."""

import math

import torch


def state_row_indices(state: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Map logical rows to their backing storage, without reading cache values.

    The returned device indices remain live during graph replay. Temporary
    metadata is proportional to the selected rows, not the cache slot count.
    Callers must supply valid slots; padding preservation belongs to the op.
    """
    if state.ndim not in (3, 4):
        raise ValueError("GDN state rows must have two or three dimensions")
    linear = torch.arange(math.prod(state.shape[1:]), device=state.device)
    offsets = torch.zeros_like(linear)
    for size, stride in zip(reversed(state.shape[1:]), reversed(state.stride()[1:])):
        offsets = offsets + (linear % size) * stride
        linear = linear // size
    return (
        slots[:, None] * state.stride(0) + offsets + state.storage_offset()
    ).flatten()
