# Copyright (c) 2026 BAAI. All rights reserved.

"""
Sunrise (PTPU) activation operator implementations.

Routes ``silu_and_mul`` to ``torch_ptpu.sgl_kernel.silu_and_mul``.
"""

from __future__ import annotations

import torch

from torch_ptpu.sgl_kernel import silu_and_mul as _ptpu_silu_and_mul


def silu_and_mul_sunrise(obj, x: torch.Tensor) -> torch.Tensor:
    """SiLU activation followed by element-wise multiplication on PTPU.

    Args:
        obj: The calling object (unused; kept for dispatch ABI parity with the
            FlagGems / reference implementations).
        x: Input tensor of shape ``[..., 2 * d]``. ``[..., :d]`` is the
            "gate" half and ``[..., d:]`` is the "up" half (gate-up
            projection convention).

    Returns:
        Output tensor of shape ``[..., d]``.
    """
    return _ptpu_silu_and_mul(x)
