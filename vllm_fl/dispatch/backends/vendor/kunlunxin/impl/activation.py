# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

"""
Kunlunxin activation operator implementations.
"""

from __future__ import annotations

import torch


def silu_and_mul_kunlunxin(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    import xtorch_ops

    d = x.shape[-1] // 2
    out = torch.empty(
        *x.shape[:-1], d, dtype=x.dtype, device=x.device
    )
    xtorch_ops.swiglu(x, out)
    return out
