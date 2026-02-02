# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend activation operator implementations.
"""

from __future__ import annotations

import torch


def silu_and_mul_ascend(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using Ascend NPU.

    Args:
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    import torch_npu

    return torch_npu.npu_swiglu(x)
