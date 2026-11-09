# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda vendor implementation of rms_norm using PyTorch native ops.

The FlagGems Triton kernel fails on txda hardware with "out of resource:
shared memory" because the txda Triton backend has a lower shared-memory
limit.  This implementation uses pure PyTorch which is functionally
correct and works on all hardware.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_txda(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """RMS normalization using PyTorch native ops.

    Works on txda hardware where the Triton-based FlagGems kernel
    runs out of shared memory.
    """
    weight = obj.weight
    epsilon = obj.variance_epsilon

    if residual is not None:
        x = x + residual
        residual = x

    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    output = weight * x

    if residual is not None:
        return output, residual
    return output