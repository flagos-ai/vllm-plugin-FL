# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference activation operator implementations using PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul_iluvatar(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using PyTorch.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    #TODO(chen.chen): replace with iluvatar kernel impl
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return F.silu(x1) * x2

    # from vllm._custom_ops import silu_and_mul as vllm_silu_and_mul
    # raise ValueError("should not reach")
    # d = x.shape[-1] // 2
    # out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
    # vllm_silu_and_mul(out, x)
    # return out
