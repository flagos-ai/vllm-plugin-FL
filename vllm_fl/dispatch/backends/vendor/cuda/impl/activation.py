# Copyright (c) 2026 BAAI. All rights reserved.

"""
CUDA activation operator implementations.
"""

from __future__ import annotations

import torch


def silu_and_mul_cuda(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using CUDA.

    Uses vLLM's optimized CUDA kernel when available.

    Args:
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    from vllm._custom_ops import silu_and_mul as vllm_silu_and_mul

    d = x.shape[-1] // 2
    out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
    vllm_silu_and_mul(out, x)
    return out
