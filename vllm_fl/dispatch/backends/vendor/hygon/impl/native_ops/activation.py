# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon activation implementations for CUDA-compatible Hygon runtimes."""

from __future__ import annotations

import torch


def silu_and_mul_hygon_out(output: torch.Tensor, x: torch.Tensor) -> None:
    """Run Hygon LightOp using vLLM's native out-parameter ABI."""
    try:
        from lightop import activation as op
    except ImportError:
        import lightop.op as op

    op.silu_and_mul_opt(output, x)


def silu_and_mul_hygon(obj, x: torch.Tensor) -> torch.Tensor:
    """SiLU activation followed by element-wise multiplication."""
    d = x.shape[-1] // 2
    out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
    silu_and_mul_hygon_out(out, x)
    return out


def gelu_and_mul_hygon(obj, x: torch.Tensor) -> torch.Tensor:
    """GELU activation followed by element-wise multiplication."""
    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    d = x.shape[-1] // 2
    out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
    if approximate == "tanh":
        torch.ops._C.gelu_tanh_and_mul(out, x)
    else:
        torch.ops._C.gelu_and_mul(out, x)
    return out
