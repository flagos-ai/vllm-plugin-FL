# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon activation operator fallbacks backed by vLLM native ops."""

from __future__ import annotations

import torch


def silu_and_mul_hygon(obj, x: torch.Tensor) -> torch.Tensor:
    """SiLU activation followed by element-wise multiplication."""
    d = x.shape[-1] // 2
    out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
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
