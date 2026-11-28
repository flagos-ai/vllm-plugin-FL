# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import torch
from flag_gems import silu_and_mul


@torch.library.custom_op("vllm_fl::gcu_silu_and_mul", mutates_args=())
def _silu_and_mul_op(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return silu_and_mul(x1, x2)


@_silu_and_mul_op.register_fake
def _silu_and_mul_fake(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x1)


def silu_and_mul_gcu(obj, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    if torch.compiler.is_compiling() and not torch.is_grad_enabled():
        return _silu_and_mul_op(x1, x2)
    return silu_and_mul(x1, x2)
