# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import torch


def rms_norm_gcu(
    obj,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Match CUDA RMSNorm's accumulation and low-precision rounding points."""
    dtype = x.dtype
    value = x.float()
    if residual is not None:
        # CUDA fused_add_rms_norm stores the residual sum in scalar_t before
        # computing variance. Keep that rounding point for BF16/FP16 inputs.
        residual = (value + residual.float()).to(dtype)
        value = residual.float()

    variance_size = getattr(obj, "variance_size_override", None)
    variance_input = value if variance_size is None else value[..., :variance_size]
    variance = variance_input.square().mean(dim=-1, keepdim=True)
    output = value * torch.rsqrt(variance + obj.variance_epsilon)
    pass_weight = getattr(
        obj, "pass_weight_add" if residual is not None else "pass_weight", True
    )
    if pass_weight:
        # Casting before this multiplication introduces a second BF16 rounding
        # that the upstream CUDA kernel does not perform.
        output = output * obj.weight.float()
    output = output.to(dtype)
    return output if residual is None else (output, residual)
