# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import torch


@torch.library.custom_op("vllm_fl::gcu_rms_norm", mutates_args=())
def _rms_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor | None,
    variance_size: int,
    eps: float,
    use_weight: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from .rms_kernel import launch

    return launch(x, weight, residual, variance_size, eps, use_weight)


@_rms_op.register_fake
def _rms_fake(x, weight, residual, variance_size, eps, use_weight):
    return torch.empty_like(x), torch.empty_like(x)


def rms_norm_gcu(
    obj,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Match CUDA RMSNorm's accumulation and low-precision rounding points."""
    variance_size = getattr(obj, "variance_size_override", None)
    pass_weight = getattr(
        obj, "pass_weight_add" if residual is not None else "pass_weight", True
    )
    if (
        x.device.type == "gcu"
        and torch.compiler.is_compiling()
        and not torch.is_grad_enabled()
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and (
            residual is None
            or (
                residual.is_contiguous()
                and residual.shape == x.shape
                and residual.dtype == x.dtype
            )
        )
        and obj.weight.is_contiguous()
        and obj.weight.numel() == x.shape[-1]
        and (variance_size is None or 0 < variance_size <= x.shape[-1])
        and 0 < x.shape[-1] <= 8192
    ):
        output, summed = _rms_op(
            x,
            obj.weight,
            residual,
            x.shape[-1] if variance_size is None else variance_size,
            obj.variance_epsilon,
            pass_weight,
        )
        return output if residual is None else (output, summed)
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
