# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference normalization operator implementations using PyTorch.
"""

from __future__ import annotations

import torch


def rms_norm_torch(
    obj,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    RMS normalization using PyTorch.

    Args:
        x: Input tensor
        residual: Optional residual tensor
        obj: The calling obj (e.g., RMSNorm layer)

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    # Match the vLLM layer contract: accumulate residuals and variance in
    # float32, then return both tensors in the input dtype.
    input_dtype = x.dtype
    x_float = x.float()
    if residual is not None:
        x_float = x_float + residual.float()
        residual = x_float.to(input_dtype)

    variance_size = getattr(obj, "variance_size_override", None)
    variance_input = x_float if variance_size is None else x_float[..., :variance_size]
    variance = variance_input.square().mean(dim=-1, keepdim=True)
    output = x_float * torch.rsqrt(variance + obj.variance_epsilon)
    pass_weight = getattr(
        obj, "pass_weight_add" if residual is not None else "pass_weight", True
    )
    if pass_weight:
        output = output.to(obj.weight.dtype) * obj.weight
    output = output.to(input_dtype)

    if residual is not None:
        return output, residual
    return output
