# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import torch


def rms_norm_gcu(
    obj,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
