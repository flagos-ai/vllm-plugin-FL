# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul_gcu(obj, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return F.silu(x1) * x2
