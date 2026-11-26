# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Iluvatar SiLU-and-Mul activation delegating to FlagGems."""

from __future__ import annotations

import torch
from flag_gems.modules.activation import gems_silu_and_mul


def silu_and_mul_iluvatar(obj, x: torch.Tensor) -> torch.Tensor:
    """Split `[..., 2 * d]` gate/up tensor in half and delegate SwiGLU to FlagGems `gems_silu_and_mul`."""
    d = x.shape[-1] // 2
    return gems_silu_and_mul(x[..., :d], x[..., d:])
