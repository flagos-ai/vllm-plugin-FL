# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Iluvatar RMS normalization delegating to FlagGems."""

from __future__ import annotations

from typing import Optional, Union

import torch
from flag_gems.modules.normalization import gems_rms_forward


def rms_norm_iluvatar(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Delegate standalone or residual-fused RMSNorm to FlagGems `gems_rms_forward`."""
    return gems_rms_forward(x, residual, obj.weight, obj.variance_epsilon)
