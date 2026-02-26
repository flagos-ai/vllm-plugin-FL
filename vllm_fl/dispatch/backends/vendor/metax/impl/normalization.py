# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX normalization operator implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_metax(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

