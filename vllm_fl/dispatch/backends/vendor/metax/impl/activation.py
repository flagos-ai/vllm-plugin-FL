# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX activation operator implementations.
"""

from __future__ import annotations

import torch


def silu_and_mul_metax(obj, x: torch.Tensor) -> torch.Tensor:

