# Copyright (c) 2026 BAAI. All rights reserved.

"""PyTorch reference quantization operator implementations."""

from __future__ import annotations

import torch


def dynamic_per_token_quant_int8_torch(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_fl.quantization.w8a8.reference import (
        dynamic_per_token_quant_int8,
    )

    return dynamic_per_token_quant_int8(x)
