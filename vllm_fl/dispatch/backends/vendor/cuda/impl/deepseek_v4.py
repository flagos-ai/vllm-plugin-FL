# Copyright (c) 2026 BAAI. All rights reserved.

"""CUDA vendor implementation of DeepSeek-V4-specific operators."""

from __future__ import annotations

import torch

from vllm_fl.ops.deepseek_v4_int8_woa import (
    fused_inv_rope_quant_int8_triton,
)


def deepseek_v4_inv_rope_quant_int8_cuda(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused Triton implementation on NVIDIA CUDA devices."""
    return fused_inv_rope_quant_int8_triton(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        nope_dim,
        rope_dim,
    )
