# Copyright (c) 2026 BAAI. All rights reserved.

"""PyTorch reference MoE router GEMM implementation."""

import torch


def router_gemm_bf16_fp32_torch(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return torch.mm(x, weight.T, out_dtype=torch.float32)
