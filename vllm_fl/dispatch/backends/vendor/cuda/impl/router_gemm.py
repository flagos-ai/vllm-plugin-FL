# Copyright (c) 2026 BAAI. All rights reserved.

"""CUDA MoE router GEMM implementation."""

import torch


def router_gemm_bf16_fp32_cuda(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    from vllm._custom_ops import router_gemm_bf16_fp32

    return router_gemm_bf16_fp32(x, weight)
