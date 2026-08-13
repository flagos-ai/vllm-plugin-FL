# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagGems MoE router GEMM implementation."""

import torch
from vllm_fl.utils import use_flaggems_vllm


def router_gemm_bf16_fp32_flaggems(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    if use_flaggems_vllm():
        from flaggems_vllm.ops.router_gemm import router_gemm
    else:
        from flag_gems import router_gemm

    # Keep vLLM's descriptive dispatch name at the plugin boundary.  The
    # current FlagGems public API calls the same bf16 x bf16 -> fp32 primitive
    # simply ``router_gemm``.
    return router_gemm(x, weight)
