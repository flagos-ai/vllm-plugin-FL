# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagGems MoE router GEMM implementation."""

import torch


def router_gemm_bf16_fp32_flaggems(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    from flag_gems import router_gemm

    # Keep vLLM's descriptive dispatch name at the plugin boundary.  The
    # current FlagGems public API calls the same bf16 x bf16 -> fp32 primitive
    # simply ``router_gemm``.
    return router_gemm(x, weight)
