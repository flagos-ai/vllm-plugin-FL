# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm.model_executor.layers.activation import (
    SiluAndMul,
    GeluAndMul,
)


def silu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using CUDA.

    Uses vLLM's optimized CUDA kernel when available.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    act_fn = SiluAndMul()
    return act_fn.forward_cuda(x)


def gelu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation followed by element-wise multiplication using CUDA.

    Uses vLLM's optimized CUDA kernel when available.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    act_fn = GeluAndMul()
    return act_fn.forward_cuda(x)
