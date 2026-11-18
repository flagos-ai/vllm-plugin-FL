# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm.model_executor.layers.activation import (
    SiluAndMul,
    GeluAndMul,
)


def silu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    """SiLU(x) * y using the FlagGems kernel.

    Do not instantiate vLLM's ``SiluAndMul`` here: its constructor resolves
    dispatch forward from the current vLLM config, which is unset at forward
    time (AssertionError: Current vLLM config is not set).
    """
    import flag_gems

    d = x.size(-1) // 2
    gate, up = x.split(d, dim=-1)
    return flag_gems.silu_and_mul(gate, up)


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
    import flag_gems

    d = x.size(-1) // 2
    gate, up = x.split(d, dim=-1)
    return flag_gems.gelu_and_mul(gate, up)
