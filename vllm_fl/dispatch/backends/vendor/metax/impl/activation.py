# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch


def silu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using CUDA.

    Uses mcoplib's _C kernel directly via torch.ops._C.silu_and_mul.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    d = x.shape[-1] // 2
    out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(out, x)
    return out


def gelu_and_mul_maca(obj, x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation followed by element-wise multiplication using CUDA.

    Uses mcoplib's _C kernel directly via torch.ops._C.gelu_and_mul.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    d = x.shape[-1] // 2
    out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
    torch.ops._C.gelu_and_mul(out, x)
    return out
