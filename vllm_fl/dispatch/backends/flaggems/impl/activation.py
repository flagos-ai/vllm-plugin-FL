# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems activation operator implementations.
"""

from __future__ import annotations

import torch

import vllm_fl.envs as fl_envs


def silu_and_mul_flaggems(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    if fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        from flaggems_vllm import silu_and_mul as gems_silu_and_mul
    else:
        from flag_gems.modules.activation import gems_silu_and_mul

    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gems_silu_and_mul(x1, x2)


def gelu_and_mul_flaggems(obj, x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation followed by element-wise multiplication using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    if fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        from flaggems_vllm import gelu_and_mul
    else:
        from flag_gems.fused import gelu_and_mul

    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gelu_and_mul(x1, x2, approximate)


def silu_and_mul_with_clamp_flaggems(obj, x: torch.Tensor) -> torch.Tensor:
    if not fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM:
        from vllm.model_executor.layers.activation import SiluAndMulWithClamp

        return SiluAndMulWithClamp.forward_native(obj, x)
    from flaggems_vllm import silu_and_mul_with_clamp

    gate, up = x.chunk(2, dim=-1)
    return silu_and_mul_with_clamp(gate, up, obj.swiglu_limit)
