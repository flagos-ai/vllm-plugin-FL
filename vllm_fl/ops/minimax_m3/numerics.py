# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""MiniMax call contracts for the explicitly selected library numerical modes."""
import torch
import triton
from flag_gems.ops.gemma_rms_norm import gemma_rms_norm_kernel
from flaggems_vllm.ops.fused_add_rms_norm import fused_add_rms_norm
from flaggems_vllm.ops.fused_minimax_m3_qknorm_rope_kv_insert import (
    fused_minimax_m3_qknorm_rope_kv_insert,
)
from flaggems_vllm.ops.fused_moe import MoEActivation, apply_moe_activation


def _matrix(x):
    if x.ndim == 2 and x.stride(-1) == 1:
        return x
    if x.is_contiguous():
        return x.view(-1, x.shape[-1])
    raise ValueError("MiniMax operators require viewable rows and unit column stride")


def gemma(x, weight, eps, residual=None):
    if residual is not None:
        return fused_add_rms_norm(x, residual, (x.shape[-1],), weight, eps, gemma=True)
    # Preserve the accepted launch of the existing FlagGems Gemma kernel.
    a = _matrix(x)
    n = a.shape[-1]
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Unsupported Gemma input")
    if weight.shape != (n,) or weight.device != x.device or not weight.is_contiguous():
        raise ValueError("Invalid Gemma weight")
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    y = out.view(-1, n)
    inv = torch.empty(a.shape[0], device=x.device, dtype=torch.float32)
    if a.numel():
        gemma_rms_norm_kernel[(a.shape[0],)](
            y,
            inv,
            a,
            weight,
            y.stride(0),
            1,
            a.stride(0),
            1,
            n,
            eps,
            triton.next_power_of_2(n),
        )
    return out


def swiglu(x, limit=7.0, alpha=1.702, beta=1.0, out=None):
    a = _matrix(x)
    if out is None:
        out = torch.empty(
            (*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype
        )
    apply_moe_activation(
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        _matrix(out),
        a,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=limit,
        sigmoid_mode="division",
    )
    return out


def qknorm_rope_insert(*args, **kwargs):
    return fused_minimax_m3_qknorm_rope_kv_insert(
        *args, **kwargs, numerics="bf16_staged"
    )
