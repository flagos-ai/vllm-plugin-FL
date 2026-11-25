# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""M3 launch adapter over the pinned FG Linear/GEMV kernels."""

import torch
import triton
from flag_gems.ops.linear import linear_kernel
from flag_gems.runtime import torch_device_fn

from vllm_fl.dispatch.jit import launch_handle

_configured_linear_kernel = launch_handle(linear_kernel)

def linear_with_config(
    input, weight, bias=None, *, config, kind="dot", order=0, group_m=1
):
    """Apply Linear with explicit launch metadata, without changing global tuning.

    This inference entry preserves the ordinary wrapper's views, allocation and
    weight materialization. The caller owns model-specific shape selection.
    """
    from flag_gems.ops.linear_primitives import linear_gemv_untransposed_kernel

    if kind not in ("dot", "gemv"):
        raise ValueError("Expected dot or gemv Linear implementation")
    if not input.is_cuda or input.device != weight.device:
        raise NotImplementedError(
            "Configured Linear requires one CUDA-compatible device"
        )
    if input.dtype != weight.dtype or input.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise NotImplementedError(
            "Configured Linear supports BF16/FP32 matching matrices"
        )
    if kind == "gemv" and bias is not None:
        raise NotImplementedError("Configured GEMV does not apply a bias")
    if input.requires_grad or weight.requires_grad:
        raise NotImplementedError("Configured Linear is inference-only")
    batch_dims = input.shape[:-1]
    flat = input.unsqueeze(0) if input.dim() == 1 else input
    m = 1
    for dim in flat.shape[:-1]:
        m *= dim
    k, n = flat.shape[-1], weight.shape[0]
    a = flat.view(m, k)
    w = weight.contiguous()
    out = torch.empty((m, n), device=input.device, dtype=input.dtype)
    cfg = dict(config)
    if m == 0:
        return out.view(*batch_dims, n)
    with torch_device_fn.device(input.device):
        if kind == "gemv":
            linear_gemv_untransposed_kernel[
                (triton.cdiv(m, cfg["BM"]), triton.cdiv(n, cfg["BN"]))
            ](a, w, out, m, n, k, **cfg)
        else:
            gm, gn = triton.cdiv(m, cfg["BLOCK_SIZE_M"]), triton.cdiv(
                n, cfg["BLOCK_SIZE_N"]
            )
            _configured_linear_kernel[(gm * gn,) if order else (gm, gn)](
                a,
                w,
                bias if bias is not None else w,
                out,
                m,
                n,
                k,
                *a.stride(),
                *w.stride(),
                *out.stride(),
                bias.stride(0) if bias is not None else 0,
                BIAS=bias is not None,
                ORDER=order,
                GROUP_M=group_m,
                **cfg,
            )
    return out.view(*batch_dims, n)
