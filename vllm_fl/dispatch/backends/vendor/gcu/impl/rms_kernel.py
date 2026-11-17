# Copyright (c) 2026 BAAI. All rights reserved.

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rms_kernel(
    X,
    W,
    R,
    Y,
    S,
    N: tl.constexpr,
    V: tl.constexpr,
    EPS: tl.constexpr,
    ADD: tl.constexpr,
    WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * 4 + tl.arange(0, 4)[:, None]
    cols = tl.arange(0, BLOCK)[None, :]
    offsets = row * N + cols
    valid = (row < ROWS) & (cols < N)
    value = tl.load(X + offsets, valid, other=0).to(tl.float32)
    if ADD:
        residual = tl.load(R + offsets, valid, other=0).to(tl.float32)
        value = (value + residual).to(X.dtype.element_ty).to(tl.float32)
        tl.store(S + offsets, value, valid)
    square = tl.where(cols < V, value * value, 0.0)
    variance = tl.sum(square, 1)[:, None] / V
    output = value * tl.rsqrt(variance + EPS)
    if WEIGHT:
        weight = tl.load(W + cols, valid, other=0).to(tl.float32)
        output = output * weight
    tl.store(Y + offsets, output, valid)


def launch(x, weight, residual, variance_size, eps, use_weight):
    output = torch.empty_like(x)
    summed = torch.empty_like(x)
    width = x.shape[-1]
    rows = x.numel() // width
    if rows:
        _rms_kernel[(triton.cdiv(rows, 4),)](
            x,
            weight,
            x if residual is None else residual,
            output,
            summed,
            width,
            variance_size,
            eps,
            residual is not None,
            use_weight,
            BLOCK=triton.next_power_of_2(width),
            ROWS=rows,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output, summed
