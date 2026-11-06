# Copyright (c) 2026 BAAI. All rights reserved.

"""Decode shared-expert activation with the original BF16 rounding points."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _shared_swiglu_bf16_kernel(X, Y, M: tl.constexpr, D: tl.constexpr,
                              BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offsets // D, offsets % D
    mask = offsets < M * D
    gate = tl.minimum(tl.load(X + row * 2 * D + col, mask, 0).to(tl.float32), 10.0)
    up = tl.load(X + row * 2 * D + D + col, mask, 0).to(tl.float32)
    up = tl.minimum(tl.maximum(up, -10.0), 10.0)
    # The unfused sigmoid and first product both materialize BF16 tensors.
    sigmoid = (1.0 / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    act = (gate * sigmoid).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + offsets, act * (up + 0.0), mask)


def try_shared_swiglu(x: torch.Tensor, limit: float, alpha: float,
                     beta: float) -> torch.Tensor | None:
    """Return None for non-DV4/decode inputs so the original op stays available."""
    if (
        os.getenv("VLLM_FL_HYGON_SHARED_SWIGLU", "1").lower() in {"0", "off", "native"}
        or torch.version.hip is None
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.shape[1] != 512
        or not 1 <= x.shape[0] <= 256
        or not x.is_contiguous()
        or x.requires_grad
        or (limit, alpha, beta) != (10.0, 1.0, 0.0)
    ):
        return None
    out = torch.empty((x.shape[0], 256), device=x.device, dtype=x.dtype)
    _shared_swiglu_bf16_kernel[(triton.cdiv(out.numel(), 256),)](
        x, out, x.shape[0], 256, 256, num_warps=4,
    )
    return out
