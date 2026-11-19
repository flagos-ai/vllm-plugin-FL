# SPDX-License-Identifier: Apache-2.0
"""Gemma normalization used explicitly by the PCP prefill path."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_square(X, R, SQUARE, RES, N: tl.constexpr, HEADS: tl.constexpr,
                XS0: tl.constexpr, XS1: tl.constexpr, XS2: tl.constexpr,
                RS0: tl.constexpr, RS1: tl.constexpr, RS2: tl.constexpr,
                HAS_RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    token, head = row // HEADS, row % HEADS
    x = tl.load(X + token * XS0 + head * XS1 + col * XS2,
                col < N, other=0).to(tl.float32)
    if HAS_RESIDUAL:
        r = tl.load(R + token * RS0 + head * RS1 + col * RS2,
                    col < N, other=0).to(tl.float32)
        # Native Gemma BF16 residual addition rounds before normalization.
        x = (x + r).to(X.dtype.element_ty).to(tl.float32)
        tl.store(RES + row * N + col, x, col < N)
    tl.store(SQUARE + row * N + col, x * x, col < N)


@triton.jit
def _gemma_scale(X, W, VAR, Y, N: tl.constexpr, HEADS: tl.constexpr,
                 XS0: tl.constexpr, XS1: tl.constexpr, XS2: tl.constexpr,
                 WS: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    token, head = row // HEADS, row % HEADS
    x = tl.load(X + token * XS0 + head * XS1 + col * XS2,
                col < N, other=0).to(tl.float32)
    weight = tl.load(W + col * WS, col < N, other=0).to(tl.float32) + 1.0
    variance = tl.load(VAR + row)
    out = (x * tl.rsqrt(variance + EPS)) * weight
    tl.store(Y + row * N + col, out, col < N)


def prefill_gemma_norm(module, x, residual=None):
    """Fuse BF16 Gemma elementwise ops around the native mean reduction.

    Called only at explicit PCP prefill sites. In particular this does not
    patch the shared GemmaRMSNorm class or alter native TP/decode dispatch.
    Three-dimensional Q/K views retain their original projection strides.
    Keeping torch.mean preserves the native FP32 reduction order; changing
    that order can flip BF16 rounding and eventually generated tokens.
    """
    if (not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim not in (2, 3)
            or not 0 < x.shape[-1] <= 8192
            or module.weight.ndim != 1
            or module.weight.numel() != x.shape[-1]
            or module.weight.device != x.device
            or module.weight.dtype not in (torch.bfloat16, torch.float32)
            or (residual is not None and (
                residual.shape != x.shape or residual.dtype != x.dtype
                or residual.device != x.device))):
        return module(x) if residual is None else module(x, residual)

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    res_out = torch.empty_like(out) if residual is not None else out
    if x.numel():
        heads = x.shape[1] if x.ndim == 3 else 1
        xs = x.stride() if x.ndim == 3 else (x.stride(0), 0, x.stride(1))
        r = residual if residual is not None else x
        rs = r.stride() if r.ndim == 3 else (r.stride(0), 0, r.stride(1))
        square = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        grid = (x.numel() // x.shape[-1],)
        _gemma_square[grid](
            x, r, square, res_out, x.shape[-1], heads, *xs, *rs,
            residual is not None, BLOCK=triton.next_power_of_2(x.shape[-1]),
            enable_fp_fusion=False,
        )
        variance = square.mean(dim=-1, keepdim=True)
        norm_input = res_out if residual is not None else x
        ns = (norm_input.stride() if x.ndim == 3
              else (norm_input.stride(0), 0, norm_input.stride(1)))
        _gemma_scale[grid](
            norm_input, module.weight, variance, out, x.shape[-1], heads,
            *ns, module.weight.stride(0), module.variance_epsilon,
            BLOCK=triton.next_power_of_2(x.shape[-1]), enable_fp_fusion=False,
        )
    return out if residual is None else (out, res_out)
