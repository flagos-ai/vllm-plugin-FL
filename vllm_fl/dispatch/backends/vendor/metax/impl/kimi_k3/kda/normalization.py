"""Diagnostic precise L2 normalization for BF16 KDA input materialization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _normalize(X, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + d, d < D, other=0).to(tl.float64)
    denominator = tl.sqrt(tl.sum(x * x, 0) + tl.full((), 1e-6, tl.float64))
    normalized = x / denominator
    tl.store(Y + row * D + d, normalized.to(tl.float32).to(Y.dtype.element_ty), d < D)


def l2norm(x):
    if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.shape[-1] != 128:
        raise ValueError(
            "This candidate requires contiguous BF16 KDA head dimension 128"
        )
    out = torch.empty_like(x)
    _normalize[(x.numel() // 128,)](
        x, out, 128, 128, num_warps=4, enable_fp_fusion=False
    )
    return out
