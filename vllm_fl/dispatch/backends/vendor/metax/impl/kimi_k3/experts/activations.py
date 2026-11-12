"""Accepted SiTU materialization and fused kernel paths."""

import torch
import triton
import triton.language as tl
from flaggems_vllm.ops.situ_and_mul import _situ_and_mul_kernel as _situ


def situ(x, beta, linear_beta):
    gate, up = x.chunk(2, dim=-1)
    gate, up = gate.float(), up.float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(x.dtype)


@triton.jit
def _shared_situ(
    X,
    OUT,
    ELEMENTS: tl.constexpr,
    I: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Inline the same canonical math, but give shared/dense FFN a distinct
    # profiler entry. This is one GPU launch, not an extra conversion or copy.
    _situ(X, OUT, ELEMENTS, I, BETA, LINEAR_BETA, BLOCK)


def _can_fuse_situ_input(x, beta, linear_beta):
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and 1 <= x.shape[0] <= 4096
        and x.shape[1] in (384, 1536, 2112, 8448)
        and x.is_contiguous()
        and beta == 4.0
        and linear_beta == 25.0
    )


def situ_out(out, x, beta=1.0, linear_beta=-1.0):
    # Qualified decode and prefill shared/dense FFN shapes. Reuse the routed-expert SiTU kernel:
    # identical BF16 boundaries and FP32 math; write the caller's out directly.
    if (
        _can_fuse_situ_input(x, beta, linear_beta)
        and out.is_contiguous()
        and out.device == x.device
        and out.dtype == x.dtype
        and tuple(out.shape) == (x.shape[0], x.shape[1] // 2)
    ):
        d = x.shape[1] // 2
        _shared_situ[(triton.cdiv(out.numel(), 256),)](
            x,
            out,
            out.numel(),
            d,
            beta,
            linear_beta,
            256,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return
    out.copy_(situ(x, beta, linear_beta if linear_beta > 0 else None))
