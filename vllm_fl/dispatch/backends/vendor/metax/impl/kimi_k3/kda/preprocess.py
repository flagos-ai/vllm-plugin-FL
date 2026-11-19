"""Materialize KDA FP32 inputs while preserving the legacy BF16 rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare(
    Q,
    K,
    V,
    QF,
    KF,
    VF,
    H: tl.constexpr,
    QT: tl.constexpr,
    QH: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    VT: tl.constexpr,
    VH: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    t, h = row // H, row % H
    d = tl.arange(0, 128)
    if part == 0:
        value = tl.load(Q + t * QT + h * QH + d)
    elif part == 1:
        value = tl.load(K + t * KT + h * KH + d)
    else:
        value = tl.load(V + t * VT + h * VH + d)
    if part < 2:
        x = value.to(tl.float64)
        denominator = tl.sqrt(tl.sum(x * x, 0) + tl.full((), 1e-6, tl.float64))
        # These two roundings are part of the existing normalization interface.
        result = (x / denominator).to(tl.float32).to(tl.bfloat16).to(tl.float32)
    else:
        result = value.to(tl.float32)
    if part == 0:
        tl.store(QF + row * 128 + d, result)
    elif part == 1:
        tl.store(KF + row * 128 + d, result)
    else:
        tl.store(VF + row * 128 + d, result)


def can_prepare(q, k, v):
    return (
        q.ndim == 4
        and q.shape[0] == 1
        and q.shape[2:] == (12, 128)
        and 1 <= q.shape[1] <= 512
        and all(
            x.shape == q.shape
            and x.is_cuda
            and x.device == q.device
            and x.dtype == torch.bfloat16
            and x.stride(-1) == 1
            for x in (q, k, v)
        )
    )


def prepare(q, k, v, record=None):
    if not can_prepare(q, k, v):
        raise ValueError("KDA fused materialization requires BF16 [1,T<=512,12,128]")
    outputs = tuple(
        torch.empty(q.shape, device=q.device, dtype=torch.float32) for _ in range(3)
    )
    kernel = _prepare[(q.shape[1] * 12, 3)](
        q,
        k,
        v,
        *outputs,
        12,
        q.stride(1),
        q.stride(2),
        k.stride(1),
        k.stride(2),
        v.stride(1),
        v.stride(2),
        num_warps=4,
        enable_fp_fusion=False,
    )
    if record is not None:
        record.update(
            n_regs=kernel.n_regs,
            n_spills=kernel.n_spills,
            shared=kernel.metadata.shared,
        )
    return outputs
