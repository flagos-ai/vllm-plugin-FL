"""TP8 MLA Q absorption; original V restoration is retained."""

import torch, triton
import triton.language as tl


@triton.jit
def mm(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AH: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BH: tl.constexpr,
    BKSTR: tl.constexpr,
    BNSTR: tl.constexpr,
    CH: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    h = tl.program_id(2)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(triton.cdiv(K, BK)):
        kk = start * BK + rk
        a = tl.load(
            A + h * AH + m[:, None] * AM + kk[None, :] * AK,
            (m[:, None] < M) & (kk[None, :] < K),
            0,
        )
        b = tl.load(
            B + h * BH + kk[:, None] * BKSTR + n[None, :] * BNSTR,
            (kk[:, None] < K) & (n[None, :] < N),
            0,
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        C + h * CH + m[:, None] * CM + n[None, :] * CN,
        acc.to(tl.bfloat16),
        (m[:, None] < M) & (n[None, :] < N),
    )


def project_query(x, w):
    if not (
        x.is_cuda
        and x.ndim == w.ndim == 3
        and x.shape[0] == 12
        and 1 <= x.shape[1] <= 16
        and x.shape[2] == 128
        and tuple(w.shape) == (12, 128, 512)
        and x.dtype == w.dtype == torch.bfloat16
        and x.device == w.device
        and x.stride(-1) == 1
    ):
        return torch.bmm(x, w)
    b = x.shape[1]
    out = torch.empty((12, b, 512), dtype=x.dtype, device=x.device)
    bn, warps = (32, 2) if b == 1 else (64, 4)
    mm[(1, triton.cdiv(512, bn), 12)](
        x,
        w,
        out,
        b,
        512,
        128,
        *x.stride(),
        *w.stride(),
        *out.stride(),
        16,
        bn,
        64,
        num_warps=warps,
        num_stages=1,
    )
    return out
