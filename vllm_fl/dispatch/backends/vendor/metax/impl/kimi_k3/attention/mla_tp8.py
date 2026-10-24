"""TP8 MLA decode:12query heads,page768,staged BF16QK/PV and FP32merge."""

import torch
import triton
import triton.language as tl
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.attention.mla4k_long import (
    _qk_long_kernel,
    _merge_long_kernel,
)


@triton.jit
def _pv_tp8(
    Cache,
    ST,
    SB,
    Table,
    STB,
    Lengths,
    PH,
    PL,
    Alpha,
    Partial,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNKS: tl.constexpr,
    PAGE: tl.constexpr,
    CAP: tl.constexpr,
    BN: tl.constexpr,
    VT: tl.constexpr,
    SHORT_SPLITS: tl.constexpr = 0,
    SHORT_LIMIT: tl.constexpr = 4096,
):
    b = tl.program_id(0)
    part = tl.program_id(1)
    tile = tl.program_id(2)
    length = tl.load(Lengths + b)
    if (length <= 0) | (length > CAP):
        return
    h = tl.arange(0, 16)
    d = tile * VT + tl.arange(0, VT)
    n = tl.arange(0, BN)
    if SHORT_SPLITS > 0:
        active = tl.where(length <= SHORT_LIMIT, SHORT_SPLITS, SPLITS)
        if part >= active:
            return
        blocks = tl.where(
            length <= SHORT_LIMIT,
            tl.cdiv(tl.cdiv(length, BN), SHORT_SPLITS),
            tl.cdiv(tl.cdiv(length, BN), SPLITS),
        )
    else:
        blocks = tl.cdiv(tl.cdiv(length, BN), SPLITS)
    start = part * blocks * BN
    end = tl.minimum((part + 1) * blocks * BN, length)
    chunks = tl.cdiv(tl.maximum(end - start, 0), BN)
    acc = tl.zeros((16, VT), tl.float32)
    for step in range(chunks):
        pos = start + step * BN + n
        valid = pos < length
        page = tl.load(Table + b * STB + pos // PAGE, valid, 0).to(tl.int64)
        offset = page * SB + (pos % PAGE) * ST
        value = tl.load(Cache + offset[:, None] + d[None, :], valid[:, None], 0)
        slot = (((b * SPLITS + part) * H + h) * CHUNKS + step).to(tl.int64)
        hi = tl.load(PH + slot[:, None] * BN + n[None, :], h[:, None] < H, 0)
        lo = tl.load(PL + slot[:, None] * BN + n[None, :], h[:, None] < H, 0)
        alpha = tl.load(Alpha + slot, h < H, 0)
        acc = acc * alpha[:, None]
        acc = tl.dot(hi, value, acc)
        acc = tl.dot(lo, value, acc)
    tl.store(
        Partial + ((b * SPLITS + part) * H + h[:, None]) * 512 + d[None, :],
        acc,
        h[:, None] < H,
    )


def can_use(q, cache, table, lengths):
    return (
        q.ndim == 4
        and 1 <= q.shape[0] <= 64
        and q.shape[1:] == (1, 12, 576)
        and cache.ndim == 4
        and cache.shape[1:] == (768, 1, 576)
        and table.ndim == 2
        and table.shape[0] == q.shape[0]
        and table.stride(-1) == 1
        and lengths.shape == (q.shape[0],)
        and lengths.is_contiguous()
        and q.dtype == cache.dtype == torch.bfloat16
        and table.dtype == lengths.dtype == torch.int32
        and q.is_cuda
        and all(t.device == q.device for t in (cache, table, lengths))
        and q.stride(-1) == cache.stride(-1) == 1
    )


def run(
    q,
    cache,
    table,
    lengths,
    scale,
    record=None,
    *,
    pv_vt=128,
    pv_warps=4,
    merge_vt=128,
    merge_warps=4,
    qk_tile=None,
    qk_warps=None,
    short_splits=None,
    short_limit=None,
):
    if not can_use(q, cache, table, lengths) or scale != 192**-0.5:
        raise ValueError("TP8 MLA requires qualified12head/page768 layout")
    if pv_vt not in (64, 128, 256, 512) or merge_vt not in (64, 128, 256, 512):
        raise ValueError("Invalid V tile")
    b = q.shape[0]
    h = 12
    splits = 64 if b >= 4 else 128
    if qk_tile is None:
        qk_tile = 128 if b in (1, 2, 4, 8) else 64
    if qk_warps is None:
        qk_warps = 4 if b in (1, 2, 4, 8) else 2
    if short_splits is None:
        short_splits = 64 if b == 2 else 32 if 4 <= b <= 16 else 0
    if short_limit is None:
        short_limit = 5120 if b == 8 else 1024 if b == 4 else 2048
    if short_limit not in (1024, 2048, 5120):
        raise ValueError("Unqualified short KV boundary")
    cap = min(69632, table.shape[1] * 768)
    chunks = triton.cdiv(triton.cdiv(cap, 16), splits)
    if short_splits:
        if short_splits not in (8, 16, 32, 64) or short_splits > splits:
            raise ValueError("Invalid short partition count")
        chunks = max(
            chunks, triton.cdiv(triton.cdiv(min(cap, short_limit), 16), short_splits)
        )
    out = torch.empty((b, 1, h, 512), device=q.device, dtype=q.dtype)
    lse = torch.empty((b, h, 1), device=q.device, dtype=torch.float32)
    ph = torch.empty((b, splits, h, chunks, 16), device=q.device, dtype=torch.bfloat16)
    pl = torch.empty_like(ph)
    alpha = torch.empty((b, splits, h, chunks), device=q.device, dtype=torch.float32)
    pm = torch.empty((b, splits, h), device=q.device, dtype=torch.float32)
    ps = torch.empty_like(pm)
    partial = torch.empty((b, splits, h, 512), device=q.device, dtype=torch.float32)
    qk = _qk_long_kernel[(1, b, splits)](
        q,
        *q.stride()[:3],
        cache,
        cache.stride(1),
        cache.stride(0),
        table,
        table.stride(0),
        lengths,
        ph,
        0,
        0,
        0,
        lse,
        pm,
        ps,
        pl,
        alpha,
        chunks,
        splits,
        0,
        0,
        scale,
        1,
        h,
        576,
        512,
        768,
        False,
        16,
        16,
        qk_tile,
        cap,
        True,
        MIN_LENGTH=0,
        SHORT_SPLITS=short_splits,
        SHORT_LIMIT=short_limit,
        num_warps=qk_warps,
        num_stages=1,
    )
    pv = _pv_tp8[(b, splits, 512 // pv_vt)](
        cache,
        cache.stride(1),
        cache.stride(0),
        table,
        table.stride(0),
        lengths,
        ph,
        pl,
        alpha,
        partial,
        h,
        splits,
        chunks,
        768,
        cap,
        16,
        pv_vt,
        SHORT_SPLITS=short_splits,
        SHORT_LIMIT=short_limit,
        num_warps=pv_warps,
        num_stages=1,
    )
    merge = _merge_long_kernel[(b, h, 512 // merge_vt)](
        partial,
        pm,
        ps,
        out,
        lse,
        lengths,
        lengths,
        b,
        h,
        512,
        splits,
        splits,
        merge_vt,
        MIN_LENGTH=0,
        SHORT_SPLITS=short_splits,
        SHORT_LIMIT=short_limit,
        num_warps=merge_warps,
        enable_fp_fusion=False,
    )
    if record is not None:
        record.update(
            heads=h,
            page=768,
            splits=splits,
            qk_tile=qk_tile,
            qk_warps=qk_warps,
            short_splits=short_splits,
            short_limit=short_limit,
            workspace_bytes=sum(
                t.numel() * t.element_size() for t in (ph, pl, alpha, pm, ps, partial)
            ),
            kernels={
                name: {
                    "n_regs": k.n_regs,
                    "n_spills": k.n_spills,
                    "shared": k.metadata.shared,
                }
                for name, k in [("qk", qk), ("pv", pv), ("merge", merge)]
            },
        )
    return out, lse
