"""Plan and execute causal GQA over BF16 or INT8 KV caches.

Batch plans and workspaces are reused across layers. INT8 prefill dequantizes
selected pages into BF16 scratch before executing attention.
"""

import math
from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

from vllm_fl.attention import kv_int8 as int8_kv

_INT8_SCALE_BASE_HALFS = tl.constexpr(int8_kv.DATA_BYTES // 2)


@triton.jit
def _merge(
    Out,
    Partial,
    Stats,
    Groups,
    gid,
    qs: tl.constexpr,
    oh: tl.constexpr,
    M: tl.constexpr,
):
    # Group: output token, KV head, first partial row, split count, valid queries.
    token = tl.load(Groups + gid * 5 + 0)
    head = tl.load(Groups + gid * 5 + 1)
    first = tl.load(Groups + gid * 5 + 2)
    splits = tl.load(Groups + gid * 5 + 3)
    valid_q = tl.load(Groups + gid * 5 + 4)
    r = tl.arange(0, M)
    d = tl.arange(0, 128)
    mx = tl.full((M,), -float("inf"), tl.float32)
    den = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, 128), tl.float32)
    for s in range(splits):
        row = first + s * M + r
        m = tl.load(Stats + row * 2, cache_modifier=".cg")
        l = tl.load(Stats + row * 2 + 1, cache_modifier=".cg")
        a = tl.load(Partial + row[:, None] * 128 + d[None, :], cache_modifier=".cg")
        nm = tl.maximum(mx, m)
        safe = tl.where(nm == -float("inf"), 0.0, nm)
        w0 = tl.exp(mx - safe)
        w1 = tl.exp(m - safe)
        acc = acc * w0[:, None] + a * w1[:, None]
        den = den * w0 + l * w1
        mx = nm
    v = acc / tl.maximum(den[:, None], 1.0e-30)
    tl.store(
        Out
        + (token + r[:, None] // 8) * qs
        + (head * 8 + r[:, None] % 8) * oh
        + d[None, :],
        v,
        mask=(r[:, None] // 8 < valid_q),
    )


@triton.jit
def _cache_page_and_offset(
    BT,
    seq,
    pos,
    bts: tl.constexpr,
    PAGE: tl.constexpr,
    length,
    CONTIG: tl.constexpr = False,
):
    if CONTIG:
        # Scratch pages of one request are contiguous; BT holds per-request bases.
        page = (tl.load(BT + seq) + pos // PAGE).to(tl.int64)
    else:
        page = tl.load(BT + seq * bts + pos // PAGE, pos < length, 0).to(tl.int64)
    return page, pos % PAGE


@triton.jit
def _load_k_tile(
    K,
    page,
    page_offset,
    head,
    d,
    pos,
    length,
    k0: tl.constexpr,
    k1: tl.constexpr,
    k2: tl.constexpr,
):
    return tl.load(
        K + page[None, :] * k0 + page_offset[None, :] * k1 + head * k2 + d[:, None],
        pos[None, :] < length,
        0.0,
    )


@triton.jit
def _load_v_slice(
    V,
    page,
    page_offset,
    head,
    d,
    pos,
    length,
    v0: tl.constexpr,
    v1: tl.constexpr,
    v2: tl.constexpr,
):
    return tl.load(
        V + page[:, None] * v0 + page_offset[:, None] * v1 + head * v2 + d[None, :],
        pos[:, None] < length,
        0.0,
    )


@triton.jit
def _load_kv_scale(
    KS,
    page,
    page_offset,
    head,
    pos,
    length,
    k0: tl.constexpr,
    PAGE: tl.constexpr,
    IS_V: tl.constexpr,
):
    # INT8 scales always start at byte 8192. Sparse pages append metadata
    # after them, so the scale offset is independent of the physical stride.
    base = page * (k0 // 2) + _INT8_SCALE_BASE_HALFS
    return tl.load(
        KS + base + (IS_V * 2 + head) * PAGE + page_offset, pos < length, 0.0
    ).to(tl.float32)


@triton.jit
def _work(
    Tasks,
    task,
    Q,
    K,
    V,
    Out,
    Cu,
    Lens,
    BT,
    Partial,
    Stats,
    scale,
    q0: tl.constexpr,
    q1: tl.constexpr,
    o0: tl.constexpr,
    o1: tl.constexpr,
    k0: tl.constexpr,
    k1: tl.constexpr,
    k2: tl.constexpr,
    v0: tl.constexpr,
    v1: tl.constexpr,
    v2: tl.constexpr,
    bts: tl.constexpr,
    PAGE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_PREFIX: tl.constexpr,
    KS=None,
    KV_INT8: tl.constexpr = False,
    CONTIG: tl.constexpr = False,
    LATE_V: tl.constexpr = False,
):
    # Task: request, query offset, KV head, KV begin/end, partial row, splits.
    base = Tasks + task * 7
    seq = tl.load(base)
    query_start = tl.load(base + 1)
    head = tl.load(base + 2)
    begin = tl.load(base + 3)
    end = tl.load(base + 4)
    partial = tl.load(base + 5)
    splits = tl.load(base + 6)
    start = tl.load(Cu + seq)
    qlen = tl.load(Cu + seq + 1) - start
    length = tl.load(Lens + seq)
    r = tl.arange(0, M)
    d = tl.arange(0, 128)
    n = tl.arange(0, N)
    qp = query_start + r // 8
    valid = qp < qlen
    q = tl.load(
        Q + (start + qp[:, None]) * q0 + (head * 8 + r[:, None] % 8) * q1 + d[None, :],
        valid[:, None],
        0.0,
    )
    mx = tl.full((M,), -float("inf"), tl.float32)
    den = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, 128), tl.float32)
    # Only visit tiles visible to at least one query in this packet.
    limit = tl.minimum(tl.minimum(end, length), length - qlen + query_start + M // 8)
    for tile in range(begin // N, tl.cdiv(limit, N)):
        pos = tile * N + n
        page, page_offset = _cache_page_and_offset(
            BT, seq, pos, bts, PAGE, length, CONTIG
        )
        k = _load_k_tile(K, page, page_offset, head, d, pos, length, k0, k1, k2)
        if not LATE_V:
            v = _load_v_slice(V, page, page_offset, head, d, pos, length, v0, v1, v2)
        if KV_INT8:
            # K scales multiply scores; V scales multiply P below.
            ks = _load_kv_scale(KS, page, page_offset, head, pos, length, k0, PAGE, 0)
            vs = _load_kv_scale(KS, page, page_offset, head, pos, length, k0, PAGE, 1)
            scores = tl.dot(q, k.to(q.dtype)) * (ks * scale)[None, :]
        else:
            scores = tl.dot(q, k) * scale
        mask = valid[:, None] & (pos[None, :] < limit)
        if SPLIT_PREFIX:
            first_partial_tile = (length - qlen + query_start + 1) // N
            if tile >= first_partial_tile:
                mask = mask & (pos[None, :] <= length - qlen + qp[:, None])
        else:
            mask = mask & (pos[None, :] <= length - qlen + qp[:, None])
        scores = tl.where(mask, scores, -float("inf"))
        nm = tl.maximum(mx, tl.max(scores, 1))
        safe = tl.where(nm == -float("inf"), 0.0, nm)
        alpha = tl.exp(mx - safe)
        p = tl.exp(scores - safe[:, None])
        if LATE_V:
            v = _load_v_slice(V, page, page_offset, head, d, pos, length, v0, v1, v2)
        if KV_INT8:
            pv = (p * vs[None, :]).to(q.dtype)
            acc = acc * alpha[:, None] + tl.dot(pv, v.to(q.dtype))
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        den = den * alpha + tl.sum(p, 1)
        mx = nm
    if splits == 1:
        result = acc / tl.maximum(den[:, None], 1.0e-30)
        tl.store(
            Out
            + (start + qp[:, None]) * o0
            + (head * 8 + r[:, None] % 8) * o1
            + d[None, :],
            result,
            valid[:, None],
        )
    else:
        row = partial + r
        tl.store(Partial + row[:, None] * 128 + d[None, :], acc)
        tl.store(Stats + row * 2, mx)
        tl.store(Stats + row * 2 + 1, den)


# Separate PV accumulators reduce peak live registers for wide query tiles.
# Shared work function for tiled prefill and decode.
@triton.jit
def _work_prefill_d64(
    Tasks,
    task,
    Q,
    K,
    V,
    Out,
    Cu,
    Lens,
    BT,
    Partial,
    Stats,
    scale,
    q0: tl.constexpr,
    q1: tl.constexpr,
    o0: tl.constexpr,
    o1: tl.constexpr,
    k0: tl.constexpr,
    k1: tl.constexpr,
    k2: tl.constexpr,
    v0: tl.constexpr,
    v1: tl.constexpr,
    v2: tl.constexpr,
    bts: tl.constexpr,
    PAGE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_PREFIX: tl.constexpr,
    KS=None,
    KV_INT8: tl.constexpr = False,
    CONTIG: tl.constexpr = False,
):
    base = Tasks + task * 7
    seq = tl.load(base)
    query_start = tl.load(base + 1)
    head = tl.load(base + 2)
    begin = tl.load(base + 3)
    end = tl.load(base + 4)
    partial = tl.load(base + 5)
    splits = tl.load(base + 6)
    start = tl.load(Cu + seq)
    qlen = tl.load(Cu + seq + 1) - start
    length = tl.load(Lens + seq)
    r = tl.arange(0, M)
    d = tl.arange(0, 128)
    d0 = tl.arange(0, 64)
    d1 = d0 + 64
    n = tl.arange(0, N)
    qp = query_start + r // 8
    valid = qp < qlen
    q = tl.load(
        Q + (start + qp[:, None]) * q0 + (head * 8 + r[:, None] % 8) * q1 + d[None, :],
        valid[:, None],
        0.0,
    )
    mx = tl.full((M,), -float("inf"), tl.float32)
    den = tl.zeros((M,), tl.float32)
    acc0 = tl.zeros((M, 64), tl.float32)
    acc1 = tl.zeros((M, 64), tl.float32)
    limit = tl.minimum(tl.minimum(end, length), length - qlen + query_start + M // 8)
    for tile in range(begin // N, tl.cdiv(limit, N)):
        pos = tile * N + n
        page, page_offset = _cache_page_and_offset(
            BT, seq, pos, bts, PAGE, length, CONTIG
        )
        k = _load_k_tile(K, page, page_offset, head, d, pos, length, k0, k1, k2)
        if KV_INT8:
            ks = _load_kv_scale(KS, page, page_offset, head, pos, length, k0, PAGE, 0)
            scores = tl.dot(q, k.to(q.dtype)) * (ks * scale)[None, :]
        else:
            scores = tl.dot(q, k) * scale
        mask = valid[:, None] & (pos[None, :] < limit)
        if SPLIT_PREFIX:
            first_partial_tile = (length - qlen + query_start + 1) // N
            if tile >= first_partial_tile:
                mask = mask & (pos[None, :] <= length - qlen + qp[:, None])
        else:
            mask = mask & (pos[None, :] <= length - qlen + qp[:, None])
        scores = tl.where(mask, scores, -float("inf"))
        nm = tl.maximum(mx, tl.max(scores, 1))
        safe = tl.where(nm == -float("inf"), 0.0, nm)
        alpha = tl.exp(mx - safe)
        p = tl.exp(scores - safe[:, None])
        if KV_INT8:
            vs = _load_kv_scale(KS, page, page_offset, head, pos, length, k0, PAGE, 1)
            pv = (p * vs[None, :]).to(q.dtype)
            v0tile = _load_v_slice(
                V, page, page_offset, head, d0, pos, length, v0, v1, v2
            )
            acc0 = acc0 * alpha[:, None] + tl.dot(pv, v0tile.to(q.dtype))
            v1tile = _load_v_slice(
                V, page, page_offset, head, d1, pos, length, v0, v1, v2
            )
            acc1 = acc1 * alpha[:, None] + tl.dot(pv, v1tile.to(q.dtype))
        else:
            v0tile = _load_v_slice(
                V, page, page_offset, head, d0, pos, length, v0, v1, v2
            )
            acc0 = acc0 * alpha[:, None] + tl.dot(p.to(v0tile.dtype), v0tile)
            v1tile = _load_v_slice(
                V, page, page_offset, head, d1, pos, length, v0, v1, v2
            )
            acc1 = acc1 * alpha[:, None] + tl.dot(p.to(v1tile.dtype), v1tile)
        den = den * alpha + tl.sum(p, 1)
        mx = nm
    if splits == 1:
        safe_den = tl.maximum(den[:, None], 1.0e-30)
        out0 = acc0 / safe_den
        out1 = acc1 / safe_den
        base_out = (start + qp[:, None]) * o0 + (head * 8 + r[:, None] % 8) * o1
        tl.store(Out + base_out + d0[None, :], out0, valid[:, None])
        tl.store(Out + base_out + d1[None, :], out1, valid[:, None])
    else:
        row = partial + r
        tl.store(Partial + row[:, None] * 128 + d0[None, :], acc0)
        tl.store(Partial + row[:, None] * 128 + d1[None, :], acc1)
        tl.store(Stats + row * 2, mx)
        tl.store(Stats + row * 2 + 1, den)


@triton.jit
def _fixed_anchor_page_stats(Scratch, ScratchWork, Lens, PageStats, PAGE: tl.constexpr):
    page_head = tl.program_id(0)
    page = page_head // 2
    head = page_head % 2
    seq = tl.load(ScratchWork + page * 2)
    logical_page = tl.load(ScratchWork + page * 2 + 1)
    length = tl.load(Lens + seq)
    slots = tl.arange(0, PAGE)
    dims = tl.arange(0, 128)
    valid_slots = logical_page * PAGE + slots < length
    page_offset = page * PAGE * 512
    k = tl.load(
        Scratch + page_offset + slots[:, None] * 256 + head * 128 + dims[None, :],
        valid_slots[:, None],
        0.0,
    ).to(tl.float32)
    v = tl.load(
        Scratch
        + PAGE * 256
        + page_offset
        + slots[:, None] * 256
        + head * 128
        + dims[None, :],
        valid_slots[:, None],
        0.0,
    ).to(tl.float32)
    k_sq = tl.where(tl.abs(k) < float("inf"), k * k, float("inf"))
    v_abs = tl.where(tl.abs(v) < float("inf"), tl.abs(v), float("inf"))
    kn2 = tl.max(tl.sum(k_sq, 1), 0)
    vmax = tl.max(tl.max(v_abs, 1), 0)
    tl.store(PageStats + page * 4 + head * 2, kn2)
    tl.store(PageStats + page * 4 + head * 2 + 1, vmax)


@triton.jit
def _fixed_anchor_classify(
    Tasks,
    Q,
    Cu,
    Lens,
    ScratchBase,
    PageStats,
    FallbackFlags,
    scale,
    q0: tl.constexpr,
    q1: tl.constexpr,
    PAGE: tl.constexpr,
    M: tl.constexpr,
    MAX_PAGES: tl.constexpr,
):
    task = tl.program_id(0)
    base = Tasks + task * 7
    seq = tl.load(base)
    query_start = tl.load(base + 1)
    head = tl.load(base + 2)
    start = tl.load(Cu + seq)
    qlen = tl.load(Cu + seq + 1) - start
    length = tl.load(Lens + seq)
    r = tl.arange(0, M)
    d = tl.arange(0, 128)
    qp = query_start + r // 8
    valid = qp < qlen
    q = tl.load(
        Q + (start + qp[:, None]) * q0 + (head * 8 + r[:, None] % 8) * q1 + d[None, :],
        valid[:, None],
        0.0,
    )
    # Guard over the full request context so every split for this query/head
    # takes the same path; a masked or empty split must not mix anchor 0 with
    # online-softmax maxima in the reducer.
    page = tl.arange(0, MAX_PAGES)
    page_count = tl.cdiv(length, PAGE)
    scratch_base = tl.load(ScratchBase + seq)
    kn2_pages = tl.load(
        PageStats + (scratch_base + page) * 4 + head * 2,
        page < page_count,
        0.0,
    )
    vmax_pages = tl.load(
        PageStats + (scratch_base + page) * 4 + head * 2 + 1,
        page < page_count,
        0.0,
    )
    knorm = tl.sqrt(tl.max(kn2_pages, 0))
    vmax = tl.max(vmax_pages, 0)
    qnorm = tl.sqrt(tl.sum(q.to(tl.float32) * q.to(tl.float32), 1))
    row_bound = qnorm * knorm * tl.abs(scale)
    bound = tl.max(tl.where(valid, row_bound, 0.0), 0)
    finite_rows = tl.sum((valid & (tl.abs(row_bound) < float("inf"))).to(tl.int32), 0)
    valid_rows = tl.sum(valid.to(tl.int32), 0)
    guard_ok = (
        (finite_rows == valid_rows)
        & (tl.abs(knorm) < float("inf"))
        & (tl.abs(vmax) < float("inf"))
        & (tl.abs(bound) < float("inf"))
        & (bound <= 64.0)
        & (vmax <= 1024.0)
        & (length >= 4096)
        & (length <= 65536)
    )
    # |scaled score|<=64 bounds p to [e^-64,e^64]; a causal key keeps den
    # above 1e-30, and length<=65536 with |V|<=1024 keeps den/numerator below FP32 max.
    tl.store(FallbackFlags + task, (~guard_ok).to(tl.int32))


@triton.jit
def _fixed_anchor_fast(
    Tasks,
    task,
    Q,
    K,
    V,
    Out,
    Cu,
    Lens,
    BT,
    Partial,
    Stats,
    scale,
    q0: tl.constexpr,
    q1: tl.constexpr,
    o0: tl.constexpr,
    o1: tl.constexpr,
    k0: tl.constexpr,
    k1: tl.constexpr,
    k2: tl.constexpr,
    v0: tl.constexpr,
    v1: tl.constexpr,
    v2: tl.constexpr,
    bts: tl.constexpr,
    PAGE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT_PREFIX: tl.constexpr,
):
    base = Tasks + task * 7
    seq = tl.load(base)
    query_start = tl.load(base + 1)
    head = tl.load(base + 2)
    begin = tl.load(base + 3)
    end = tl.load(base + 4)
    partial = tl.load(base + 5)
    splits = tl.load(base + 6)
    start = tl.load(Cu + seq)
    qlen = tl.load(Cu + seq + 1) - start
    length = tl.load(Lens + seq)
    r = tl.arange(0, M)
    d = tl.arange(0, 128)
    n = tl.arange(0, N)
    qp = query_start + r // 8
    valid = qp < qlen
    q = tl.load(
        Q + (start + qp[:, None]) * q0 + (head * 8 + r[:, None] % 8) * q1 + d[None, :],
        valid[:, None],
        0.0,
    )
    limit = tl.minimum(tl.minimum(end, length), length - qlen + query_start + M // 8)
    acc = tl.zeros((M, 128), tl.float32)
    den_lanes = tl.zeros((M, N), tl.float32)
    for tile in range(begin // N, tl.cdiv(limit, N)):
        pos = tile * N + n
        page_ids, page_offsets = _cache_page_and_offset(
            BT, seq, pos, bts, PAGE, length, True
        )
        k = _load_k_tile(K, page_ids, page_offsets, head, d, pos, length, k0, k1, k2)
        scores = tl.dot(q, k) * (scale * 1.4426950408889634)
        mask = valid[:, None] & (pos[None, :] < limit)
        if SPLIT_PREFIX:
            first_partial_tile = (length - qlen + query_start + 1) // N
            if tile >= first_partial_tile:
                mask = mask & (pos[None, :] <= length - qlen + qp[:, None])
        else:
            mask = mask & (pos[None, :] <= length - qlen + qp[:, None])
        scores = tl.where(mask, scores, -float("inf"))
        p = tl.exp2(scores)
        v = _load_v_slice(V, page_ids, page_offsets, head, d, pos, length, v0, v1, v2)
        acc += tl.dot(p.to(v.dtype), v)
        den_lanes += p
    den = tl.sum(den_lanes, 1)
    if splits == 1:
        result = acc / tl.maximum(den[:, None], 1.0e-30)
        tl.store(
            Out
            + (start + qp[:, None]) * o0
            + (head * 8 + r[:, None] % 8) * o1
            + d[None, :],
            result,
            valid[:, None],
        )
    else:
        row = partial + r
        tl.store(Partial + row[:, None] * 128 + d[None, :], acc)
        tl.store(Stats + row * 2, 0.0)
        tl.store(Stats + row * 2 + 1, den)


@triton.jit
def _execute(
    P,
    D,
    np,
    nd,
    Q,
    K,
    V,
    Out,
    Cu,
    Lens,
    BT,
    Partial,
    Stats,
    State,
    scale,
    q0: tl.constexpr,
    q1: tl.constexpr,
    o0: tl.constexpr,
    o1: tl.constexpr,
    k0: tl.constexpr,
    k1: tl.constexpr,
    k2: tl.constexpr,
    v0: tl.constexpr,
    v1: tl.constexpr,
    v2: tl.constexpr,
    bts: tl.constexpr,
    PAGE: tl.constexpr,
    PM: tl.constexpr,
    PN: tl.constexpr,
    DN: tl.constexpr,
    MODE: tl.constexpr,
    RATIO: tl.constexpr,
    SPLIT_PREFIX: tl.constexpr,
    PREFILL_REBLOCK: tl.constexpr = False,
    KS=None,
    KV_INT8: tl.constexpr = False,
    KB=None,
    SB=None,
    PREFILL_SCRATCH: tl.constexpr = False,
    PREFILL_FIXED_ANCHOR: tl.constexpr = False,
    FIXED_FALLBACK_FLAGS=None,
    LATE_V: tl.constexpr = False,
):
    idx = tl.program_id(0)
    if PREFILL_FIXED_ANCHOR:
        if idx < np:
            if tl.load(FIXED_FALLBACK_FLAGS + idx) != 0:
                _work(
                    P,
                    idx,
                    Q,
                    KB,
                    KB + PAGE * 256,
                    Out,
                    Cu,
                    Lens,
                    SB,
                    Partial,
                    Stats,
                    scale,
                    q0,
                    q1,
                    o0,
                    o1,
                    PAGE * 512,
                    256,
                    128,
                    PAGE * 512,
                    256,
                    128,
                    0,
                    PAGE,
                    PM,
                    PN,
                    SPLIT_PREFIX,
                    None,
                    False,
                    True,
                    True,
                )
            else:
                _fixed_anchor_fast(
                    P,
                    idx,
                    Q,
                    KB,
                    KB + PAGE * 256,
                    Out,
                    Cu,
                    Lens,
                    SB,
                    Partial,
                    Stats,
                    scale,
                    q0,
                    q1,
                    o0,
                    o1,
                    PAGE * 512,
                    256,
                    128,
                    PAGE * 512,
                    256,
                    128,
                    0,
                    PAGE,
                    PM,
                    PN,
                    SPLIT_PREFIX,
                )
    else:
        if MODE == 0:
            op = 0
        elif MODE == 1:
            op = 1
        elif KV_INT8:
            # Closed-form interleaving has no cross-program atomics.
            paired = tl.minimum(np // RATIO, nd)
            paired_ctas = paired * (RATIO + 1)
            if idx < paired_ctas:
                group = idx // (RATIO + 1)
                lane = idx % (RATIO + 1)
                op = tl.where(lane < RATIO, 0, 1)
                idx = tl.where(op == 0, group * RATIO + lane, group)
            else:
                tail = idx - paired_ctas
                left = np - paired * RATIO
                op = tl.where(tail < left, 0, 1)
                idx = tl.where(op == 0, paired * RATIO + tail, paired + tail - left)
        else:
            # One global ticket interleaves RATIO prefill : 1 decode in launch
            # order (portable; no per-SM id, which would need vendor inline asm).
            ticket = tl.atomic_add(State, 1, sem="relaxed")
            op = tl.where(ticket % (RATIO + 1) < RATIO, 0, 1)
            idx = tl.atomic_add(State + 1024 + op, 1, sem="relaxed")
            if op == 0:
                if idx >= np:
                    op = 1
                    idx = tl.atomic_add(State + 1025, 1, sem="relaxed")
            else:
                if idx >= nd:
                    op = 0
                    idx = tl.atomic_add(State + 1024, 1, sem="relaxed")
        if op == 0:
            if idx < np:
                if PREFILL_SCRATCH:
                    # INT8 KV: prefill tasks read the per-layer BF16 scratch (fixed
                    # (2, PAGE, 2, 128) page layout, contiguous pages per request).
                    if PREFILL_REBLOCK:
                        _work_prefill_d64(
                            P,
                            idx,
                            Q,
                            KB,
                            KB + PAGE * 256,
                            Out,
                            Cu,
                            Lens,
                            SB,
                            Partial,
                            Stats,
                            scale,
                            q0,
                            q1,
                            o0,
                            o1,
                            PAGE * 512,
                            256,
                            128,
                            PAGE * 512,
                            256,
                            128,
                            0,
                            PAGE,
                            PM,
                            PN,
                            SPLIT_PREFIX,
                            None,
                            False,
                            True,
                        )
                    else:
                        _work(
                            P,
                            idx,
                            Q,
                            KB,
                            KB + PAGE * 256,
                            Out,
                            Cu,
                            Lens,
                            SB,
                            Partial,
                            Stats,
                            scale,
                            q0,
                            q1,
                            o0,
                            o1,
                            PAGE * 512,
                            256,
                            128,
                            PAGE * 512,
                            256,
                            128,
                            0,
                            PAGE,
                            PM,
                            PN,
                            SPLIT_PREFIX,
                            None,
                            False,
                            True,
                            LATE_V,
                        )
                elif PREFILL_REBLOCK:
                    _work_prefill_d64(
                        P,
                        idx,
                        Q,
                        K,
                        V,
                        Out,
                        Cu,
                        Lens,
                        BT,
                        Partial,
                        Stats,
                        scale,
                        q0,
                        q1,
                        o0,
                        o1,
                        k0,
                        k1,
                        k2,
                        v0,
                        v1,
                        v2,
                        bts,
                        PAGE,
                        PM,
                        PN,
                        SPLIT_PREFIX,
                        KS,
                        KV_INT8,
                    )
                else:
                    _work(
                        P,
                        idx,
                        Q,
                        K,
                        V,
                        Out,
                        Cu,
                        Lens,
                        BT,
                        Partial,
                        Stats,
                        scale,
                        q0,
                        q1,
                        o0,
                        o1,
                        k0,
                        k1,
                        k2,
                        v0,
                        v1,
                        v2,
                        bts,
                        PAGE,
                        PM,
                        PN,
                        SPLIT_PREFIX,
                        KS,
                        KV_INT8,
                    )
        elif op == 1:  # noqa: SIM102 - Keep Triton dispatch and bounds checks separate.
            if idx < nd:
                _work(
                    D,
                    idx,
                    Q,
                    K,
                    V,
                    Out,
                    Cu,
                    Lens,
                    BT,
                    Partial,
                    Stats,
                    scale,
                    q0,
                    q1,
                    o0,
                    o1,
                    k0,
                    k1,
                    k2,
                    v0,
                    v1,
                    v2,
                    bts,
                    PAGE,
                    16,
                    DN,
                    SPLIT_PREFIX,
                    KS,
                    KV_INT8,
                )


@triton.jit
def _reduce(
    Out,
    Partial,
    Stats,
    Groups,
    Ids,
    o0: tl.constexpr,
    o1: tl.constexpr,
    M: tl.constexpr,
):
    gid = tl.load(Ids + tl.program_id(0))
    _merge(Out, Partial, Stats, Groups, gid, o0, o1, M)


@dataclass
class HybridPlan:
    prefill: torch.Tensor
    decode: torch.Tensor
    groups: torch.Tensor
    prefill_merge: torch.Tensor
    decode_merge: torch.Tensor
    partial: torch.Tensor
    stats: torch.Tensor
    state: torch.Tensor
    pm: int
    pn: int
    dn: int
    ratio: int
    split_prefix_mask: bool
    prefill_reblock: bool
    prefill_wide: bool = False
    # True: decode rows (query length 1) have no plan tasks; a standalone decode
    # kernel serves them (mixed "separate" execution).
    decode_external: bool = False
    # INT8 KV only: per-layer BF16 scratch for prefill requests.
    scratch_work: torch.Tensor | None = None  # (P, 2) request, logical page
    scratch_base: torch.Tensor | None = None  # (num_reqs,) first scratch page
    scratch: torch.Tensor | None = None  # (P, 2, 16, 2, 128) BF16
    prefill_fixed_anchor: bool = False
    fixed_anchor_page_stats: torch.Tensor | None = None
    fixed_anchor_fallback_flags: torch.Tensor | None = None

    @classmethod
    def build(
        cls,
        query_lengths,
        kv_lengths,
        *,
        device="cuda",
        pm=64,
        pn=64,
        dn=32,
        segments=4,
        prefill_segments=1,
        layers=1,
        ratio=1,
        workspace=None,
        split_prefix_mask=False,
        prefill_order="natural",
        prefill_reblock=False,
        prefill_wide=False,
        kv_int8_scratch_tokens=0,
        decode_external=False,
        prefill_fixed_anchor=False,
    ):
        has_prefill = any(q > 1 for q in query_lengths)
        has_decode_rows = any(q == 1 for q in query_lengths)
        scratch_pages = sum(
            math.ceil(length / int8_kv.PAGE_TOKENS)
            for q, length in zip(query_lengths, kv_lengths)
            if q > 1
        )
        prefill_wide = bool(
            prefill_wide
            and (pm, pn, prefill_reblock) == (128, 32, True)
            and kv_int8_scratch_tokens > 0
            and has_prefill
            and scratch_pages > 0
            and scratch_pages * int8_kv.PAGE_TOKENS <= kv_int8_scratch_tokens
            and (decode_external or not has_decode_rows)
        )
        if prefill_wide:
            pm, pn, prefill_reblock = 128, 64, False
        prefill_fixed_anchor = bool(
            prefill_fixed_anchor
            and prefill_wide
            and kv_int8_scratch_tokens > 0
            and has_prefill
            and all(
                4096 <= length <= 65536
                for qlen, length in zip(query_lengths, kv_lengths)
                if qlen > 1
            )
        )
        if pm not in (32, 64, 128) or pn not in (32, 64, 128) or dn not in (32, 64):
            raise ValueError("unsupported tile")
        if (
            len(query_lengths) != len(kv_lengths)
            or segments < 1
            or prefill_segments < 1
            or layers < 1
            or ratio < 1
            or prefill_order not in ("natural", "lpt")
            or (prefill_reblock and (pm, pn) != (128, 32))
        ):
            raise ValueError("invalid lengths/splits")
        prefill_tasks, decode_tasks, merge_groups, prefill_merge, decode_merge = (
            [],
            [],
            [],
            [],
            [],
        )
        rows = token = 0
        for seq, (qlen, length) in enumerate(zip(query_lengths, kv_lengths)):
            if not 0 <= qlen <= length:
                raise ValueError("require 0 <= query length <= KV length")
            if not qlen:
                continue
            if qlen == 1 and decode_external:
                token += 1
                continue
            m, n, s = (16, dn, segments) if qlen == 1 else (pm, pn, prefill_segments)
            s = min(s, max(1, math.ceil(length / n)))
            span = math.ceil(math.ceil(length / n) / s) * n
            for qb in range(0, qlen, m // 8):
                for head in range(2):
                    queue = decode_tasks if qlen == 1 else prefill_tasks
                    for split in range(s):
                        queue.append(
                            [
                                seq,
                                qb,
                                head,
                                split * span,
                                (split + 1) * span,
                                rows + split * m,
                                s,
                            ]
                        )
                    if s > 1:
                        gid = len(merge_groups)
                        merge_groups.append(
                            [token + qb, head, rows, s, min(m // 8, qlen - qb)]
                        )
                        (decode_merge if qlen == 1 else prefill_merge).append(gid)
                        rows += s * m
            token += qlen

        if prefill_order == "lpt":

            def estimated_tiles(task):
                seq, query_start, _, begin, end, _, _ = task
                qlen = query_lengths[seq]
                length = kv_lengths[seq]
                limit = min(end, length, length - qlen + query_start + pm // 8)
                return max(0, math.ceil(limit / pn) - begin // pn)

            prefill_tasks.sort(key=estimated_tiles, reverse=True)

        def tensor(name, x, width=1):
            host = (
                torch.tensor(
                    x, dtype=torch.int32, pin_memory=workspace is not None
                ).reshape(-1, width)
                if width > 1
                else torch.tensor(
                    x, dtype=torch.int32, pin_memory=workspace is not None
                )
            )
            if workspace is None:
                return host.to(device)
            dst = workspace.get(name, host.shape, torch.int32)
            dst.copy_(host, non_blocking=True)
            return dst

        def allocate(name, shape, dtype=torch.float32):
            return (
                torch.empty(shape, device=device, dtype=dtype)
                if workspace is None
                else workspace.get(name, shape, dtype)
            )

        scratch = {}
        if kv_int8_scratch_tokens > 0 and prefill_tasks:
            # Dequantize every page of each prefill request once per layer;
            # skipped (direct INT8 reads) if the batch exceeds the cap.
            pages = [
                math.ceil(length / int8_kv.PAGE_TOKENS) if qlen > 1 else 0
                for qlen, length in zip(query_lengths, kv_lengths)
            ]
            used = sum(pages)
            if used * int8_kv.PAGE_TOKENS <= kv_int8_scratch_tokens:
                work, base, offset = [], [0] * len(pages), 0
                for seq, count in enumerate(pages):
                    if count:
                        base[seq] = offset
                        work.extend([seq, lp] for lp in range(count))
                        offset += count
                scratch = dict(
                    scratch_work=tensor("scratch_work", work, 2),
                    scratch_base=tensor("scratch_base", base),
                    scratch=int8_kv.prefill_scratch(device, kv_int8_scratch_tokens)[
                        :used
                    ],
                )

        fixed_anchor_page_stats = None
        fixed_anchor_fallback_flags = None
        if prefill_fixed_anchor and scratch.get("scratch") is not None:
            fixed_anchor_page_stats = allocate(
                "fixed_anchor_page_stats",
                (scratch["scratch"].shape[0], 2, 2),
                torch.float32,
            )
            fixed_anchor_fallback_flags = allocate(
                "fixed_anchor_fallback_flags",
                (max(len(prefill_tasks), 1),),
                torch.int32,
            )
        else:
            prefill_fixed_anchor = False

        # INT8 uses closed-form interleaving and does not use the scheduling counter.
        state_rows = (
            0
            if kv_int8_scratch_tokens > 0
            else (layers if prefill_tasks and decode_tasks else 0)
        )
        return cls(
            prefill=tensor("prefill_tasks", prefill_tasks, 7),
            decode=tensor("decode_tasks", decode_tasks, 7),
            groups=tensor("merge_groups", merge_groups, 5),
            prefill_merge=tensor("prefill_merge", prefill_merge),
            decode_merge=tensor("decode_merge", decode_merge),
            partial=allocate("partial", (max(rows, 1), 128)),
            stats=allocate("stats", (max(rows, 1), 2)),
            state=allocate(
                "state",
                (state_rows, 1026),
                torch.int32,
            ),
            pm=pm,
            pn=pn,
            dn=dn,
            ratio=ratio,
            split_prefix_mask=split_prefix_mask,
            prefill_reblock=prefill_reblock,
            prefill_wide=prefill_wide,
            decode_external=decode_external,
            **scratch,
            prefill_fixed_anchor=prefill_fixed_anchor,
            fixed_anchor_page_stats=fixed_anchor_page_stats,
            fixed_anchor_fallback_flags=fixed_anchor_fallback_flags,
        )

    def reset(self):
        if self.state.numel():
            self.state.zero_()

    def run(self, kwargs, *, mode="fused", layer=0):
        if mode not in ("fused", "separate"):
            raise ValueError("unknown execution mode")
        q, k, v, out = (kwargs[n] for n in ("q", "k", "v", "out"))
        kv_scales = kwargs.get("kv_scales")
        kv_int8 = kv_scales is not None
        kv_dtype = torch.int8 if kv_int8 else q.dtype
        if (
            q.dtype != torch.bfloat16
            or k.dtype != kv_dtype
            or v.dtype != kv_dtype
            or out.dtype != q.dtype
            or q.shape[1:] != (16, 128)
            or k.shape[2:] != (2, 128)
            or v.shape != k.shape
            or out.shape != q.shape
        ):
            raise ValueError("only BF16 16Q:2KV:d128 (BF16 or INT8 KV) is supported")
        if kv_int8 and (
            kv_scales.dtype != torch.float16
            or k.stride()[:3]
            not in (
                (
                    int8_kv.PAGE_BYTES,
                    int8_kv.NUM_KV_HEADS * int8_kv.HEAD_DIM,
                    int8_kv.HEAD_DIM,
                ),
                (
                    int8_kv.SPARSE_PAGE_BYTES,
                    int8_kv.NUM_KV_HEADS * int8_kv.HEAD_DIM,
                    int8_kv.HEAD_DIM,
                ),
            )
            or v.stride() != k.stride()
            or v.data_ptr() - k.data_ptr() != int8_kv.DATA_BYTES // 2
            or kv_scales.data_ptr() != k.data_ptr()
        ):
            raise ValueError(
                "INT8 KV requires the plugin page layout and its fp16 scale base"
            )
        if (
            q.stride(-1) != 1
            or k.stride(-1) != 1
            or v.stride(-1) != 1
            or out.stride(-1) != 1
        ):
            raise ValueError("last dimension must be contiguous")
        num_prefill_tasks, num_decode_tasks = len(self.prefill), len(self.decode)
        args = (
            self.prefill,
            self.decode,
            num_prefill_tasks,
            num_decode_tasks,
            q,
            k,
            v,
            out,
            kwargs["cu_seqlens_q"],
            kwargs["seqused_k"],
            kwargs["block_table"],
            self.partial,
            self.stats,
            self.state[layer] if self.state.numel() else self.state,
            kwargs["softmax_scale"],
            q.stride(0),
            q.stride(1),
            out.stride(0),
            out.stride(1),
            *k.stride()[:3],
            *v.stride()[:3],
            kwargs["block_table"].stride(0),
            k.shape[1],
            self.pm,
            self.pn,
            self.dn,
        )
        # Pass the INT8 options only when used, so BF16 launches keep their
        # original specialization.
        extra = dict(KS=kv_scales, KV_INT8=True) if kv_int8 else {}
        if kv_int8 and num_prefill_tasks and self.scratch is not None:
            int8_kv.kv_int8_dequant_pages(
                k, kv_scales, kwargs["block_table"], self.scratch_work, self.scratch
            )
            extra.update(KB=self.scratch, SB=self.scratch_base, PREFILL_SCRATCH=True)
            if self.prefill_fixed_anchor:
                _fixed_anchor_page_stats[(self.scratch.shape[0] * 2,)](
                    self.scratch,
                    self.scratch_work,
                    kwargs["seqused_k"],
                    self.fixed_anchor_page_stats,
                    PAGE=int8_kv.PAGE_TOKENS,
                    num_warps=4,
                    num_stages=1,
                )
                _fixed_anchor_classify[(num_prefill_tasks,)](
                    self.prefill,
                    q,
                    kwargs["cu_seqlens_q"],
                    kwargs["seqused_k"],
                    self.scratch_base,
                    self.fixed_anchor_page_stats,
                    self.fixed_anchor_fallback_flags,
                    kwargs["softmax_scale"],
                    q.stride(0),
                    q.stride(1),
                    PAGE=int8_kv.PAGE_TOKENS,
                    M=self.pm,
                    MAX_PAGES=4096,
                    num_warps=4,
                    num_stages=1,
                )
        if mode == "fused" and num_prefill_tasks and num_decode_tasks:
            _execute[(num_prefill_tasks + num_decode_tasks,)](
                *args,
                MODE=2,
                RATIO=self.ratio,
                SPLIT_PREFIX=self.split_prefix_mask,
                PREFILL_REBLOCK=self.prefill_reblock,
                **extra,
                num_warps=4,
                num_stages=1 if kv_int8 else 2,
            )
        else:
            if num_prefill_tasks:
                fixed_anchor_extra = (
                    dict(
                        PREFILL_FIXED_ANCHOR=True,
                        FIXED_FALLBACK_FLAGS=self.fixed_anchor_fallback_flags,
                    )
                    if self.prefill_fixed_anchor
                    and kv_int8
                    and self.scratch is not None
                    else {}
                )
                _execute[(num_prefill_tasks,)](
                    *args,
                    MODE=0,
                    RATIO=self.ratio,
                    SPLIT_PREFIX=self.split_prefix_mask,
                    PREFILL_REBLOCK=self.prefill_reblock,
                    LATE_V=self.prefill_wide and kv_int8 and self.scratch is not None,
                    **extra,
                    **fixed_anchor_extra,
                    num_warps=8
                    if self.prefill_wide and kv_int8 and self.scratch is not None
                    else 4,
                    num_stages=1 if kv_int8 else 2,
                )
            if num_decode_tasks:
                _execute[(num_decode_tasks,)](
                    *args,
                    MODE=1,
                    RATIO=self.ratio,
                    SPLIT_PREFIX=self.split_prefix_mask,
                    PREFILL_REBLOCK=self.prefill_reblock,
                    LATE_V=False,
                    **extra,
                    num_warps=4,
                    num_stages=1 if kv_int8 else 2,
                )
        for ids, m in ((self.prefill_merge, self.pm), (self.decode_merge, 16)):
            if len(ids):
                _reduce[(len(ids),)](
                    out,
                    self.partial,
                    self.stats,
                    self.groups,
                    ids,
                    out.stride(0),
                    out.stride(1),
                    m,
                    num_warps=4,
                )


class HybridWorkspace:
    """Capacity reuse for sequential launches on one execution stream.

    Each concurrent execution lane needs its own instance. Growing buffers
    is forbidden inside graph capture; the builder uses baseline capture.
    """

    def __init__(self, device):
        self.device = device
        self.buffers = {}

    def get(self, name, shape, dtype):
        size = math.prod(shape)
        previous = self.buffers.get(name)
        if previous is None or previous.numel() < size:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("hybrid workspace growth during capture")
            self.buffers[name] = torch.empty(
                max(1, triton.next_power_of_2(size)), dtype=dtype, device=self.device
            )
        return self.buffers[name][:size].view(shape)
