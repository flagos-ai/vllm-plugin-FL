# Copyright (c) 2026 BAAI. All rights reserved.
"""Paged C4 index scores and logical TopK, entirely in Triton.

Only the [tokens, candidate capacity] score matrix is materialized. Cache page
strides and metadata use int64 addresses, including before masked loads.
"""

import torch
import triton
import triton.language as tl

from .triton_utils import (
    get_aicore_num,
    get_vectorcore_num,
    init_device_properties_triton,
)


@triton.jit
def _metadata(
    BT,
    R,
    POS,
    LEN,
    t,
    j,
    NR: tl.constexpr,
    NL: tl.constexpr,
    NC: tl.constexpr,
    BS: tl.constexpr,
    NP: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    R0: tl.constexpr,
    P0: tl.constexpr,
    L0: tl.constexpr,
):
    req = tl.load(R + t * R0).to(tl.int64)
    pos = tl.load(POS + t * P0).to(tl.int64)
    ok = (req >= 0) & (req < NR) & (req < NL) & (pos >= 0)
    safe_req = tl.where(ok, req, 0)
    length = tl.load(LEN + safe_req * L0, mask=ok, other=0).to(tl.int64)
    # Avoid overflow of pos+1 for arbitrary INT64 metadata.
    limit = pos // 4 + (pos % 4 == 3).to(tl.int64)
    valid = ok & (j < tl.minimum(limit, length)) & (j < NC * BS)
    page = tl.load(
        BT + safe_req * B0 + (j // BS).to(tl.int64) * B1, mask=valid, other=-1
    ).to(tl.int64)
    valid = valid & (page >= 0) & (page < NP)
    return tl.where(valid, page, 0), valid


@triton.jit
def _score(
    Q,
    K,
    W,
    BT,
    R,
    POS,
    LEN,
    QS,
    KS,
    S,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    CAP: tl.constexpr,
    NR: tl.constexpr,
    NL: tl.constexpr,
    NC: tl.constexpr,
    BS: tl.constexpr,
    NP: tl.constexpr,
    NSP: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    R0: tl.constexpr,
    P0: tl.constexpr,
    L0: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QUANT: tl.constexpr,
    HAS_QS: tl.constexpr,
    HAS_KS: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(CAP, BN)
    for work in range(tl.program_id(0), T * tiles, tl.num_programs(0)):
        t = (work // tiles).to(tl.int64)
        tile = work % tiles
        j = tile * BN + tl.arange(0, BN)
        fast: tl.constexpr = (
            QUANT
            and D > 0
            and D <= 128
            and BS >= BN
            and BS % BN == 0
            and K2 == 1
            and Q2 == 1
            and Q1 >= D
            and K1 >= D
            and BS <= 2**31
            and (BS - 1) * K1 + (D - 1) * K2 < 2**31
            and (BH - 1) * Q1 + (BD - 1) * Q2 < 2**31
            and NP > 0
        )

        active = True
        if fast:
            # This tile belongs to one physical page. Keep the global page
            # address scalar INT64; only page-local row/column offsets vary.
            j0 = tile * BN
            pp0, valid0 = _metadata(
                BT, R, POS, LEN, t, j0, NR, NL, NC, BS, NP, B0, B1, R0, P0, L0
            )
            valid0 = valid0 & ((pp0 < NSP) if HAS_KS else True)
            pp0 = tl.where(valid0, pp0, 0)
            active = valid0
        # A page-contained tile shares its page validity. Causal validity is
        # a monotone prefix, so an invalid first candidate invalidates the tile.
        if active:
            page, valid = _metadata(
                BT, R, POS, LEN, t, j, NR, NL, NC, BS, NP, B0, B1, R0, P0, L0
            )
            valid = valid & ((page < NSP) if HAS_KS else True)
            page = tl.where(valid, page, 0)
            h = tl.arange(0, BH).to(tl.int64)
            if fast:
                key_tile = tl.make_block_ptr(
                    base=K + pp0 * K0,
                    shape=(tl.where(valid0, BS, 0), D),
                    strides=(K1, K2),
                    offsets=((j0 % BS).to(tl.int32), 0),
                    block_shape=(BN, BD),
                    order=(1, 0),
                )
                k = tl.load(key_tile, boundary_check=(0, 1), padding_option="zero")
                d = tl.arange(0, BD)
                q = tl.load(
                    Q + t * Q0 + h[:, None] * Q1 + d[None, :],
                    mask=(h[:, None] < H) & (d[None, :] < D),
                    other=0,
                )
                dots = tl.dot(q, tl.trans(k)).to(tl.float32)
            elif QUANT:
                acc = tl.zeros((BH, BN), tl.int32)
                for start in range(triton.cdiv(D, 32)):
                    d = start * 32 + tl.arange(0, 32)
                    q = tl.load(
                        Q + t * Q0 + h[:, None] * Q1 + d[None, :] * Q2,
                        mask=(h[:, None] < H) & (d[None, :] < D),
                        other=0,
                    )
                    # Flat gather avoids the backend's strided 2D DMA lowering.
                    f = tl.arange(0, BN * 32)
                    jj = tile * BN + f // 32
                    pp, vv = _metadata(
                        BT, R, POS, LEN, t, jj, NR, NL, NC, BS, NP, B0, B1, R0, P0, L0
                    )
                    vv = vv & ((pp < NSP) if HAS_KS else True)
                    pp = tl.where(vv, pp, 0)
                    dd = start * 32 + f % 32
                    off = (
                        pp * K0
                        + tl.where(vv, jj % BS, 0).to(tl.int64) * K1
                        + dd.to(tl.int64) * K2
                    )
                    k = tl.load(K + off, mask=vv & (dd < D), other=0).reshape(BN, 32)
                    acc += tl.dot(q, tl.trans(k))
                dots = acc.to(tl.float32)
            else:
                # General floating/mixed input path: exact FP32 vector products, with
                # scales applied after the reduction just as in the INT8 path.
                dots = tl.full((BH, BN), 0, tl.float32)
                for d in range(D):
                    q = tl.load(Q + t * Q0 + h * Q1 + d * Q2, h < H, other=0).to(
                        tl.float32
                    )
                    off = (
                        page * K0
                        + tl.where(valid, j % BS, 0).to(tl.int64) * K1
                        + d * K2
                    )
                    k = tl.load(K + off, valid, other=0).to(tl.float32)
                    dots = dots + q[:, None] * k[None, :]
            if HAS_QS:
                qs = tl.load(QS + t * QS0 + h * QS1, h < H, other=0).to(tl.float32)
                dots = dots * qs[:, None]
            if HAS_KS:
                ks = tl.load(
                    KS + page * S0 + tl.where(valid, j % BS, 0).to(tl.int64) * S1,
                    valid,
                    other=0,
                ).to(tl.float32)
                dots = dots * ks[None, :]
            weight = tl.load(W + t * W0 + h * W1, h < H, other=0).to(tl.float32)
            scores = tl.sum(tl.maximum(dots, 0) * weight[:, None], axis=0)
            tl.store(S + t * CAP + j, tl.where(valid, scores, -float("inf")), j < CAP)
        else:
            tl.store(S + t * CAP + j, -float("inf"), j < CAP)


@triton.jit
def _paired_sort(value, index, N: tl.constexpr):
    r = tl.arange(0, N)
    for level in tl.static_range(1, N.bit_length()):
        for step in tl.static_range(level - 1, -1, -1):
            partner = r ^ (1 << step)
            v = tl.gather(value, partner, axis=0)
            # Ascend gather supports floating payloads. Bitcast transport keeps
            # every ID bit, without numeric FP32 ID rounding.
            i = tl.gather(index.to(tl.float32, bitcast=True), partner, axis=0).to(
                tl.int32, bitcast=True
            )
            better = (v > value) | ((v == value) & (i < index))
            descending = (r & (1 << level)) == 0
            lower = (r & (1 << step)) == 0
            take = tl.where(descending == lower, better, ~better)
            value = tl.where(take, v, value)
            index = tl.where(take, i, index)
    return value, index


@triton.jit(do_not_specialize=["CAP", "NS", "START"])
def _sort_segments(S, V, I, CAP, P: tl.constexpr, NS, START):
    work = tl.program_id(0).to(tl.int64) + START
    t, seg = work // NS, work % NS
    r = tl.arange(0, P)
    j = seg * P + r
    value = tl.load(S + t * CAP + j, j < CAP, other=-float("inf"))
    index = tl.where(j < CAP, j, 2147483647).to(tl.int32)
    value, index = _paired_sort(value, index, P)
    tl.store(V + (t * NS + seg) * P + r, value)
    tl.store(I + (t * NS + seg) * P + r, index)


@triton.jit(do_not_specialize=["NS", "OS", "START"])
def _merge(V, I, OV, OI, P: tl.constexpr, NS, OS, START):
    work = tl.program_id(0).to(tl.int64) + START
    t, seg = work // OS, work % OS
    r = tl.arange(0, 2 * P)
    source = seg * 2 * P + r
    value = tl.load(V + t * NS * P + source, source < NS * P, other=-float("inf"))
    index = tl.load(I + t * NS * P + source, source < NS * P, other=2147483647)
    # Keep global reads contiguous, then form the bitonic sequence on chip.
    reverse = tl.where(r < P, r, 3 * P - 1 - r)
    value = tl.gather(value, reverse, axis=0)
    index = tl.gather(index.to(tl.float32, bitcast=True), reverse, axis=0).to(
        tl.int32, bitcast=True
    )
    for step in tl.static_range((2 * P).bit_length() - 2, -1, -1):
        partner = r ^ (1 << step)
        v = tl.gather(value, partner, axis=0)
        i = tl.gather(index.to(tl.float32, bitcast=True), partner, axis=0).to(
            tl.int32, bitcast=True
        )
        better = (v > value) | ((v == value) & (i < index))
        lower = (r & (1 << step)) == 0
        take = tl.where(lower, better, ~better)
        value = tl.where(take, v, value)
        index = tl.where(take, i, index)
    tl.store(OV + (t * OS + seg) * P + r, value, r < P)
    tl.store(OI + (t * OS + seg) * P + r, index, r < P)


@triton.jit(do_not_specialize=["TOP", "START"])
def _output(V, I, O, TOP, P: tl.constexpr, B: tl.constexpr, START):
    tiles = (TOP + B - 1) // B
    work = tl.program_id(0).to(tl.int64) + START
    t, tile = work // tiles, work % tiles
    r = tile * B + tl.arange(0, B)
    v = tl.load(V + t * P + r, r < P, other=-float("inf"))
    i = tl.load(I + t * P + r, r < P, other=-1)
    tl.store(
        O + t * TOP + r,
        tl.where((v > -float("inf")) & (v < float("inf")), i, -1),
        r < TOP,
    )


def _launch_chunks(kernel, work, *args):
    # Heavy sorting bodies compile much faster without a persistent outer
    # loop. Bound the flat launch dimension independently of input batch size.
    for start in range(0, work, 65535):
        kernel[(min(65535, work - start),)](*args, start)


def _indexer_scores(
    query,
    key_cache,
    weights,
    block_table,
    request_ids,
    positions,
    key_lengths,
    query_scale=None,
    key_scale_cache=None,
):
    """Private score seam for independent high precision operator diagnostics."""
    t, h, d = query.shape
    cap = block_table.shape[1] * key_cache.shape[1]
    scores = torch.empty((t, cap), dtype=torch.float32, device=query.device)
    if t and cap:
        init_device_properties_triton()
        cores = (
            get_aicore_num()
            if query.dtype == key_cache.dtype == torch.int8
            else get_vectorcore_num()
        )
        _score[(min(int(cores), t * triton.cdiv(cap, 32)),)](
            query,
            key_cache,
            weights,
            block_table,
            request_ids,
            positions,
            key_lengths,
            query_scale if query_scale is not None else query,
            key_scale_cache if key_scale_cache is not None else key_cache,
            scores,
            t,
            h,
            d,
            cap,
            block_table.shape[0],
            key_lengths.numel(),
            block_table.shape[1],
            key_cache.shape[1],
            key_cache.shape[0],
            key_scale_cache.shape[0]
            if key_scale_cache is not None
            else key_cache.shape[0],
            *query.stride(),
            *key_cache.stride(),
            *weights.stride(),
            *(key_scale_cache.stride()[:2] if key_scale_cache is not None else (0, 0)),
            *block_table.stride(),
            request_ids.stride(0),
            positions.stride(0),
            key_lengths.stride(0),
            *(
                (h * query_scale.stride(0), query_scale.stride(0))
                if query_scale is not None and query_scale.ndim == 1
                else query_scale.stride()[:2]
                if query_scale is not None
                else (0, 0)
            ),
            query.dtype == torch.int8
            and key_cache.dtype == torch.int8
            and d * 16384 < 2**31,
            query_scale is not None,
            key_scale_cache is not None,
            max(16, triton.next_power_of_2(h)),
            32,
            max(32, triton.next_power_of_2(d)),
            enable_fp_fusion=False,
        )
    return scores


def deepseek_v4_indexer(
    query,
    key_cache,
    weights,
    block_table,
    request_ids,
    positions,
    key_lengths,
    topk,
    compress_ratio=4,
    query_scale=None,
    key_scale_cache=None,
):
    if compress_ratio != 4 or topk < 1:
        raise ValueError("The compressed indexer requires C4 and positive topk")
    if query.ndim != 3 or weights.shape != query.shape[:2]:
        raise ValueError(
            "Expected [tokens, heads, dim] queries and [tokens, heads] weights"
        )
    if key_cache.ndim != 3 or key_cache.shape[2] != query.shape[2]:
        raise ValueError("Expected paged keys [pages, block, query dim]")
    if block_table.ndim != 2:
        raise ValueError("Expected a request block table matrix")
    if any(
        x.ndim != 1 or x.numel() != query.shape[0] for x in (request_ids, positions)
    ):
        raise ValueError("Expected request and position vectors")
    if key_lengths.ndim != 1:
        raise ValueError("Expected key lengths vector")
    metadata = (block_table, request_ids, positions, key_lengths)
    if any(x.dtype not in (torch.int32, torch.int64) for x in metadata):
        raise ValueError("Indexer metadata must be INT32 or INT64")
    tensors = (key_cache, weights, *metadata)
    tensors += tuple(x for x in (query_scale, key_scale_cache) if x is not None)
    if any(x.device != query.device for x in tensors):
        raise ValueError("All indexer tensors must be on the query device")
    supported = (torch.int8, torch.float16, torch.bfloat16, torch.float32)
    if query.dtype not in supported or key_cache.dtype not in supported:
        raise ValueError("Expected INT8 or floating point query/key tensors")
    if query.dtype == torch.int8 and query_scale is None:
        raise ValueError("INT8 queries require per-token/head scales")
    if key_cache.dtype == torch.int8 and key_scale_cache is None:
        raise ValueError("INT8 keys require per-row scales")
    if query_scale is not None and query_scale.shape not in (
        (query.shape[0] * query.shape[1],),
        query.shape[:2],
        (*query.shape[:2], 1),
    ):
        raise ValueError("Expected per-token/head query scales")
    if key_scale_cache is not None and (
        key_scale_cache.ndim != 3
        or key_scale_cache.shape[1:] != (key_cache.shape[1], 1)
    ):
        raise ValueError("Expected paged per-row key scales")
    t = query.shape[0]
    out = torch.empty((t, topk), dtype=torch.int32, device=query.device)
    if not t:
        return out
    scores = _indexer_scores(
        query,
        key_cache,
        weights,
        block_table,
        request_ids,
        positions,
        key_lengths,
        query_scale,
        key_scale_cache,
    )
    cap = scores.shape[1]
    p = triton.next_power_of_2(min(max(topk, 32), max(cap, 1)))
    ns = max(1, triton.cdiv(cap, p))
    values = torch.empty((t, ns, p), dtype=torch.float32, device=query.device)
    indices = torch.empty((t, ns, p), dtype=torch.int32, device=query.device)
    _launch_chunks(_sort_segments, t * ns, scores, values, indices, cap, p, ns)
    while ns > 1:
        os = triton.cdiv(ns, 2)
        nv = torch.empty((t, os, p), dtype=torch.float32, device=query.device)
        ni = torch.empty((t, os, p), dtype=torch.int32, device=query.device)
        _launch_chunks(_merge, t * os, values, indices, nv, ni, p, ns, os)
        values, indices, ns = nv, ni, os
    _launch_chunks(
        _output, t * triton.cdiv(topk, 256), values, indices, out, topk, p, 256
    )
    return out
