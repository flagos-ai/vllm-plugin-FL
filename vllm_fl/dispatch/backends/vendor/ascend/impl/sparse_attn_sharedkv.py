# SPDX-License-Identifier: Apache-2.0
"""Paged shared-KV sparse attention with FP32 online softmax.

The caller owns page allocation and query scheduling. Page zero is valid at
this interface: the scheduler adapter converts its null pages to -1 before
dispatch. No metadata values are copied to the host, including during capture.
"""

import torch

from vllm.triton_utils import tl, triton

from .triton_utils import (
    get_aicore_num,
    get_vectorcore_num,
    init_device_properties_triton,
)


@triton.jit
def _paged_rows(
    CACHE,
    TABLE,
    request,
    logical,
    valid,
    cols,
    PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    REQUESTS: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    TABLE_S0: tl.constexpr,
    TABLE_S1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    DIM: tl.constexpr,
    AS_FLOAT: tl.constexpr = True,
):
    block_id = logical // BLOCK
    valid = valid & (request >= 0) & (request < REQUESTS)
    valid = valid & (logical >= 0) & (block_id < TABLE_WIDTH)
    # Keep the full address calculation wide, even for a small view into a
    # large shared cache allocation. All invalid accesses are masked loads.
    safe_request = tl.where((request >= 0) & (request < REQUESTS), request, 0)
    safe_block = tl.where(valid, block_id, 0)
    table_offset = safe_request.to(tl.int64) * TABLE_S0 + safe_block * TABLE_S1
    if (REQUESTS - 1) * TABLE_S0 + (TABLE_WIDTH - 1) * TABLE_S1 < 2**31:
        table_offset = table_offset.to(tl.int32)
    page = tl.load(TABLE + table_offset, mask=valid, other=-1).to(tl.int64)
    valid = valid & (page >= 0) & (page < PAGES)
    safe_page = tl.where(valid, page, 0)
    row = tl.where(valid, logical % BLOCK, 0)
    address = safe_page * S0 + row * S1
    # This proof concerns offsets within this view, not the backing allocation
    # or its absolute device address. Wide views retain the int64 path.
    if (PAGES - 1) * S0 + (BLOCK - 1) * S1 + (DIM - 1) * S2 < 2**31:
        address = address.to(tl.int32)
    if S2 != 1:
        # Express a strided inner dimension as a flat gather, rather than a
        # strided two-dimensional DMA feeding a Cube operand.
        rows: tl.constexpr = logical.shape[0]
        width: tl.constexpr = cols.shape[0]
        flat = tl.arange(0, rows * width)
        column = flat % width
        base = tl.broadcast_to(address[:, None], (rows, width)).reshape(rows * width)
        live = tl.broadcast_to(valid[:, None], (rows, width)).reshape(rows * width)
        values = tl.load(
            CACHE + base + column.to(base.dtype) * S2,
            mask=live & (column < DIM),
            other=0,
        ).reshape(rows, width)
    else:
        values = tl.load(
            CACHE + address[:, None] + cols[None, :],
            mask=valid[:, None] & (cols[None, :] < DIM),
            other=0,
        )
    if AS_FLOAT:
        values = values.to(tl.float32)
    return values, valid


@triton.jit
def _sparse_partial_kernel(
    Q,
    SWA,
    COMPACT,
    ST,
    CT,
    REQUEST_IDS,
    POSITIONS,
    SL,
    CL,
    INDICES,
    PARTIAL,
    MAXIMA,
    SUMS,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
    RATIO: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_COMPACT: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    RS: tl.constexpr,
    PS: tl.constexpr,
    SLS: tl.constexpr,
    CLS: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    SP: tl.constexpr,
    SB: tl.constexpr,
    SR: tl.constexpr,
    SW: tl.constexpr,
    ST0: tl.constexpr,
    ST1: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    CP: tl.constexpr,
    CB: tl.constexpr,
    CR: tl.constexpr,
    CW: tl.constexpr,
    CT0: tl.constexpr,
    CT1: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    CS2: tl.constexpr,
    SLN: tl.constexpr,
    CLN: tl.constexpr,
    SPLITS: tl.constexpr,
    TILES: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
):
    cols = tl.arange(0, BD)
    lanes = tl.arange(0, BK)
    for task in range(tl.program_id(0), TOKENS * HEADS * SPLITS, PROGRAMS):
        split = task % SPLITS
        head = task // SPLITS % HEADS
        token = task // (SPLITS * HEADS)
        req = tl.load(REQUEST_IDS + token * RS).to(tl.int64)
        position = tl.load(POSITIONS + token * PS).to(tl.int64)
        active = (req >= 0) & (position >= 0)
        swa_length = tl.load(
            SL + tl.where((req >= 0) & (req < SLN), req, 0) * SLS,
            mask=(req >= 0) & (req < SLN),
            other=0,
        ).to(tl.int64)
        if HAS_COMPACT:
            compact_length = tl.load(
                CL + tl.where((req >= 0) & (req < CLN), req, 0) * CLS,
                mask=(req >= 0) & (req < CLN),
                other=0,
            ).to(tl.int64)
            compact_limit = tl.minimum((position + 1) // RATIO, compact_length)
        q = tl.load(
            Q + token.to(tl.int64) * QS0 + head * QS1 + cols * QS2,
            mask=active & (cols < DIM),
            other=0,
        ).to(tl.float32)
        maximum = tl.full((), -float("inf"), tl.float32)
        denominator = tl.full((), 0, tl.float32)
        numerator = tl.full((BD,), 0, tl.float32)
        for tile in range(split, TILES, SPLITS):
            if tile < tl.cdiv(WINDOW, BK):
                slot = tile * BK + lanes
                logical = position - WINDOW + 1 + slot
                kv, valid = _paged_rows(
                    SWA,
                    ST,
                    req,
                    logical,
                    active & (slot < WINDOW) & (logical < swa_length),
                    cols,
                    SP,
                    SB,
                    SR,
                    SW,
                    ST0,
                    ST1,
                    SS0,
                    SS1,
                    SS2,
                    DIM,
                )
            else:
                if HAS_COMPACT:
                    slot = (tile - tl.cdiv(WINDOW, BK)) * BK + lanes
                    logical = tl.load(
                        INDICES + token.to(tl.int64) * IS0 + slot * IS1,
                        mask=slot < TOPK,
                        other=-1,
                    ).to(tl.int64)
                    kv, valid = _paged_rows(
                        COMPACT,
                        CT,
                        req,
                        logical,
                        active & (slot < TOPK) & (logical < compact_limit),
                        cols,
                        CP,
                        CB,
                        CR,
                        CW,
                        CT0,
                        CT1,
                        CS0,
                        CS1,
                        CS2,
                        DIM,
                    )
                else:
                    kv = tl.full((BK, BD), 0, tl.float32)
                    valid = tl.full((BK,), False, tl.int1)
            scores = tl.sum(kv * q[None, :], axis=1) * SCALE
            scores = tl.where(valid, scores, -float("inf"))
            next_max = tl.maximum(maximum, tl.max(scores, axis=0))
            # Empty tiles must preserve an empty accumulator without -inf - -inf.
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            correction = tl.exp(maximum - safe_max)
            weights = tl.exp(scores - safe_max)
            denominator = denominator * correction + tl.sum(weights, axis=0)
            numerator = numerator * correction + tl.sum(weights[:, None] * kv, axis=0)
            maximum = next_max
        tl.store(MAXIMA + task, maximum)
        tl.store(SUMS + task, denominator)
        tl.store(
            PARTIAL + task.to(tl.int64) * VALUE_DIM + cols,
            numerator,
            mask=cols < VALUE_DIM,
        )


@triton.jit
def _sparse_cube_partial_kernel(
    Q,
    SWA,
    COMPACT,
    ST,
    CT,
    REQUEST_IDS,
    POSITIONS,
    SL,
    CL,
    INDICES,
    PARTIAL,
    MAXIMA,
    SUMS,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
    RATIO: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_COMPACT: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    RS: tl.constexpr,
    PS: tl.constexpr,
    SLS: tl.constexpr,
    CLS: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    SP: tl.constexpr,
    SB: tl.constexpr,
    SR: tl.constexpr,
    SW: tl.constexpr,
    ST0: tl.constexpr,
    ST1: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    CP: tl.constexpr,
    CB: tl.constexpr,
    CR: tl.constexpr,
    CW: tl.constexpr,
    CT0: tl.constexpr,
    CT1: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    CS2: tl.constexpr,
    SLN: tl.constexpr,
    CLN: tl.constexpr,
    SPLITS: tl.constexpr,
    TILES: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    BH: tl.constexpr,
):
    cols = tl.arange(0, BD)
    lanes = tl.arange(0, BK)
    groups: tl.constexpr = tl.cdiv(HEADS, BH)
    for task in range(tl.program_id(0), TOKENS * groups * SPLITS, PROGRAMS):
        split = task % SPLITS
        group = task // SPLITS % groups
        token = task // (SPLITS * groups)
        heads = group * BH + tl.arange(0, BH)
        req = tl.load(REQUEST_IDS + token * RS).to(tl.int64)
        position = tl.load(POSITIONS + token * PS).to(tl.int64)
        active = (req >= 0) & (position >= 0)
        swa_length = tl.load(
            SL + tl.where((req >= 0) & (req < SLN), req, 0) * SLS,
            mask=(req >= 0) & (req < SLN),
            other=0,
        ).to(tl.int64)
        if HAS_COMPACT:
            compact_length = tl.load(
                CL + tl.where((req >= 0) & (req < CLN), req, 0) * CLS,
                mask=(req >= 0) & (req < CLN),
                other=0,
            ).to(tl.int64)
            compact_limit = tl.minimum((position + 1) // RATIO, compact_length)
        q = tl.load(
            Q + token.to(tl.int64) * QS0 + heads[:, None] * QS1 + cols[None, :] * QS2,
            mask=active & (heads[:, None] < HEADS) & (cols[None, :] < DIM),
            other=0,
        )
        maximum = tl.full((BH,), -float("inf"), tl.float32)
        denominator = tl.full((BH,), 0, tl.float32)
        numerator = tl.full((BH, BD), 0, tl.float32)
        for tile in range(split, TILES, SPLITS):
            if tile < tl.cdiv(WINDOW, BK):
                slot = tile * BK + lanes
                logical = position - WINDOW + 1 + slot
                kv, valid = _paged_rows(
                    SWA,
                    ST,
                    req,
                    logical,
                    active & (slot < WINDOW) & (logical < swa_length),
                    cols,
                    SP,
                    SB,
                    SR,
                    SW,
                    ST0,
                    ST1,
                    SS0,
                    SS1,
                    SS2,
                    DIM,
                )
            else:
                if HAS_COMPACT:
                    slot = (tile - tl.cdiv(WINDOW, BK)) * BK + lanes
                    logical = tl.load(
                        INDICES + token.to(tl.int64) * IS0 + slot * IS1,
                        mask=slot < TOPK,
                        other=-1,
                    ).to(tl.int64)
                    kv, valid = _paged_rows(
                        COMPACT,
                        CT,
                        req,
                        logical,
                        active & (slot < TOPK) & (logical < compact_limit),
                        cols,
                        CP,
                        CB,
                        CR,
                        CW,
                        CT0,
                        CT1,
                        CS0,
                        CS1,
                        CS2,
                        DIM,
                    )
                else:
                    kv = tl.full((BK, BD), 0, tl.float32)
                    valid = tl.full((BK,), False, tl.int1)
            k = kv.to(q.dtype)
            scores = tl.dot(q, tl.trans(k)) * SCALE
            scores = tl.where(valid[None, :], scores, -float("inf"))
            next_max = tl.maximum(maximum, tl.max(scores, axis=1))
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            correction = tl.exp(maximum - safe_max)
            weights = tl.exp(scores - safe_max[:, None])
            denominator = denominator * correction + tl.sum(weights, axis=1)
            # Two low-precision parts retain substantially more of the FP32
            # probability than a single BF16 cast; both dot products accumulate
            # in FP32. Query and shared KV retain their input dtype throughout.
            high = weights.to(q.dtype)
            low = (weights - high.to(tl.float32)).to(q.dtype)
            numerator = tl.dot(high, k, numerator * correction[:, None])
            numerator = tl.dot(low, k, numerator)
            maximum = next_max
        out_task = (token.to(tl.int64) * HEADS + heads) * SPLITS + split
        if TOKENS * HEADS * SPLITS * VALUE_DIM < 2**31:
            out_task = tl.where(heads < HEADS, out_task, 0).to(tl.int32)
        tl.store(MAXIMA + out_task, maximum, mask=heads < HEADS)
        tl.store(SUMS + out_task, denominator, mask=heads < HEADS)
        tl.store(
            PARTIAL + out_task[:, None] * VALUE_DIM + cols[None, :],
            numerator,
            mask=(heads[:, None] < HEADS) & (cols[None, :] < VALUE_DIM),
        )


@triton.jit
def _sparse_merge_kernel(
    PARTIAL,
    MAXIMA,
    SUMS,
    SINK,
    OUTPUT,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SINK_STRIDE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BV: tl.constexpr,
):
    split = tl.arange(0, SPLITS)
    cols = tl.arange(0, BV)
    for row in range(tl.program_id(0), ROWS, PROGRAMS):
        task = row.to(tl.int64) * SPLITS + split
        if ROWS * SPLITS * ROW_STRIDE < 2**31:
            task = task.to(tl.int32)
        maximum = tl.load(MAXIMA + task)
        denominator = tl.load(SUMS + task)
        global_max = tl.max(maximum, axis=0)
        if HAS_SINK:
            sink = tl.load(SINK + (row % HEADS) * SINK_STRIDE).to(tl.float32)
            global_max = tl.maximum(global_max, sink)
        safe_max = tl.where(global_max == -float("inf"), 0.0, global_max)
        correction = tl.exp(maximum - safe_max)
        total = tl.sum(denominator * correction, axis=0)
        if HAS_SINK:
            total += tl.exp(sink - safe_max)
        partial = tl.load(
            PARTIAL + task[:, None] * ROW_STRIDE + cols[None, :],
            mask=cols[None, :] < VALUE_DIM,
            other=0,
        )
        value = tl.sum(partial * correction[:, None], axis=0)
        value = value / tl.where(total > 0, total, 1.0)
        tl.store(
            OUTPUT + row.to(tl.int64) * ROW_STRIDE + cols, value, mask=cols < VALUE_DIM
        )


@triton.jit
def _packed_rows(
    CACHE,
    TABLE,
    request,
    logical,
    valid,
    cols,
    PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    REQUESTS: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    TABLE_S0: tl.constexpr,
    TABLE_S1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    DIM: tl.constexpr,
    limit,
):
    # FP32 classification is exact at these integer endpoints. Incoming
    # int64 indices remain wide until validity is established, including 2**40.
    if REQUESTS <= 2**24:
        rf = request.to(tl.float32)
        request_ok = (rf >= 0.0) & (rf < REQUESTS)
    else:
        request_ok = (request >= 0) & (request < REQUESTS)
    capacity: tl.constexpr = TABLE_WIDTH * BLOCK
    if capacity <= 2**24:
        lf = logical.to(tl.float32)
        bounded_limit = tl.minimum(limit, capacity)
        valid = valid & request_ok & (lf >= 0.0) & (lf < bounded_limit.to(tl.float32))
    else:
        valid = (
            valid
            & request_ok
            & (logical >= 0)
            & (logical < capacity)
            & (logical < limit)
        )
    safe_logical = tl.where(valid, logical, 0)
    safe_request = tl.where(request_ok, request, 0)
    if capacity <= 2**31:
        safe_logical = safe_logical.to(tl.int32)
    block_id = safe_logical // BLOCK
    if (REQUESTS - 1) * TABLE_S0 + (TABLE_WIDTH - 1) * TABLE_S1 < 2**31:
        table_offset = (
            safe_request.to(tl.int32) * TABLE_S0 + block_id.to(tl.int32) * TABLE_S1
        )
    else:
        table_offset = (
            safe_request.to(tl.int64) * TABLE_S0 + block_id.to(tl.int64) * TABLE_S1
        )
    page = tl.load(TABLE + table_offset, mask=valid, other=-1)
    if PAGES <= 2**24:
        pf = page.to(tl.float32)
        valid = valid & (pf >= 0.0) & (pf < PAGES)
    else:
        valid = valid & (page >= 0) & (page < PAGES)
    safe_page = tl.where(valid, page, 0)
    row = safe_logical - block_id * BLOCK
    if (PAGES - 1) * S0 + (BLOCK - 1) * S1 + (DIM - 1) * S2 < 2**31:
        address = safe_page.to(tl.int32) * S0 + row.to(tl.int32) * S1
    else:
        address = safe_page.to(tl.int64) * S0 + row.to(tl.int64) * S1
    values = tl.load(
        CACHE + address[:, None] + cols[None, :] * S2,
        mask=valid[:, None] & (cols[None, :].to(tl.float32) < DIM),
        other=0,
    )
    return values, valid


@triton.jit
def _sparse_pack_kernel(
    SWA,
    COMPACT,
    ST,
    CT,
    REQUEST_IDS,
    POSITIONS,
    SL,
    CL,
    INDICES,
    PACKED,
    VALID,
    TOKENS: tl.constexpr,
    DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
    RATIO: tl.constexpr,
    HAS_COMPACT: tl.constexpr,
    RS: tl.constexpr,
    PS: tl.constexpr,
    SLS: tl.constexpr,
    CLS: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    SP: tl.constexpr,
    SB: tl.constexpr,
    SR: tl.constexpr,
    SW: tl.constexpr,
    ST0: tl.constexpr,
    ST1: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    CP: tl.constexpr,
    CB: tl.constexpr,
    CR: tl.constexpr,
    CW: tl.constexpr,
    CT0: tl.constexpr,
    CT1: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    CS2: tl.constexpr,
    SLN: tl.constexpr,
    CLN: tl.constexpr,
    TILES: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
):
    cols = tl.arange(0, BD)
    lanes = tl.arange(0, BK)
    CAPACITY: tl.constexpr = TILES * BK
    for task in range(tl.program_id(0), TOKENS * TILES, PROGRAMS):
        token = task // TILES
        tile = task % TILES
        req = tl.load(REQUEST_IDS + token * RS).to(tl.int64)
        position = tl.load(POSITIONS + token * PS).to(tl.int64)
        active = (req >= 0) & (position >= 0)
        swa_length = tl.load(
            SL + tl.where((req >= 0) & (req < SLN), req, 0) * SLS,
            mask=(req >= 0) & (req < SLN),
            other=0,
        ).to(tl.int64)
        if tile < tl.cdiv(WINDOW, BK):
            slot = tile * BK + lanes
            logical = position - WINDOW + 1 + slot
            kv, valid = _packed_rows(
                SWA,
                ST,
                req,
                logical,
                active & (slot.to(tl.float32) < WINDOW),
                cols,
                SP,
                SB,
                SR,
                SW,
                ST0,
                ST1,
                SS0,
                SS1,
                SS2,
                DIM,
                swa_length,
            )
        else:
            if HAS_COMPACT:
                compact_length = tl.load(
                    CL + tl.where((req >= 0) & (req < CLN), req, 0) * CLS,
                    mask=(req >= 0) & (req < CLN),
                    other=0,
                ).to(tl.int64)
                limit = tl.minimum((position + 1) // RATIO, compact_length)
                slot = (tile - tl.cdiv(WINDOW, BK)) * BK + lanes
                logical = tl.load(
                    INDICES
                    + token.to(tl.int64) * IS0
                    + tl.where(slot < TOPK, slot, 0) * IS1,
                    mask=slot < TOPK,
                    other=-1,
                ).to(tl.int64)
                kv, valid = _packed_rows(
                    COMPACT,
                    CT,
                    req,
                    logical,
                    active & (slot.to(tl.float32) < TOPK),
                    cols,
                    CP,
                    CB,
                    CR,
                    CW,
                    CT0,
                    CT1,
                    CS0,
                    CS1,
                    CS2,
                    DIM,
                    limit,
                )
            else:
                kv = tl.full((BK, BD), 0, SWA.dtype.element_ty)
                valid = tl.full((BK,), False, tl.int1)
        row = tile * BK + lanes
        tl.store(
            PACKED
            + token.to(tl.int64) * CAPACITY * DIM
            + row[:, None] * DIM
            + cols[None, :],
            kv,
            mask=cols[None, :] < DIM,
        )
        tl.store(VALID + token.to(tl.int64) * CAPACITY + row, valid.to(tl.int8))


@triton.jit
def _sparse_online_packed_kernel(
    Q,
    PACKED,
    VALID,
    SINK,
    OUTPUT,
    TOKENS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SINK_STRIDE: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
):
    heads = tl.arange(0, BH)
    cols = tl.arange(0, BD)
    lanes = tl.arange(0, BK)
    for token in range(tl.program_id(0), TOKENS, PROGRAMS):
        q = tl.load(
            Q + token.to(tl.int64) * QS0 + heads[:, None] * QS1 + cols[None, :] * QS2,
            mask=(heads[:, None] < HEADS) & (cols[None, :] < DIM),
            other=0,
        )
        maximum = tl.full((BH,), -float("inf"), tl.float32)
        denominator = tl.full((BH,), 0, tl.float32)
        numerator = tl.full((BH, BD), 0, tl.float32)
        for tile in range(tl.cdiv(CAPACITY, BK)):
            kp = tl.make_block_ptr(
                base=PACKED + token.to(tl.int64) * CAPACITY * DIM,
                shape=(CAPACITY, DIM),
                strides=(DIM, 1),
                offsets=(tile * BK, 0),
                block_shape=(BK, BD),
                order=(1, 0),
            )
            k = tl.load(kp, boundary_check=(0, 1), padding_option="zero")
            keys = tile * BK + lanes
            valid = (
                tl.load(
                    VALID + token.to(tl.int64) * CAPACITY + keys,
                    mask=keys < CAPACITY,
                    other=0,
                ).to(tl.float32)
                > 0.0
            )
            scores = tl.dot(q, tl.trans(k)) * SCALE
            scores = tl.where(valid[None, :], scores, -float("inf"))
            next_max = tl.maximum(maximum, tl.max(scores, axis=1))
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            correction = tl.exp(maximum - safe_max)
            weights = tl.exp(scores - safe_max[:, None])
            denominator = denominator * correction + tl.sum(weights, axis=1)
            high = weights.to(Q.dtype.element_ty)
            low = (weights - high.to(tl.float32)).to(Q.dtype.element_ty)
            numerator = tl.dot(high, k, numerator * correction[:, None])
            numerator = tl.dot(low, k, numerator)
            maximum = next_max
        global_max = maximum
        if HAS_SINK:
            sink = tl.load(
                SINK + heads * SINK_STRIDE, mask=heads < HEADS, other=-float("inf")
            ).to(tl.float32)
            global_max = tl.maximum(global_max, sink)
        safe_max = tl.where(global_max == -float("inf"), 0.0, global_max)
        correction = tl.exp(maximum - safe_max)
        total = denominator * correction
        if HAS_SINK:
            total += tl.exp(sink - safe_max)
        result = numerator * correction[:, None]
        result = result / tl.where(total > 0.0, total, 1.0)[:, None]
        address = (token.to(tl.int64) * HEADS + heads[:, None]) * ROW_STRIDE + cols[
            None, :
        ]
        tl.store(
            OUTPUT + address,
            result,
            mask=(heads[:, None] < HEADS) & (cols[None, :] < VALUE_DIM),
        )


def _packed_layout(window_size, topk):
    logical_capacity = (triton.cdiv(window_size, 16) + triton.cdiv(topk, 16)) * 16
    block_keys = min(256, triton.next_power_of_2(logical_capacity))
    # Keep physical tail extents compatible with the dense load2d feature tile.
    # Extra rows are explicitly packed as zero with an invalid attention mask.
    alignment = min(128, block_keys)
    capacity = triton.cdiv(logical_capacity, alignment) * alignment
    return capacity // 16, capacity, block_keys, triton.cdiv(capacity, block_keys)


def _launch_packed_chunk(
    query,
    swa_cache,
    compact_cache,
    swa_block_table,
    compact_block_table,
    request_ids,
    positions,
    swa_lengths,
    compact_lengths,
    compact_indices,
    window_size,
    compress_ratio,
    scale,
    attn_sink,
    value_dim,
    row_stride,
    output_storage,
    workspace,
):
    (tokens, heads, dim) = query.shape
    has_compact = compact_cache is not None
    topk = compact_indices.shape[1] if has_compact else 0
    (tiles, capacity, block_keys, splits) = _packed_layout(window_size, topk)
    (packed, valid) = workspace
    compact = compact_cache if has_compact else swa_cache
    ct = compact_block_table if has_compact else swa_block_table
    cl = compact_lengths if has_compact else swa_lengths
    indices = compact_indices if has_compact else request_ids
    init_device_properties_triton()
    programs = min(int(get_vectorcore_num()), tokens * tiles)
    _sparse_pack_kernel[programs,](
        swa_cache,
        compact,
        swa_block_table,
        ct,
        request_ids,
        positions,
        swa_lengths,
        cl,
        indices,
        packed,
        valid,
        tokens,
        dim,
        window_size,
        topk,
        compress_ratio,
        has_compact,
        request_ids.stride(0),
        positions.stride(0),
        swa_lengths.stride(0),
        cl.stride(0),
        indices.stride(0),
        indices.stride(1) if has_compact else 0,
        swa_cache.shape[0],
        swa_cache.shape[1],
        *swa_block_table.shape,
        *swa_block_table.stride(),
        *swa_cache.stride(),
        compact.shape[0],
        compact.shape[1],
        *ct.shape,
        *ct.stride(),
        *compact.stride(),
        swa_lengths.numel(),
        cl.numel(),
        tiles,
        programs,
        16,
        dim,
        enable_fp_fusion=False,
    )
    programs = min(int(get_aicore_num()), tokens)
    _sparse_online_packed_kernel[programs,](
        query,
        packed,
        valid,
        query if attn_sink is None else attn_sink,
        output_storage,
        tokens,
        heads,
        dim,
        value_dim,
        row_stride,
        capacity,
        *query.stride(),
        dim ** (-0.5) if scale is None else scale,
        attn_sink is not None,
        0 if attn_sink is None else attn_sink.stride(0),
        programs,
        block_keys,
        8,
        dim,
        enable_fp_fusion=False,
        multibuffer=False,
    )
    return output_storage[..., :value_dim]


def _packed_prefill(
    query,
    swa_cache,
    compact_cache,
    swa_block_table,
    compact_block_table,
    request_ids,
    positions,
    swa_lengths,
    compact_lengths,
    compact_indices,
    window_size,
    compress_ratio,
    scale,
    attn_sink,
    value_dim,
    row_stride,
    output_storage,
):
    (tokens, _, dim) = query.shape
    chunk = min(tokens, 128)
    topk = 0 if compact_cache is None else compact_indices.shape[1]
    (_, capacity, _, _) = _packed_layout(window_size, topk)
    packed = torch.empty((chunk, capacity, dim), device=query.device, dtype=query.dtype)
    valid = torch.empty((chunk, capacity), device=query.device, dtype=torch.int8)
    workspace = (packed, valid)
    for start in range(0, tokens, chunk):
        stop = min(start + chunk, tokens)
        _launch_packed_chunk(
            query[start:stop],
            swa_cache,
            compact_cache,
            swa_block_table,
            compact_block_table,
            request_ids[start:stop],
            positions[start:stop],
            swa_lengths,
            compact_lengths,
            None if compact_indices is None else compact_indices[start:stop],
            window_size,
            compress_ratio,
            scale,
            attn_sink,
            value_dim,
            row_stride,
            output_storage[start:stop],
            workspace=workspace,
        )
    return output_storage[..., :value_dim]


def deepseek_v4_sparse_attention(
    query,
    swa_cache,
    compact_cache,
    swa_block_table,
    compact_block_table,
    request_ids,
    positions,
    swa_lengths,
    compact_lengths,
    compact_indices,
    window_size,
    compress_ratio=4,
    scale=None,
    attn_sink=None,
    value_dim=None,
):
    """Joint SWA/compact attention, with no Torch computation or fallback."""
    if query.ndim != 3 or query.device.type != "npu":
        raise ValueError("Expected NPU query [tokens, heads, dim]")
    tokens, heads, dim = query.shape
    value_dim = dim if value_dim is None else value_dim
    if window_size < 1 or compress_ratio not in (0, 4, 128):
        raise ValueError("Expected positive window and compression ratio 0, 4 or 128")
    if not 0 < value_dim <= dim or dim > 512 or heads < 1:
        raise ValueError(
            "Expected positive head/value widths within shared KV width <=512"
        )
    if query.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("Expected BF16, FP16 or FP32 query")
    has_compact = compact_cache is not None
    if has_compact and (compress_ratio == 0 or compact_indices is None):
        raise ValueError("Compact attention requires indices and a compression ratio")
    integer = (torch.int32, torch.int64)
    for tensor in (request_ids, positions):
        if tensor.shape != (tokens,) or tensor.dtype not in integer:
            raise ValueError("Expected one integer request ID and position per query")
    caches = [(swa_cache, swa_block_table, swa_lengths)]
    if has_compact:
        caches.append((compact_cache, compact_block_table, compact_lengths))
        if (
            compact_indices.ndim != 2
            or compact_indices.shape[0] != tokens
            or compact_indices.dtype not in integer
        ):
            raise ValueError("Expected integer compact indices [tokens, topk]")
    tensors = [query, request_ids, positions]
    for cache, table, lengths in caches:
        if (
            cache.ndim != 3
            or cache.shape[-1] != dim
            or cache.shape[0] < 1
            or cache.shape[1] < 1
        ):
            raise ValueError(
                "Expected nonempty paged shared-KV cache [pages, block, dim]"
            )
        if table is None or table.ndim != 2 or lengths is None or lengths.ndim != 1:
            raise ValueError(
                "Expected block table [requests, blocks] and request lengths"
            )
        if (
            table.shape[0] != lengths.numel()
            or table.dtype not in integer
            or lengths.dtype not in integer
        ):
            raise ValueError(
                "Block table and integer lengths must describe the same requests"
            )
        if cache.dtype != query.dtype:
            raise TypeError("Shared KV and query must have the same dtype")
        tensors.extend((cache, table, lengths))
    if has_compact:
        tensors.append(compact_indices)
    if attn_sink is not None:
        if attn_sink.shape != (heads,) or not attn_sink.is_floating_point():
            raise ValueError("Expected one floating sink logit per query head")
        tensors.append(attn_sink)
    if any(t.device != query.device or any(s < 0 for s in t.stride()) for t in tensors):
        raise ValueError(
            "Inputs must share the NPU device and have nonnegative strides"
        )
    row_stride = triton.next_power_of_2(value_dim)
    output_storage = torch.empty(
        (tokens, heads, row_stride), device=query.device, dtype=query.dtype
    )
    output = output_storage[..., :value_dim]
    if tokens == 0:
        return output
    topk = compact_indices.shape[1] if has_compact else 0
    use_cube = (
        query.dtype in (torch.bfloat16, torch.float16)
        and dim >= 32
        and swa_cache.stride(-1) == 1
        and (not has_compact or compact_cache.stride(-1) == 1)
    )
    if (
        use_cube
        and dim == 512
        and heads == 8
        and tokens >= 32
        and query.is_contiguous()
    ):
        return _packed_prefill(
            query,
            swa_cache,
            compact_cache,
            swa_block_table,
            compact_block_table,
            request_ids,
            positions,
            swa_lengths,
            compact_lengths,
            compact_indices,
            window_size,
            compress_ratio,
            scale,
            attn_sink,
            value_dim,
            row_stride,
            output_storage,
        )
    bk = 16
    tiles = triton.cdiv(window_size, bk) + triton.cdiv(topk, bk)
    splits = min(8, triton.next_power_of_2(tiles))
    rows = tokens * heads
    partial = torch.empty(
        (rows, splits, row_stride), device=query.device, dtype=torch.float32
    )
    maxima = torch.empty((rows, splits), device=query.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    init_device_properties_triton()
    work = tokens * triton.cdiv(heads, 16) * splits if use_cube else rows * splits
    programs = min(int(get_aicore_num() if use_cube else get_vectorcore_num()), work)
    compact = compact_cache if has_compact else swa_cache
    ct = compact_block_table if has_compact else swa_block_table
    cl = compact_lengths if has_compact else swa_lengths
    indices = compact_indices if has_compact else request_ids
    kernel = _sparse_cube_partial_kernel if use_cube else _sparse_partial_kernel
    extra = {"BH": 16} if use_cube else {}
    kernel[(programs,)](
        query,
        swa_cache,
        compact,
        swa_block_table,
        ct,
        request_ids,
        positions,
        swa_lengths,
        cl,
        indices,
        partial,
        maxima,
        sums,
        tokens,
        heads,
        dim,
        row_stride,
        window_size,
        topk,
        compress_ratio,
        dim**-0.5 if scale is None else scale,
        has_compact,
        *query.stride(),
        request_ids.stride(0),
        positions.stride(0),
        swa_lengths.stride(0),
        cl.stride(0),
        indices.stride(0),
        indices.stride(1) if has_compact else 0,
        swa_cache.shape[0],
        swa_cache.shape[1],
        *swa_block_table.shape,
        *swa_block_table.stride(),
        *swa_cache.stride(),
        compact.shape[0],
        compact.shape[1],
        *ct.shape,
        *ct.stride(),
        *compact.stride(),
        swa_lengths.numel(),
        cl.numel(),
        splits,
        tiles,
        programs,
        bk,
        triton.next_power_of_2(dim),
        **extra,
        enable_fp_fusion=False,
    )
    programs = min(int(get_vectorcore_num()), rows)
    _sparse_merge_kernel[(programs,)](
        partial,
        maxima,
        sums,
        query if attn_sink is None else attn_sink,
        output_storage,
        rows,
        heads,
        value_dim,
        row_stride,
        splits,
        attn_sink is not None,
        0 if attn_sink is None else attn_sink.stride(0),
        programs,
        triton.next_power_of_2(value_dim),
        enable_fp_fusion=False,
    )
    return output
