# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build graph-stable IndexPool request and cache copy plans on device."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class IndexPoolPlan:
    group_positions: torch.Tensor
    gather_key_rows: torch.Tensor
    gather_tail_slots: torch.Tensor
    pool_slots: torch.Tensor
    tail_key_rows: torch.Tensor
    tail_slots: torch.Tensor
    owner: torch.Tensor
    logical: torch.Tensor
    upper: torch.Tensor
    live: torch.Tensor


@triton.jit
def _group_prefix(
    Starts,
    Lengths,
    Ends,
    BATCH: tl.constexpr,
    TOKENS: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.arange(0, BLOCK)
    start = tl.load(Starts + request, request < BATCH, 0)
    end = tl.load(Starts + request + 1, request < BATCH, 0)
    length = tl.load(Lengths + request, request < BATCH, 0)
    context = length - (end - start)
    live = (
        (request < BATCH)
        & (start >= 0)
        & (end > start)
        & (end <= TOKENS)
        & (context >= 0)
    )
    count = tl.where(live, length // R - tl.maximum(context, 0) // R, 0)
    tl.store(Ends + request, tl.cumsum(count), request < BATCH)


@triton.jit
def _group_plan(
    Positions,
    Tail,
    PoolTable,
    TailTable,
    Starts,
    Lengths,
    Ends,
    XYZ,
    KeyRows,
    TailSlots,
    PoolSlots,
    BATCH: tl.constexpr,
    TOKENS: tl.constexpr,
    SLOTS: tl.constexpr,
    R: tl.constexpr,
    RING: tl.constexpr,
    TAIL_BLOCKS: tl.constexpr,
    PAGE: tl.constexpr,
    PAGES: tl.constexpr,
    POOL_COLS: tl.constexpr,
    POS_PLANE: tl.constexpr,
    POS_TOKEN: tl.constexpr,
    THREE: tl.constexpr,
    TAIL_BLOCK: tl.constexpr,
    TAIL_ROW: tl.constexpr,
    TAIL_COL: tl.constexpr,
    POOL_T0: tl.constexpr,
    POOL_T1: tl.constexpr,
    TAIL_T0: tl.constexpr,
):
    slot = tl.program_id(0)
    lo, hi = 0, BATCH
    while lo < hi:
        mid = (lo + hi) // 2
        end_count = tl.load(Ends + mid, mid < BATCH, 2147483647)
        left = slot < end_count
        hi = tl.where(left, mid, hi)
        lo = tl.where(left, lo, mid + 1)
    live = lo < BATCH
    owner = tl.minimum(lo, BATCH - 1)
    start = tl.load(Starts + owner)
    end = tl.load(Starts + owner + 1)
    length = tl.load(Lengths + owner)
    context = tl.maximum(length - (end - start), 0).to(tl.int64)
    prior = tl.load(Ends + owner - 1, live & (owner > 0), 0)
    group = context // R + slot - prior
    first = group * R
    page_index = group // PAGE
    page = tl.load(
        PoolTable + owner * POOL_T0 + page_index * POOL_T1,
        live & (page_index < POOL_COLS),
        -1,
    ).to(tl.int64)
    pool_live = live & (page >= 0) & (page < PAGES)
    tl.store(PoolSlots + slot, tl.where(pool_live, page * PAGE + group % PAGE, -1))

    block = tl.load(TailTable + owner * TAIL_T0, live, -1).to(tl.int64)
    tail_live = live & (block >= 0) & (block < TAIL_BLOCKS)
    member = first + tl.arange(0, R)
    key = start + member - context
    old = member < context
    idx = slot * R + tl.arange(0, R)
    tl.store(
        KeyRows + idx, tl.where(live & ~old & (key >= 0) & (key < TOKENS), key, -1)
    )
    tl.store(
        TailSlots + idx, tl.where(tail_live & old, block * RING + member % RING, -1)
    )

    # Read rotary-position metadata only. Key payload movement is an operator.
    axis = tl.arange(0, 4)
    retained = tl.load(
        Tail + block * TAIL_BLOCK + (first % RING) * TAIL_ROW + (128 + axis) * TAIL_COL,
        tail_live & (first < context) & (axis < 3),
        0,
    ).to(tl.int64)
    source_first = start + first - context
    new_pos = live & (first >= context) & (source_first >= 0) & (source_first < TOKENS)
    if THREE:
        position = tl.load(
            Positions + axis * POS_PLANE + source_first * POS_TOKEN,
            new_pos & (axis < 3),
            0,
        ).to(tl.int64)
    else:
        position = tl.load(Positions + source_first * POS_TOKEN, new_pos, 0).to(
            tl.int64
        )
    tl.store(
        XYZ + axis * SLOTS + slot,
        tl.where(first < context, retained, position),
        axis < 3,
    )


@triton.jit
def _tail_plan(
    Starts,
    Lengths,
    Table,
    KeyRows,
    Slots,
    TOKENS: tl.constexpr,
    RING: tl.constexpr,
    TAIL_BLOCKS: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    start = tl.load(Starts + request)
    end = tl.load(Starts + request + 1)
    length = tl.load(Lengths + request)
    context = length - (end - start)
    token = length - RING + offset
    live = (start >= 0) & (end > start) & (end <= TOKENS) & (context >= 0)
    source = start + token - context
    block = tl.load(Table + request * TABLE_ROW, live, -1).to(tl.int64)
    valid = live & (token >= context) & (block >= 0) & (block < TAIL_BLOCKS)
    item = request * RING + offset
    tl.store(KeyRows + item, tl.where(valid, source, -1), offset < RING)
    tl.store(
        Slots + item, tl.where(valid, block * RING + token % RING, -1), offset < RING
    )


@triton.jit
def _query_metadata(
    Starts,
    Lengths,
    Owner,
    Logical,
    Upper,
    Live,
    BATCH: tl.constexpr,
    TOKENS: tl.constexpr,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), BATCH, tl.int32)
    for _ in range((BATCH + 1).bit_length()):
        mid = (lo + hi) // 2
        end = tl.load(Starts + mid + 1, mid < BATCH, 2147483647)
        left = token < end
        hi = tl.where(left, mid, hi)
        lo = tl.where(left, lo, mid + 1)
    request = tl.minimum(lo, BATCH - 1)
    start = tl.load(Starts + request)
    end = tl.load(Starts + request + 1)
    length = tl.load(Lengths + request)
    context = length - (end - start)
    live = (
        (lo < BATCH) & (start >= 0) & (end > start) & (end <= TOKENS) & (context >= 0)
    )
    logical = tl.maximum(context, 0).to(tl.int64) + token - start
    tl.store(Owner + token, request, token < TOKENS)
    tl.store(Logical + token, logical, token < TOKENS)
    tl.store(Upper + token, tl.where(live, logical // R, 0), token < TOKENS)
    tl.store(Live + token, live, token < TOKENS)


def prepare_indexpool_metadata(
    positions, tail, cache, pool_table, tail_table, starts, lengths, tokens, ratio
):
    """Prepare fixed-capacity mapping tensors; the scheduler owns request order.

    Request starts must be nondecreasing and cache producers own distinct
    live pooled cells and tail ring blocks. Negative copy coordinates are
    absent rows. Shapes are graph-static; lengths and tables stay on device.
    """
    if ratio not in (2, 4, 8, 16):
        raise ValueError("Unsupported IndexPool group ratio")
    batch, device = lengths.numel(), lengths.device
    if (
        tokens < 0
        or (tokens and not batch)
        or starts.shape != (batch + 1,)
        or lengths.shape != (batch,)
        or not starts.is_contiguous()
        or not lengths.is_contiguous()
        or positions.ndim not in (1, 2)
        or (positions.ndim == 2 and positions.shape[0] != 3)
        or positions.shape[-1] < tokens
        or any(
            x.dtype not in (torch.int32, torch.int64)
            for x in (starts, lengths, positions, pool_table, tail_table)
        )
        or any(
            x.device != device
            for x in (starts, positions, tail, cache, pool_table, tail_table)
        )
        or any(
            x.ndim != 2 or x.shape[0] != batch or x.shape[1] == 0
            for x in (pool_table, tail_table)
        )
        or tail.ndim != 3
        or tail.shape[1] == 0
        or tail.shape[2] != 131
        or cache.ndim != 3
        or cache.shape[1] == 0
    ):
        raise ValueError("Invalid IndexPool request metadata or cache shape")

    ring = tail.shape[1]
    slots = (tokens + batch * (ratio - 1)) // ratio

    # During an all-padding capture, no real layer should ask for a plan;
    # still initialize any reserved group/tail entries in the same kernels.
    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)

    prefix = empty(batch, torch.int32)
    coordinates = empty((3, slots), torch.int64)
    group_keys = empty((slots, ratio), torch.int64)
    group_tails = empty((slots, ratio), torch.int64)
    pool_slots = empty(slots, torch.int64)
    tail_keys = empty(batch * ring, torch.int64)
    tail_slots = empty(batch * ring, torch.int64)
    owner = empty(tokens, torch.int64)
    logical = empty(tokens, torch.int64)
    upper = empty(tokens, torch.int32)
    live = empty(tokens, torch.bool)
    if batch:
        _group_prefix[(1,)](
            starts, lengths, prefix, batch, tokens, ratio, triton.next_power_of_2(batch)
        )
        _tail_plan[(batch,)](
            starts,
            lengths,
            tail_table,
            tail_keys,
            tail_slots,
            tokens,
            ring,
            tail.shape[0],
            tail_table.stride(0),
            triton.next_power_of_2(ring),
        )
    if slots:
        _group_plan[(slots,)](
            positions,
            tail,
            pool_table,
            tail_table,
            starts,
            lengths,
            prefix,
            coordinates,
            group_keys,
            group_tails,
            pool_slots,
            batch,
            tokens,
            slots,
            ratio,
            ring,
            tail.shape[0],
            cache.shape[1],
            cache.shape[0],
            pool_table.shape[1],
            positions.stride(0) if positions.ndim == 2 else 0,
            positions.stride(-1),
            positions.ndim == 2,
            *tail.stride(),
            *pool_table.stride(),
            tail_table.stride(0),
        )
    if tokens:
        _query_metadata[(triton.cdiv(tokens, 256),)](
            starts, lengths, owner, logical, upper, live, batch, tokens, ratio, 256
        )
    return IndexPoolPlan(
        coordinates if positions.ndim == 2 else coordinates[0],
        group_keys,
        group_tails,
        pool_slots,
        tail_keys,
        tail_slots,
        owner,
        logical,
        upper,
        live,
    )


@triton.jit
def _flash_expand(
    G,
    P,
    A,
    Output,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    BUDGET: tl.constexpr,
    R: tl.constexpr,
    ACTIVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    pos = tl.load(P + row)
    upper = pos // R
    live = pos >= 0
    if ACTIVE:
        live = live & tl.load(A + row)
    group_index = j // R
    group = tl.load(G + row * GS0 + group_index * GS1, group_index < BUDGET, other=-1)
    history = j < BUDGET * R
    value = tl.where(history, group * R + j % R, upper * R + j % R)
    valid = tl.where(history, (group >= 0) & (group < upper), value <= pos)
    valid = valid & live & (j < BUDGET * R + R)
    total: tl.constexpr = BUDGET * R + R
    in_range = j < total
    nv = tl.sum(valid.to(tl.int32), 0)
    good = tl.cumsum(valid.to(tl.int32), 0) - 1
    bad = nv + tl.cumsum((in_range & ~valid).to(tl.int32), 0) - 1
    dst = tl.where(valid, good, bad)
    tl.store(Output + row * total + dst, tl.where(valid, value, -1), in_range)


def expand_groups(group_ids, positions, ratio, active=None):
    if (
        group_ids.ndim != 2
        or group_ids.dtype not in (torch.int32, torch.int64)
        or ratio not in (2, 4, 8, 16)
    ):
        raise ValueError("Invalid Flash group expansion inputs")
    if group_ids.device.type != "cuda":
        raise NotImplementedError("Flash group expansion requires a CUDA device")
    n, budget = group_ids.shape
    if (
        positions.dtype not in (torch.int32, torch.int64)
        or positions.shape != (n,)
        or positions.device != group_ids.device
        or not positions.is_contiguous()
    ):
        raise ValueError("Invalid group positions")
    if active is not None and (
        active.shape != (n,)
        or active.dtype != torch.bool
        or active.device != group_ids.device
        or not active.is_contiguous()
    ):
        raise ValueError("Invalid group activation mask")
    out = torch.empty(
        (n, budget * ratio + ratio), dtype=torch.int32, device=group_ids.device
    )
    if n:
        _flash_expand[(n,)](
            group_ids,
            positions,
            active,
            out,
            *group_ids.stride(),
            budget,
            ratio,
            active is not None,
            triton.next_power_of_2(out.shape[1]),
            num_warps=4,
        )
    return out
