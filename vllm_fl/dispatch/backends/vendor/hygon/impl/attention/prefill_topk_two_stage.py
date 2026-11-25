# Copyright (c) 2026 BAAI. All rights reserved.
"""Exact Top512 using a bounded intermediate candidate set.

Reduce a wide row to at most 1024 candidates before resolving the exact cutoff.
No sampling or approximate threshold is used. Cutoff ties may select any valid
subset, consistent with the existing prefill selector. Invalid ranges are never
read, even when their storage contains stale values or NaNs.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "XS", "XC"])
def _compact(X, S, E, V, I, C, N, XS, XC,
             B: tl.constexpr, CAP: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    start = tl.minimum(N, tl.maximum(0, tl.load(S + row)))
    end = tl.minimum(N, tl.maximum(start, tl.load(E + row)))
    valid = (col < N) & (col >= start) & (col < end)
    value = tl.load(X + row * XS + col * XC, valid, other=float('-inf'))
    valid = valid & (value != float('-inf'))
    value = tl.where(value == 0, 0., value)  # Treat signed zeros as ties.
    bits = value.to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0,
                   ~bits, bits ^ 0x80000000).to(tl.uint32)
    count = tl.sum(valid.to(tl.int32), 0)
    want = tl.minimum(count, 512)
    prefix = tl.full((), 0, tl.uint32)
    shift = 31
    remaining = count
    # Maintain at least `want` candidates; stop once they fit the workspace.
    while (remaining > CAP) & (shift >= 0):
        trial = prefix | (tl.full((), 1, tl.uint32) << shift)
        num = tl.sum(((key >= trial) & valid).to(tl.int32), 0)
        accept = num >= want
        prefix = tl.where(accept, trial, prefix)
        remaining = tl.where(accept, num, remaining)
        shift -= 1
    selected = valid & (key >= prefix)
    if remaining > CAP:
        # The exact cutoff has a large tie. Keep all higher values, then only
        # the required ties, so no write can overflow the candidate buffer.
        higher = (key > prefix) & valid
        equal = (key == prefix) & valid
        num_higher = tl.sum(higher.to(tl.int32), 0)
        tie_rank = tl.cumsum(equal.to(tl.int32), 0)
        selected = higher | (equal & (tie_rank <= want - num_higher))
        remaining = want
    position = tl.cumsum(selected.to(tl.int32), 0) - 1
    write = selected & (position < CAP)
    tl.store(V + row * CAP + position, value, write)
    tl.store(I + row * CAP + position, col - start, write)
    tl.store(C + row, remaining)


@triton.jit(do_not_specialize=["OS", "OC"])
def _finish(V, I, C, O, OS, OC, CAP: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, CAP)
    count = tl.load(C + row)
    valid = col < count
    value = tl.load(V + row * CAP + col, valid, other=float('-inf'))
    bits = value.to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0,
                   ~bits, bits ^ 0x80000000).to(tl.uint32)
    selected = valid
    want = tl.minimum(count, 512)
    if count > 512:
        prefix = tl.full((), 0, tl.uint32)
        shift = 31
        done = False
        while (shift >= 0) & (~done):
            trial = prefix | (tl.full((), 1, tl.uint32) << shift)
            num = tl.sum(((key >= trial) & valid).to(tl.int32), 0)
            prefix = tl.where(num >= want, trial, prefix)
            done = num == want
            shift -= 1
        higher = (key > prefix) & valid
        equal = (key == prefix) & valid
        num_higher = tl.sum(higher.to(tl.int32), 0)
        tie_rank = tl.cumsum(equal.to(tl.int32), 0)
        selected = higher | (equal & (tie_rank <= want - num_higher))
    position = tl.cumsum(selected.to(tl.int32), 0) - 1
    index = tl.load(I + row * CAP + col, valid, other=-1)
    tl.store(O + row * OS + position * OC, index,
             selected & (position < 512))
    tl.store(O + row * OS + col * OC, -1, (col < 512) & (col >= want))


def prefill_topk_two_stage(logits, starts, ends, output):
    """Caller validates FP32 rows, int32 output, Top512 and measured widths."""
    rows, width = logits.shape
    capacity = 1024
    values = torch.empty((rows, capacity), device=logits.device,
                         dtype=torch.float32)
    indices = torch.empty((rows, capacity), device=logits.device,
                          dtype=torch.int32)
    counts = torch.empty(rows, device=logits.device, dtype=torch.int32)
    _compact[(rows,)](
        logits, starts, ends, values, indices, counts, width,
        *logits.stride(), triton.next_power_of_2(width), capacity, num_warps=8,
    )
    _finish[(rows,)](
        values, indices, counts, output, *output.stride(), capacity, num_warps=4,
    )
