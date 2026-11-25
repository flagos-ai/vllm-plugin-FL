# Copyright (c) 2026 BAAI. All rights reserved.
"""Exact prefill TopK with range masking, relative indices and -1 padding.

LightOp's K512 implementation fails fixed-input regression. Never fall back to
it here. Prefill auto can opt into exact two-stage selection on measured widths.
The wide-row fallback uses native ATen TopK's out overload so FlagGems' functional
override cannot change its stride or compilation behavior. Ties may select any
valid subset.
"""
import os

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "XS"])
def _radix(X, S, E, O, N, XS, XC: tl.constexpr,
           OS: tl.constexpr, OC: tl.constexpr, TOP: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    s = tl.minimum(N, tl.maximum(0, tl.load(S + row)))
    e = tl.minimum(N, tl.maximum(s, tl.load(E + row)))
    valid = (i < N) & (i >= s) & (i < e)
    x = tl.load(X + row * XS + i * XC, valid, other=float('-inf')).to(tl.float32)
    valid = valid & (x != float('-inf'))
    x = tl.where(x == 0, 0., x)
    bits = x.to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
    count = tl.sum(valid.to(tl.int32), 0)
    want = tl.minimum(TOP, count)
    selected = valid
    if count > TOP:
        prefix = tl.full((), 0, tl.uint32)
        shift = 31
        done = False
        while (shift >= 0) & (~done):
            trial = prefix | (tl.full((), 1, tl.uint32) << shift)
            num = tl.sum(((key >= trial) & valid).to(tl.int32), 0)
            prefix = tl.where(num >= want, trial, prefix)
            done = num == want
            shift -= 1
        high = (key > prefix) & valid
        eq = (key == prefix) & valid
        nhigh = tl.sum(high.to(tl.int32), 0)
        eqrank = tl.cumsum(eq.to(tl.int32), 0)
        selected = high | (eq & (eqrank <= want - nhigh))
    position = tl.cumsum(selected.to(tl.int32), 0) - 1
    tl.store(O + row * OS + position * OC, i - s, selected & (position < TOP))
    tl.store(O + row * OS + i * OC, -1, (i < TOP) & (i >= want))


@triton.jit(do_not_specialize=["M", "N", "XS"])
def _mask(X, S, E, Y, M, N, XS, XC: tl.constexpr, B: tl.constexpr):
    z = tl.program_id(0) * B + tl.arange(0, B)
    row, col = z // N, z % N
    s = tl.minimum(N, tl.maximum(0, tl.load(S + row, row < M, other=0)))
    e = tl.minimum(N, tl.maximum(s, tl.load(E + row, row < M, other=0)))
    x = tl.load(X + row * XS + col * XC,
                (row < M) & (col >= s) & (col < e), other=float('-inf'))
    tl.store(Y + z, x, row < M)


@triton.jit(do_not_specialize=["M", "N", "TAKE"])
def _convert(V, I, S, O, M, N, TAKE,
             TOP: tl.constexpr, OS: tl.constexpr, OC: tl.constexpr, B: tl.constexpr):
    z = tl.program_id(0) * B + tl.arange(0, B)
    row, col = z // TOP, z % TOP
    valid = (row < M) & (col < TAKE)
    v = tl.load(V + row * TAKE + col, valid, other=float('-inf'))
    i = tl.load(I + row * TAKE + col, valid, other=0)
    s = tl.minimum(N, tl.maximum(0, tl.load(S + row, row < M, other=0)))
    tl.store(O + row * OS + col * OC, tl.where(v == float('-inf'), -1, i - s), row < M)


def prefill_topk(logits, starts, ends, output, *, already_masked=False,
                 reference=False, optimize_wide=False):
    """Write TopK indices; already_masked requires -inf outside valid ranges."""
    m, n = logits.shape
    top = output.shape[1]
    if m == 0 or top == 0:
        return
    if n == 0:
        output.fill_(-1)
        return
    starts = starts.to(device=logits.device, dtype=torch.int32).reshape(-1)[:m].contiguous()
    ends = ends.to(device=logits.device, dtype=torch.int32).reshape(-1)[:m].contiguous()
    if n <= 4096 and top == 512 and not reference:
        _radix[(m,)](logits, starts, ends, output, n, *logits.stride(),
                     *output.stride(), top, triton.next_power_of_2(max(n, top)), num_warps=4)
        return
    if (optimize_wide and not reference and top == 512 and m >= 128
            and 4096 < n <= 16384 and logits.dtype == torch.float32
            and output.dtype == torch.int32
            and os.getenv("VLLM_FL_HYGON_PREFILL_TOPK_TWO_STAGE", "1").lower()
            not in ("0", "false", "off", "no")):
        from .prefill_topk_two_stage import prefill_topk_two_stage

        prefill_topk_two_stage(logits, starts, ends, output)
        return
    if not already_masked:
        masked = torch.empty((m, n), device=logits.device, dtype=logits.dtype)
        _mask[(triton.cdiv(m * n, 1024),)](
            logits, starts, ends, masked, m, n, *logits.stride(), 1024)
        logits = masked
    take = min(n, top)
    values = torch.empty((m, take), device=logits.device, dtype=logits.dtype)
    indices = torch.empty((m, take), device=logits.device, dtype=torch.int64)
    # FlagGems overrides aten::topk, but not aten::topk.values. Calling the
    # functional overload here silently selects its shape-specialized kernels.
    torch.ops.aten.topk.values(
        logits, take, -1, True, False, values=values, indices=indices)
    _convert[(triton.cdiv(m * top, 1024),)](
        values, indices, starts, output, m, n, take, top, *output.stride(), 1024)
