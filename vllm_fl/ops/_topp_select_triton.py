"""FP64 weighted-boundary Triton kernels for top-p selection."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_BLOCK = 1024
_SEARCH_STEPS = 16


@triton.jit
def _float_to_ordered_u32(x):
    signed = x.to(tl.int32, bitcast=True)
    unsigned = x.to(tl.uint32, bitcast=True)
    sign_bit = tl.full((), 0x80000000, tl.uint32)
    sign = (signed >> 31).to(tl.uint32, bitcast=True)
    return unsigned ^ (sign_bit | sign)


@triton.jit
def _ordered_u32_to_float(key):
    positive = (key & 0x80000000) != 0
    bits = tl.where(positive, key ^ 0x80000000, ~key).to(tl.uint32)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _init_partials(
    X, PART_MIN, PART_MAX, PART_BAD, V: tl.constexpr, BLOCK: tl.constexpr
):
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    valid = cols < V
    x = tl.load(X + row * V + cols, valid, other=0.0)
    invalid = valid & ((x != x) | (x == float("inf")))
    finite = valid & (x > -float("inf")) & (x < float("inf"))
    keys = _float_to_ordered_u32(x)
    low = tl.min(tl.where(finite, keys, 0xFFFFFFFF).to(tl.uint32), axis=0)
    high = tl.max(tl.where(finite, keys, 0).to(tl.uint32), axis=0)
    bad_count = tl.sum(invalid.to(tl.int32), axis=0)
    offset = chunk * tl.num_programs(1) + row
    tl.store(PART_MIN + offset, low)
    tl.store(PART_MAX + offset, high)
    tl.store(PART_BAD + offset, bad_count)


@triton.jit
def _init_rows(
    PART_MIN,
    PART_MAX,
    PART_BAD,
    P,
    ROW_LO,
    ROW_HI,
    ROW_BAD,
    ROW_MAX,
    NCHUNK: tl.constexpr,
    NCHUNK_POW2: tl.constexpr,
):
    row = tl.program_id(0)
    chunks = tl.arange(0, NCHUNK_POW2)
    offsets = chunks * tl.num_programs(0) + row
    part_min = tl.load(PART_MIN + offsets, chunks < NCHUNK, other=0xFFFFFFFF)
    part_max = tl.load(PART_MAX + offsets, chunks < NCHUNK, other=0)
    part_bad = tl.load(PART_BAD + offsets, chunks < NCHUNK, other=0)
    low = tl.min(part_min, axis=0)
    high = tl.max(part_max, axis=0)
    invalid = (tl.sum(part_bad, axis=0) != 0) | (low == 0xFFFFFFFF)
    p = tl.load(P + row)
    invalid = invalid | (p != p)
    if invalid:
        zero_key = tl.full((), 0, tl.uint32)
        low = zero_key
        high = zero_key
        row_max = tl.full((), 0.0, tl.float64)
    else:
        row_max = _ordered_u32_to_float(high).to(tl.float64)
        if (p <= 0.0) | (p > 1.0):
            low = high
    tl.store(ROW_LO + row, low)
    tl.store(ROW_HI + row, high)
    tl.store(ROW_BAD + row, invalid.to(tl.int32))
    tl.store(ROW_MAX + row, row_max)


@triton.jit
def _weight_partials(
    X,
    ROW_MAX,
    ROW_BAD,
    WEIGHTS,
    PART_Z,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    valid = cols < V
    zero64 = tl.full((), 0.0, tl.float64)
    if tl.load(ROW_BAD + row) != 0:
        weights = tl.full((BLOCK,), 0.0, tl.float64)
    else:
        row_max = tl.load(ROW_MAX + row)
        x = tl.load(X + row * V + cols, valid, other=0.0)
        safe_x = tl.where(valid, x, row_max.to(tl.float32))
        delta = safe_x.to(tl.float64) - row_max
        weights = tl.where(valid, libdevice.exp(delta), zero64)
    tl.store(WEIGHTS + row * V + cols, weights, valid)
    tl.store(PART_Z + chunk * tl.num_programs(1) + row, tl.sum(weights, axis=0))


@triton.jit
def _finish_z(
    P, ROW_BAD, PART_Z, ROW_ALPHA, NCHUNK: tl.constexpr, NCHUNK_POW2: tl.constexpr
):
    row = tl.program_id(0)
    chunks = tl.arange(0, NCHUNK_POW2)
    offsets = chunks * tl.num_programs(0) + row
    zero64 = tl.full((), 0.0, tl.float64)
    one64 = tl.full((), 1.0, tl.float64)
    partials = tl.load(PART_Z + offsets, chunks < NCHUNK, other=zero64)
    z = tl.sum(partials, axis=0)
    p = tl.load(P + row).to(tl.float64)
    if tl.load(ROW_BAD + row) != 0:  # noqa: SIM108 - Preserve target-compiler control flow.
        alpha = zero64
    else:
        alpha = (one64 - p) * z
    tl.store(ROW_ALPHA + row, alpha)


@triton.jit
def _search_partial(
    X,
    WEIGHTS,
    ROW_LO,
    ROW_HI,
    ROW_BAD,
    PART_MASS,
    PART_MIN,
    PART_MAX,
    NCHUNK: tl.constexpr,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    low = tl.load(ROW_LO + row).to(tl.uint32)
    high = tl.load(ROW_HI + row).to(tl.uint32)
    invalid = tl.load(ROW_BAD + row)
    batch = tl.num_programs(1)
    offset = chunk * batch + row
    plane_stride = NCHUNK * batch
    zero64 = tl.full((), 0.0, tl.float64)
    if (invalid == 0) & (low < high):
        mid = low + ((high - low) >> 1)
        cut_lo = low + ((mid - low) >> 1)
        mid_plus_one = mid + tl.full((), 1, tl.uint32)
        cut_hi = mid_plus_one + ((high - mid_plus_one) >> 1)
        cols = chunk * BLOCK + tl.arange(0, BLOCK)
        valid = cols < V
        x = tl.load(X + row * V + cols, valid, other=0.0)
        weights = tl.load(WEIGHTS + row * V + cols, valid, other=zero64)
        keys = _float_to_ordered_u32(x)
        mass_lo = tl.sum(tl.where(valid & (keys <= cut_lo), weights, zero64), axis=0)
        mass_mid = tl.sum(tl.where(valid & (keys <= mid), weights, zero64), axis=0)
        mass_hi = tl.sum(tl.where(valid & (keys <= cut_hi), weights, zero64), axis=0)
        finite = valid & (x > -float("inf")) & (x < float("inf"))
        min_above = tl.min(
            tl.where(finite & (keys > mid), keys, 0xFFFFFFFF).to(tl.uint32), axis=0
        )
        max_below = tl.max(
            tl.where(finite & (keys <= mid), keys, 0).to(tl.uint32), axis=0
        )
    else:
        mass_lo = zero64
        mass_mid = zero64
        mass_hi = zero64
        min_above = tl.full((), 0xFFFFFFFF, tl.uint32)
        max_below = tl.full((), 0, tl.uint32)
    tl.store(PART_MASS + offset, mass_lo)
    tl.store(PART_MASS + plane_stride + offset, mass_mid)
    tl.store(PART_MASS + 2 * plane_stride + offset, mass_hi)
    tl.store(PART_MIN + offset, min_above)
    tl.store(PART_MAX + offset, max_below)


@triton.jit
def _search_update(
    ROW_ALPHA,
    ROW_LO,
    ROW_HI,
    ROW_BAD,
    PART_MASS,
    PART_MIN,
    PART_MAX,
    NCHUNK: tl.constexpr,
    NCHUNK_POW2: tl.constexpr,
):
    row = tl.program_id(0)
    low = tl.load(ROW_LO + row).to(tl.uint32)
    high = tl.load(ROW_HI + row).to(tl.uint32)
    invalid = tl.load(ROW_BAD + row)
    if (invalid == 0) & (low < high):
        chunks = tl.arange(0, NCHUNK_POW2)
        offsets = chunks * tl.num_programs(0) + row
        zero64 = tl.full((), 0.0, tl.float64)
        plane_stride = NCHUNK * tl.num_programs(0)
        masses_lo = tl.load(PART_MASS + offsets, chunks < NCHUNK, other=zero64)
        masses_mid = tl.load(
            PART_MASS + plane_stride + offsets, chunks < NCHUNK, other=zero64
        )
        masses_hi = tl.load(
            PART_MASS + 2 * plane_stride + offsets, chunks < NCHUNK, other=zero64
        )
        min_above_parts = tl.load(PART_MIN + offsets, chunks < NCHUNK, other=0xFFFFFFFF)
        max_below_parts = tl.load(PART_MAX + offsets, chunks < NCHUNK, other=0)
        mass_lo = tl.sum(masses_lo, axis=0)
        mass_mid = tl.sum(masses_mid, axis=0)
        mass_hi = tl.sum(masses_hi, axis=0)
        min_above = tl.min(min_above_parts, axis=0)
        max_below = tl.max(max_below_parts, axis=0)
        mid = low + ((high - low) >> 1)
        cut_lo = low + ((mid - low) >> 1)
        mid_plus_one = mid + tl.full((), 1, tl.uint32)
        cut_hi = mid_plus_one + ((high - mid_plus_one) >> 1)
        alpha = tl.load(ROW_ALPHA + row)
        if mass_mid <= alpha:
            low = min_above
            selected_cut = cut_hi
            selected_mass = mass_hi
        else:
            high = max_below
            selected_cut = cut_lo
            selected_mass = mass_lo
        if (low <= selected_cut) & (selected_cut < high):
            if selected_mass <= alpha:
                low = selected_cut + tl.full((), 1, tl.uint32)
            else:
                high = selected_cut
        tl.store(ROW_LO + row, low)
        tl.store(ROW_HI + row, high)


@triton.jit
def _boundary_stats(
    X,
    WEIGHTS,
    ROW_LO,
    ROW_BAD,
    PART_LESS,
    PART_EQ_COUNT,
    PART_EQ_WEIGHT,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    offset = chunk * tl.num_programs(1) + row
    zero64 = tl.full((), 0.0, tl.float64)
    if tl.load(ROW_BAD + row) != 0:
        tl.store(PART_LESS + offset, zero64)
        tl.store(PART_EQ_COUNT + offset, 0)
        tl.store(PART_EQ_WEIGHT + offset, zero64)
    else:
        boundary = tl.load(ROW_LO + row).to(tl.uint32)
        cols = chunk * BLOCK + tl.arange(0, BLOCK)
        valid = cols < V
        x = tl.load(X + row * V + cols, valid, other=0.0)
        weights = tl.load(WEIGHTS + row * V + cols, valid, other=zero64)
        keys = _float_to_ordered_u32(x)
        less = valid & (keys < boundary)
        equal = valid & (keys == boundary)
        mass_less = tl.sum(tl.where(less, weights, zero64), axis=0)
        equal_count = tl.sum(equal.to(tl.int32), axis=0)
        equal_weight = tl.sum(tl.where(equal, weights, zero64), axis=0)
        tl.store(PART_LESS + offset, mass_less)
        tl.store(PART_EQ_COUNT + offset, equal_count)
        tl.store(PART_EQ_WEIGHT + offset, equal_weight)


@triton.jit
def _finish_rows(
    P,
    ROW_ALPHA,
    ROW_BAD,
    PART_LESS,
    PART_EQ_COUNT,
    PART_EQ_WEIGHT,
    N_REMOVE,
    EQ_PREFIX,
    NCHUNK: tl.constexpr,
    NCHUNK_POW2: tl.constexpr,
):
    row = tl.program_id(0)
    chunks = tl.arange(0, NCHUNK_POW2)
    offsets = chunks * tl.num_programs(0) + row
    zero64 = tl.full((), 0.0, tl.float64)
    one64 = tl.full((), 1.0, tl.float64)
    mass_less_parts = tl.load(PART_LESS + offsets, chunks < NCHUNK, other=zero64)
    eq_count_parts = tl.load(PART_EQ_COUNT + offsets, chunks < NCHUNK, other=0)
    eq_weight_parts = tl.load(PART_EQ_WEIGHT + offsets, chunks < NCHUNK, other=zero64)
    mass_less = tl.sum(mass_less_parts, axis=0)
    equal_count = tl.sum(eq_count_parts, axis=0)
    equal_weight = tl.sum(eq_weight_parts, axis=0)
    p = tl.load(P + row)
    invalid = tl.load(ROW_BAD + row) != 0
    if invalid | (p > 1.0):
        n_remove = 0
    elif p <= 0.0:
        n_remove = tl.maximum(equal_count - 1, 0)
    else:
        denom = tl.where(equal_weight > zero64, equal_weight, one64)
        numerator = tl.maximum(tl.load(ROW_ALPHA + row) - mass_less, zero64)
        raw = (numerator * equal_count.to(tl.float64) / denom).to(tl.int32)
        n_remove = tl.minimum(raw, equal_count)
        n_remove = tl.minimum(n_remove, tl.maximum(equal_count - 1, 0))
    tl.store(N_REMOVE + row, n_remove)
    prefix = tl.cumsum(eq_count_parts, axis=0) - eq_count_parts
    tl.store(EQ_PREFIX + offsets, prefix, chunks < NCHUNK)


@triton.jit
def _apply_mask(
    X,
    Y,
    P,
    ROW_LO,
    ROW_BAD,
    N_REMOVE,
    EQ_PREFIX,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    valid = cols < V
    x = tl.load(X + row * V + cols, valid, other=0.0)
    p = tl.load(P + row)
    invalid = tl.load(ROW_BAD + row) != 0
    if invalid | (p > 1.0):
        out = x
    else:
        boundary = tl.load(ROW_LO + row).to(tl.uint32)
        n_remove = tl.load(N_REMOVE + row)
        prefix = tl.load(EQ_PREFIX + chunk * tl.num_programs(1) + row)
        keys = _float_to_ordered_u32(x)
        equal = valid & (keys == boundary)
        rank = prefix + tl.cumsum(equal.to(tl.int32), axis=0)
        remove = valid & ((keys < boundary) | (equal & (rank <= n_remove)))
        out = tl.where(remove, -float("inf"), x)
    tl.store(Y + row * V + cols, out, valid)


def apply(
    logits: torch.Tensor, p: torch.Tensor, *, block: int = _BLOCK
) -> torch.Tensor:
    """Return an out-of-place FP32 top-p mask using FP64 unnormalized weights."""
    batch, vocab = logits.shape
    chunks = triton.cdiv(vocab, block)
    chunks_pow2 = triton.next_power_of_2(chunks)
    partial_shape = (chunks, batch)
    part_min = torch.empty(partial_shape, dtype=torch.uint32, device=logits.device)
    part_max = torch.empty_like(part_min)
    part_bad = torch.empty(partial_shape, dtype=torch.int32, device=logits.device)
    row_lo = torch.empty((batch,), dtype=torch.uint32, device=logits.device)
    row_hi = torch.empty_like(row_lo)
    row_bad = torch.empty((batch,), dtype=torch.int32, device=logits.device)
    row_max = torch.empty((batch,), dtype=torch.float64, device=logits.device)
    weights = torch.empty(logits.shape, dtype=torch.float64, device=logits.device)
    part_z = torch.empty(partial_shape, dtype=torch.float64, device=logits.device)
    row_alpha = torch.empty((batch,), dtype=torch.float64, device=logits.device)
    part_mass = torch.empty(
        (3, chunks, batch), dtype=torch.float64, device=logits.device
    )
    part_less = torch.empty_like(part_z)
    part_eq_count = torch.empty(partial_shape, dtype=torch.int32, device=logits.device)
    part_eq_weight = torch.empty_like(part_z)
    n_remove = torch.empty((batch,), dtype=torch.int32, device=logits.device)
    eq_prefix = torch.empty(partial_shape, dtype=torch.int32, device=logits.device)
    output = torch.empty_like(logits)

    _init_partials[(chunks, batch)](
        logits, part_min, part_max, part_bad, vocab, block, num_warps=4
    )
    _init_rows[(batch,)](
        part_min,
        part_max,
        part_bad,
        p,
        row_lo,
        row_hi,
        row_bad,
        row_max,
        chunks,
        chunks_pow2,
        num_warps=4,
    )
    _weight_partials[(chunks, batch)](
        logits, row_max, row_bad, weights, part_z, vocab, block, num_warps=4
    )
    _finish_z[(batch,)](p, row_bad, part_z, row_alpha, chunks, chunks_pow2, num_warps=4)
    for _ in range(_SEARCH_STEPS):
        _search_partial[(chunks, batch)](
            logits,
            weights,
            row_lo,
            row_hi,
            row_bad,
            part_mass,
            part_min,
            part_max,
            chunks,
            vocab,
            block,
            num_warps=4,
        )
        _search_update[(batch,)](
            row_alpha,
            row_lo,
            row_hi,
            row_bad,
            part_mass,
            part_min,
            part_max,
            chunks,
            chunks_pow2,
            num_warps=4,
        )
    _boundary_stats[(chunks, batch)](
        logits,
        weights,
        row_lo,
        row_bad,
        part_less,
        part_eq_count,
        part_eq_weight,
        vocab,
        block,
        num_warps=4,
    )
    _finish_rows[(batch,)](
        p,
        row_alpha,
        row_bad,
        part_less,
        part_eq_count,
        part_eq_weight,
        n_remove,
        eq_prefix,
        chunks,
        chunks_pow2,
        num_warps=4,
    )
    _apply_mask[(chunks, batch)](
        logits,
        output,
        p,
        row_lo,
        row_bad,
        n_remove,
        eq_prefix,
        vocab,
        block,
        num_warps=4,
    )
    return output
