"""Triton kernel loaded only when a non-empty GCU vocabulary mask runs."""

from vllm.triton_utils import tl, triton


@triton.jit
def _mask_kernel(
    INPUT,
    OUTPUT,
    MASK,
    COUNT: tl.constexpr,
    START: tl.constexpr,
    END: tl.constexpr,
    PADDING: tl.constexpr,
    ADDED_START: tl.constexpr,
    ADDED_END: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(INPUT + offsets, offsets < COUNT, other=0).to(tl.int64)
    original = (values >= START) & (values < END)
    added = (values >= ADDED_START) & (values < ADDED_END)
    added_offset = ADDED_START - (END - START) - PADDING
    offset = tl.where(original, START, 0) + tl.where(added, added_offset, 0)
    valid = original | added
    tl.store(OUTPUT + offsets, tl.where(valid, values - offset, 0), offsets < COUNT)
    tl.store(MASK + offsets, ~valid, offsets < COUNT)
