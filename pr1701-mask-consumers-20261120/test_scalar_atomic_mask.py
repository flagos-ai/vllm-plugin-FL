"""Regression contracts for scalar masked atomics.

Every address is valid even when the mask is false, so the control compiler can
expose incorrect writes without relying on an out-of-bounds device exception.
The original vLLM reproduction remains unchanged and is a separate requirement.
"""

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _scalar_atomic_mask_contract(
    ids_ptr,
    bounds_ptr,
    counts_ptr,
    old_values_ptr,
    unrelated_ptr,
    BLOCK: tl.constexpr,
    SENTINEL: tl.constexpr,
):
    position = tl.arange(0, BLOCK)
    lower = tl.load(bounds_ptr)
    upper = tl.load(bounds_ptr + 1)
    ids = tl.load(ids_ptr + position)
    mask = (position < upper) & (position >= lower)
    old = tl.atomic_add(counts_ptr + ids, 1, mask=mask, sem="acq_rel")
    tl.store(old_values_ptr + position, tl.where(mask, old, SENTINEL))
    # A repair must not move this independent side effect under the RMW mask.
    tl.store(unrelated_ptr + position, position + 23)


@pytest.mark.parametrize(
    "lower,upper",
    [(0, 0), (0, 1024), (72, 175), (135, 185), (40, 41), (1024, 1024)],
)
def test_scalar_compound_mask_old_values_and_unrelated_store(lower, upper):
    block, bins, initial, sentinel = 1024, 17, 9, -777
    position = torch.arange(block, dtype=torch.int32)
    ids = (position * 13 + 7) % bins
    bounds = torch.tensor([lower, upper], dtype=torch.int32)
    counts = torch.full((bins, ), initial, dtype=torch.int32, device="cuda")
    old = torch.full((block, ), sentinel, dtype=torch.int32, device="cuda")
    unrelated = torch.full((block, ), -1, dtype=torch.int32, device="cuda")
    _scalar_atomic_mask_contract[(1, )](
        ids.to("cuda"),
        bounds.to("cuda"),
        counts,
        old,
        unrelated,
        BLOCK=block,
        SENTINEL=sentinel,
    )
    torch.cuda.synchronize()
    active = (position < upper) & (position >= lower)
    expected = torch.bincount(ids[active].long(), minlength=bins).to(torch.int32)
    torch.testing.assert_close(counts.cpu(), expected + initial, rtol=0, atol=0)
    torch.testing.assert_close(unrelated.cpu(), position + 23, rtol=0, atol=0)
    old = old.cpu()
    torch.testing.assert_close(old[~active], torch.full_like(old[~active], sentinel), rtol=0, atol=0)
    # Atomic execution order is unspecified. For repeated IDs, the multiset of
    # returned old values must still be the complete consecutive count range.
    for token in range(bins):
        observed = old[active & (ids == token)].sort().values
        wanted = torch.arange(initial, initial + int(expected[token]), dtype=torch.int32)
        torch.testing.assert_close(observed, wanted, rtol=0, atol=0)


@triton.jit
def _defined_other_and_atomic(input_ptr, length_ptr, copied_ptr, counter_ptr, BLOCK: tl.constexpr):
    position = tl.arange(0, BLOCK)
    length = tl.load(length_ptr)
    values = tl.load(input_ptr + position, mask=position < length, other=23)
    tl.store(copied_ptr + position, values)
    # Activate the scalar simulator while retaining a load with defined other.
    tl.atomic_add(counter_ptr + position, 1, mask=position < length)


def test_defined_load_other_is_preserved():
    block, length = 1024, 103
    values = torch.arange(block, dtype=torch.int32)
    copied = torch.empty((block, ), dtype=torch.int32, device="cuda")
    counts = torch.full((block, ), 9, dtype=torch.int32, device="cuda")
    _defined_other_and_atomic[(1, )](
        values.to("cuda"),
        torch.tensor([length], dtype=torch.int32, device="cuda"),
        copied,
        counts,
        BLOCK=block,
    )
    torch.cuda.synchronize()
    active = values < length
    expected_copy = torch.where(active, values, torch.full_like(values, 23))
    torch.testing.assert_close(copied.cpu(), expected_copy, rtol=0, atol=0)
    torch.testing.assert_close(counts.cpu(), 9 + active.int(), rtol=0, atol=0)


@triton.jit
def _non_atomic_masked_copy(input_ptr, length_ptr, output_ptr, BLOCK: tl.constexpr):
    position = tl.arange(0, BLOCK)
    length = tl.load(length_ptr)
    values = tl.load(input_ptr + position, mask=position < length, other=0)
    tl.store(output_ptr + position, values)


def test_non_atomic_vector_copy_is_unchanged():
    block, length = 1024, 1001
    values = torch.arange(block, dtype=torch.int32)
    output = torch.empty((block, ), dtype=torch.int32, device="cuda")
    _non_atomic_masked_copy[(1, )](
        values.to("cuda"),
        torch.tensor([length], dtype=torch.int32, device="cuda"),
        output,
        BLOCK=block,
    )
    torch.cuda.synchronize()
    expected = torch.where(values < length, values, torch.zeros_like(values))
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
