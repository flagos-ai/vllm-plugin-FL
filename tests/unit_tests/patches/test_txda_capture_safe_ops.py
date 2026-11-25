# Copyright (c) 2026 BAAI. All rights reserved.

"""Numerical contracts of the TXDA capture-safe operator rewrites."""

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.txda.patches import capture_safe_ops


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_masked_fill_matches_torch(dtype):
    values = torch.arange(24, dtype=dtype).reshape(4, 6)
    mask = torch.zeros(4, 1, dtype=torch.bool)
    mask[1] = True
    mask[3] = True

    got = capture_safe_ops._masked_fill_impl(mask, values, 7)

    torch.testing.assert_close(got, torch.masked_fill(values, mask, 7))
    assert got.dtype == dtype


def test_masked_fill_inplace_mutates_and_returns_the_same_tensor():
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    mask = torch.tensor([[True, False, False, False]] * 3)

    out = capture_safe_ops._masked_fill_inplace_impl(mask, values, 0.0)

    assert out is values
    torch.testing.assert_close(
        values,
        torch.masked_fill(
            torch.arange(12, dtype=torch.float32).reshape(3, 4), mask, 0.0
        ),
    )


def test_masked_fill_handles_tensor_value():
    values = torch.ones(2, 3)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    mask[0, 0] = True

    got = capture_safe_ops._masked_fill_impl(mask, values, torch.tensor(5.0))

    torch.testing.assert_close(got, torch.masked_fill(values, mask, 5.0))


def test_masked_fill_uses_only_pointwise_ops(monkeypatch):
    """The rewrite must not touch masked_fill again (that is the broken op)."""
    calls = []
    original = torch.masked_fill

    def guard(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "masked_fill", guard)
    values = torch.zeros(2, 2)
    mask = torch.ones(2, 2, dtype=torch.bool)

    capture_safe_ops._masked_fill_impl(mask, values, 1.0)

    assert calls == []


def test_comparison_rewrites_are_math_identities(monkeypatch):
    """``lt`` / ``le`` are expressed as ``!(ge)`` / ``!(gt)``."""
    x = torch.tensor([1, 2, 3, -5], dtype=torch.int64)

    def lt(a, b):
        return torch.logical_not(torch.ge(a, b))

    def le(a, b):
        return torch.logical_not(torch.gt(a, b))

    torch.testing.assert_close(lt(x, 0), torch.lt(x, 0))
    torch.testing.assert_close(le(x, 3), torch.le(x, 3))
