# Copyright (c) 2026 BAAI. All rights reserved.

"""Numerical contracts of the TXDA capture-safe operator rewrites."""

import os

import pytest
import torch

# Defense-in-depth alongside the conftest gating: skip the whole module when a
# CI job pins a non-TXDA platform (``tests/run.py`` exports FL_TEST_PLATFORM
# from --platform), even if the conftest hook is ever bypassed. An unset value
# means a direct pytest run on a TXDA host, so the tests still run.
_platform = os.environ.get("FL_TEST_PLATFORM")
if _platform and _platform != "txda":
    pytest.skip("TXDA-only contract test", allow_module_level=True)

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
    """The rewrite must not call ``masked_fill`` again (that is the broken op).

    The implementation now uses ``torch.where``; either way, re-entering
    ``masked_fill`` would fall back to the CPU kernel and sync.
    """
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


def _install_and_capture_comparisons(monkeypatch):
    """Register the lt/le overrides on a stub Library and return them.

    The real kernels are closures inside ``install_capture_safe_ops``; the
    stub records what would be registered with the PrivateUse1 key so the
    tests can exercise the exact callables.
    """
    captured = {}

    class _StubLib:
        def impl(self, op, fn, key):
            captured[op] = fn

    monkeypatch.setattr(capture_safe_ops, "_install_masked_fill", lambda lib: None)
    for name in (
        "_install_pow_square",
        "_install_gdn_split_ba",
        "_install_moe_meta_ondevice",
        "_install_gdn_qkv_gate",
        "_install_capture_safe_where",
        "_install_new_zeros_device",
    ):
        monkeypatch.setattr(capture_safe_ops, name, lambda *a, **k: None)
    monkeypatch.setattr(capture_safe_ops, "_installed", False)

    import flag_gems

    monkeypatch.setattr(flag_gems, "aten_lib", _StubLib(), raising=False)
    capture_safe_ops.install_capture_safe_ops()
    return captured


def test_comparison_rewrites_preserve_nan_semantics(monkeypatch):
    """NaN must compare False, as ``torch.lt``/``torch.le`` do."""
    ops = _install_and_capture_comparisons(monkeypatch)
    lt = ops["aten::lt.Scalar"]
    le = ops["aten::le.Scalar"]

    a = torch.tensor([float("nan"), -1.0, 0.0, 1.0])
    # The old ``~ge`` form returned True for NaN; the fix must not.
    assert torch.equal(lt(a, 0.0), torch.lt(a, 0.0))
    assert torch.equal(le(a, 0.0), torch.le(a, 0.0))

    # Same length as ``a``; a shorter tensor would not broadcast.
    b = torch.tensor([float("nan"), 1.0, 0.0, -1.0])
    assert torch.equal(ops["aten::lt.Tensor"](a, b), torch.lt(a, b))
    assert torch.equal(ops["aten::le.Tensor"](a, b), torch.le(a, b))


def test_comparison_rewrites_match_reference(monkeypatch):
    ops = _install_and_capture_comparisons(monkeypatch)
    x = torch.randn(32)
    y = torch.randn(32)
    assert torch.equal(ops["aten::lt.Tensor"](x, y), torch.lt(x, y))
    assert torch.equal(ops["aten::le.Tensor"](x, y), torch.le(x, y))
    assert torch.equal(ops["aten::lt.Scalar"](x, 0.5), torch.lt(x, 0.5))
    assert torch.equal(ops["aten::le.Scalar"](x, 0.5), torch.le(x, 0.5))


def test_comparison_out_overload_writes_the_caller_buffer(monkeypatch):
    ops = _install_and_capture_comparisons(monkeypatch)
    x = torch.tensor([1.0, -1.0])
    out = torch.empty(2, dtype=torch.bool)
    ret = ops["aten::lt.Scalar_out"](x, 0.0, out=out)
    assert ret is out
    assert out.tolist() == [False, True]


# --- Contracts fixed per review 4715513875 / 4715513878 / 4715513883 -------
# These exercise the module-level helpers on CPU; nothing touches a device.


def test_masked_fill_matches_reference_for_special_values():
    """`where`-based fill must equal `torch.masked_fill`, unlike `a*(1-m)+m*v`."""
    from vllm_fl.dispatch.backends.vendor.txda.patches import capture_safe_ops as c

    mask = torch.tensor([True, False])
    cases = [
        (torch.tensor([1.0, 2.0]), float("-inf")),
        (torch.tensor([float("nan"), 2.0]), 0.0),
        (torch.tensor([1.0, float("nan")]), float("inf")),
        (torch.tensor([1, 2], dtype=torch.int64), 5),
        (torch.tensor([1, 2], dtype=torch.int32), 7.0),
        (torch.tensor([True, False]), True),
    ]
    for inp, value in cases:
        got = c._masked_fill_impl(mask, inp, value)
        want = torch.masked_fill(inp, mask, value)
        assert got.dtype == want.dtype, (inp.dtype, got.dtype, want.dtype)
        # A NaN fill makes plain ``torch.equal`` False on matching tensors.
        assert torch.equal(got, want) or (
            torch.allclose(got.float(), want.float(), rtol=0, atol=0, equal_nan=True)
        ), (inp.tolist(), value, got.tolist(), want.tolist())


def test_masked_fill_inplace_returns_same_tensor():
    from vllm_fl.dispatch.backends.vendor.txda.patches import capture_safe_ops as c

    inp = torch.tensor([1.0, 2.0])
    out = c._masked_fill_inplace_impl(torch.tensor([False, True]), inp, 9.0)
    assert out is inp
    assert inp.tolist() == [1.0, 9.0]


def test_masked_fill_never_produces_nan_from_inf():
    """Regression for `0 * -inf == nan` in the old multiply-add form."""
    from vllm_fl.dispatch.backends.vendor.txda.patches import capture_safe_ops as c

    inp = torch.tensor([1.0, 2.0])
    got = c._masked_fill_impl(torch.tensor([True, False]), inp, float("-inf"))
    assert not torch.isnan(got).any()
    assert got.tolist() == [float("-inf"), 2.0]
