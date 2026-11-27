"""RMS fallback and fake metadata preserve the normalization contract."""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.gcu.impl.normalization import (
    _rms_op,
    rms_norm_gcu,
)


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("weight_enabled", [False, True])
@pytest.mark.parametrize("variance", [None, 3])
def test_rms_reference_rounding(residual, weight_enabled, variance):
    x = torch.tensor([[0.125, -2.5, 0.375, 4.25, -0.625]], dtype=torch.bfloat16)
    w = torch.tensor([1.25, -0.5, 0.75, 2.0, -1.0], dtype=x.dtype)
    r = torch.full_like(x, 0.25) if residual else None
    obj = SimpleNamespace(
        weight=w,
        variance_epsilon=1e-6,
        variance_size_override=variance,
        pass_weight=weight_enabled,
        pass_weight_add=weight_enabled,
    )
    value = x.float() if r is None else (x.float() + r.float()).to(x.dtype).float()
    used = value if variance is None else value[..., :variance]
    expected = value * torch.rsqrt(used.square().mean(-1, keepdim=True) + 1e-6)
    if weight_enabled:
        expected = expected * w.float()
    actual = rms_norm_gcu(obj, x, r)
    if residual:
        torch.testing.assert_close(actual[1], value.to(x.dtype), rtol=0, atol=0)
        actual = actual[0]
    torch.testing.assert_close(actual, expected.to(x.dtype), rtol=0, atol=0)


def test_gcu_custom_op_is_eligible_for_decode_rows(monkeypatch):
    """Decode-shaped graphs must not fall back to native fused RMSNorm."""
    if not hasattr(torch, "gcu") or not torch.gcu.is_available():
        pytest.skip("GCU runtime is unavailable")

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    x = torch.randn(1, 8, device="gcu", dtype=torch.bfloat16)
    weight = torch.ones(8, device="gcu", dtype=torch.bfloat16)
    obj = SimpleNamespace(
        weight=weight,
        variance_epsilon=1e-6,
        variance_size_override=None,
        pass_weight=True,
    )
    actual = rms_norm_gcu(obj, x)
    assert actual.shape == x.shape
    assert actual.device == x.device


def test_rms_fake_outputs_are_distinct_and_match_metadata():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(2, 3, 127, dtype=torch.bfloat16)
        y, residual = _rms_op(x, torch.empty(127), None, 127, 1e-6, True)
        assert y is not residual and y is not x
        assert y.shape == residual.shape == x.shape
        assert y.dtype == residual.dtype == x.dtype
