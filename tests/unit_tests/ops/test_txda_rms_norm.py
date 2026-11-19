# Copyright (c) 2026 BAAI. All rights reserved.

"""Numerical contract of the TXDA RMSNorm fallback."""

import pytest
import torch

from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch
from vllm_fl.dispatch.backends.vendor.txda.impl.normalization import rms_norm_txda


class _Layer:
    """Minimal stand-in for ``vllm.model_executor.layers.layernorm.RMSNorm``."""

    def __init__(
        self, hidden_size, eps=1e-6, var_hidden_size=None, dtype=torch.bfloat16
    ):
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size in (None, hidden_size) else var_hidden_size
        )
        self.weight = torch.ones(hidden_size, dtype=dtype)
        self.pass_weight = True
        self.pass_weight_add = True


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_matches_reference_implementation(dtype):
    layer = _Layer(64, dtype=dtype)
    torch.manual_seed(0)
    layer.weight = torch.randn(64, dtype=dtype)
    x = torch.randn(8, 64, dtype=dtype)
    residual = torch.randn(8, 64, dtype=dtype)

    got, got_residual = rms_norm_txda(layer, x, residual)
    want, want_residual = rms_norm_torch(layer, x, residual)

    assert got.dtype == dtype
    assert got_residual.dtype == dtype
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    torch.testing.assert_close(got_residual, want_residual, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_large_fp16_inputs_do_not_overflow(dtype):
    """Squaring in the input dtype would overflow; FP32 accumulation must not."""
    layer = _Layer(32, dtype=dtype)
    x = torch.full((4, 32), 300.0, dtype=dtype)

    out = rms_norm_txda(layer, x)

    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    # A unit weight leaves the normalized vector at ~1 for a constant input.
    torch.testing.assert_close(out, torch.ones_like(out), rtol=1e-2, atol=1e-2)


def test_honors_variance_size_override():
    """A truncated variance window must change the result."""
    full = _Layer(64, var_hidden_size=64)
    part = _Layer(64, var_hidden_size=16)
    assert full.variance_size_override is None
    assert part.variance_size_override == 16

    x = torch.randn(2, 64)
    # Make the tail dominate the variance so the two windows differ visibly.
    x[:, 16:] *= 40.0

    out_full = rms_norm_txda(full, x)
    out_part = rms_norm_txda(part, x)

    assert not torch.allclose(out_full, out_part)
    torch.testing.assert_close(out_part, rms_norm_torch(part, x), rtol=0, atol=0)


def test_residual_is_returned_as_the_accumulated_sum():
    layer = _Layer(8)
    x = torch.ones(1, 8)
    residual = torch.full((1, 8), 2.0)

    out, new_residual = rms_norm_txda(layer, x, residual)

    torch.testing.assert_close(new_residual, torch.full((1, 8), 3.0))
    torch.testing.assert_close(out, torch.ones(1, 8))


def test_weightless_layer_skips_the_multiply():
    """``has_weight=False`` layers pass ``weight=None`` semantics upstream."""
    layer = _Layer(8)
    layer.pass_weight = False
    layer.pass_weight_add = False
    layer.weight = torch.full((8,), 5.0)

    x = torch.randn(2, 8)
    out = rms_norm_txda(layer, x)

    # Same as the reference, and not scaled by the (unused) weight.
    torch.testing.assert_close(out, rms_norm_torch(layer, x), rtol=0, atol=0)
    expected = x.float() * torch.rsqrt(
        x.float().square().mean(dim=-1, keepdim=True) + layer.variance_epsilon
    )
    torch.testing.assert_close(out, expected.to(x.dtype), rtol=0, atol=0)
    assert not torch.allclose(out, expected.to(x.dtype) * 5.0)
