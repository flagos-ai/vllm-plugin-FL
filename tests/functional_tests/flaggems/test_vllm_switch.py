# Copyright (c) 2026 BAAI. All rights reserved.

import sys

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.flaggems]


@pytest.mark.parametrize("value", [None])
def test_disabled_clamp_uses_native_without_package(monkeypatch, value):
    from vllm_fl.ops.activation import _silu_and_mul_with_clamp_graph

    if value is None:
        monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_VLLM", raising=False)
    else:
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", value)
    monkeypatch.setitem(sys.modules, "flaggems_vllm", None)
    torch.manual_seed(29)
    x = torch.randn(7, 192, device="cuda", dtype=torch.bfloat16)
    gate, up = x.chunk(2, dim=-1)
    gate, up = gate.clamp(max=3.0), up.clamp(min=-3.0, max=3.0)
    expected = gate * torch.sigmoid(gate) * up
    actual = _silu_and_mul_with_clamp_graph(x, 3.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("value", [None, "1"])
def test_quantization_switch_preserves_dynamic_contract(monkeypatch, value):
    from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
        dynamic_per_token_quant_int8_flaggems_vllm,
    )

    if value is None:
        monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_VLLM", raising=False)
    else:
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", value)
    torch.manual_seed(31)
    x = torch.randn(7, 37, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    q, scale = dynamic_per_token_quant_int8_flaggems_vllm(x)
    absmax = x.float().abs().amax(dim=-1, keepdim=True)
    inverse = torch.where(absmax != 0, 127.0 / absmax, 0.0)
    expected = (x.float() * inverse).round().clamp(-128, 127).to(torch.int8)
    assert q.shape == x.shape and q.dtype == torch.int8
    assert scale.shape == (x.shape[0], 1) and scale.dtype == torch.float32
    torch.testing.assert_close(q, expected, rtol=0, atol=0)
    torch.testing.assert_close(scale, absmax / 127, rtol=1e-5, atol=1e-7)
