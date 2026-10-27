# SPDX-License-Identifier: Apache-2.0
"""CPU leaf-module contracts without importing a device runtime."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]


def _leaf(monkeypatch, name, filename):
    path = ROOT / "vllm_fl/models/qwen3_8_flash_next" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_residual_default_and_injected_projection_match(monkeypatch):
    residual = _leaf(
        monkeypatch, "_qwen38_residual_projection_test", "common/hyperconnection.py"
    )
    config = residual.HyperConnectionConfig(
        hidden_size=2,
        hc_count=4,
        hc_lowrank=2,
        params_dtype=torch.bfloat16,
        use_fused_ops=False,
    )
    plain = residual.GatedResidualSimple(config, use_mix=False)
    calls = []

    def projection(x, weight):
        value = F.linear(x, weight)
        calls.append((x, weight, value.dtype))
        return value

    injected = residual.GatedResidualSimple(
        config, use_mix=False, small_projection=projection
    )
    injected.block_inject_weight.weight = plain.block_inject_weight.weight
    hyper = torch.randn(3, 8, dtype=torch.bfloat16)
    normalized = torch.randn_like(hyper)
    block = torch.randn(3, 2, dtype=torch.bfloat16)
    expected = plain.combine(block, (hyper, normalized, None))
    actual = injected.combine(block, (hyper, normalized, None))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(calls) == 1
    assert calls[0][0] is normalized
    assert calls[0][1] is plain.block_inject_weight.weight
    assert calls[0][2] == torch.bfloat16


def test_gate_helper_preserves_weight_loader_and_shared_alias(monkeypatch):
    class Unquantized:
        def create_weights(self):
            return "original loader"

        def process_weights_after_loading(self):
            return "original post-load"

    linear_module = ModuleType("vllm.model_executor.layers.linear")
    linear_module.UnquantizedLinearMethod = Unquantized
    dispatch_module = ModuleType("vllm_fl.dispatch")
    dispatch_module.resolve_op = lambda name: F.linear
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", linear_module)
    monkeypatch.setitem(sys.modules, "vllm_fl.dispatch", dispatch_module)
    helpers = _leaf(monkeypatch, "_qwen38_projection_helper_test", "projections.py")
    gate = torch.nn.Module()
    gate.weight = torch.nn.Parameter(torch.randn(1, 8, dtype=torch.bfloat16))
    gate.weight.weight_loader = object()
    gate.bias = None
    gate.quant_method = Unquantized()
    shared = SimpleNamespace(gate=gate)
    weight, loader = gate.weight, gate.weight.weight_loader

    returned = helpers.use_qwen38_shared_expert_projection(gate, F.linear)
    assert returned is gate and shared.gate is gate
    assert gate.weight is weight and gate.weight.weight_loader is loader
    assert gate.quant_method.create_weights() == "original loader"
    assert gate.quant_method.process_weights_after_loading() == "original post-load"
    assert (
        helpers.Qwen38SmallProjectionMethod.create_weights is Unquantized.create_weights
    )
    assert helpers.use_qwen38_shared_expert_projection(gate) is gate
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    torch.testing.assert_close(
        gate.quant_method.apply(gate, x), F.linear(x, weight), atol=0, rtol=0
    )
