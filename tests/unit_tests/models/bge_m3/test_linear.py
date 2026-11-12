# Copyright (c) 2026 BAAI. All rights reserved.
"""Instance isolation, native behavior and real MetaX launch correctness."""

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.parameter import ModelWeightParameter

from vllm_fl.dispatch.backends.flaggems.impl import bge_m3 as portable
from vllm_fl.dispatch.backends.vendor.metax.impl import bge_m3 as metax
from vllm_fl.models.bge_m3 import linear as binding


class FakeLinear(LinearBase):
    def __init__(self):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.randn(4, 3))
        self.quant_method = UnquantizedLinearMethod()


def test_binding_preserves_instances_parameters_loading_and_gradients(monkeypatch):
    monkeypatch.setattr(binding, "resolve_op", lambda name: portable.linear)
    monkeypatch.setattr(
        UnquantizedLinearMethod,
        "apply",
        lambda self, layer, x, bias=None: nn.functional.linear(x, layer.weight, bias),
    )
    model = nn.ModuleList([FakeLinear(), nn.Linear(3, 4)])
    untouched = nn.Linear(3, 4)
    original_forward = untouched.forward
    weights = [layer.weight for layer in model]
    loaders = [model[0].quant_method.process_weights_after_loading]
    binding.bind_linear_policy(model)
    binding.bind_linear_policy(model)
    assert model[0].quant_method.process_weights_after_loading == loaders[0]
    assert untouched.forward == original_forward
    for i, layer in enumerate(model):
        assert layer.weight is weights[i]
        x = torch.randn(2, 3, requires_grad=True)
        bias = None if i == 0 else layer.bias
        expected = nn.functional.linear(x, weights[i], bias)
        actual = layer.quant_method.apply(layer, x) if i == 0 else layer(x)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        assert x.grad is not None and weights[i].grad is not None
    assert set(model.state_dict()) == {"0.weight", "1.weight", "1.bias"}


def test_quantized_method_is_not_replaced(monkeypatch):
    monkeypatch.setattr(binding, "resolve_op", lambda name: portable.linear)
    layer = FakeLinear()
    layer.quant_method = SimpleNamespace(apply=lambda *a: None)
    original = layer.quant_method.apply
    binding.bind_linear_policy(layer)
    assert layer.quant_method.apply is original


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cpu_keeps_native_without_importing_kernel(monkeypatch, dtype):
    monkeypatch.setattr(
        metax, "_linear_entry", lambda: pytest.fail("unexpected kernel import")
    )
    x = torch.randn(2, 1024, dtype=dtype)
    layer = SimpleNamespace(weight=torch.randn(1024, 1024, dtype=dtype))
    seen = []
    sentinel = object()

    def native(*args):
        seen.append(args)
        return sentinel

    assert metax.linear(native, layer, x) is sentinel
    assert seen == [(layer, x, None)]


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("GEMS_VENDOR") != "metax" or not torch.cuda.is_available(),
    reason="requires MetaX GPU",
)
@pytest.mark.parametrize(
    "rows,n,k",
    [
        (1, 1024, 1024),
        (32, 1024, 1024),
        (1024, 3072, 1024),
        (1024, 1024, 1024),
        (1024, 4096, 1024),
        (1024, 1024, 4096),
        (1025, 1024, 1024),
    ],
)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("parameter_kind", ["tensor", "vllm"])
def test_metax_matches_independent_fp64_and_keeps_global_fg(
    monkeypatch, rows, n, k, has_bias, parameter_kind
):
    import importlib

    import flag_gems

    module = importlib.import_module(flag_gems.linear.__module__)
    original_entry = module.linear_kernel
    before = [dict(c.all_kwargs()) for c in original_entry.fn.configs]
    x = torch.randn(rows, k, device="cuda", dtype=torch.float32) * 0.1
    weight = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.1
    bias = torch.randn(n, device="cuda") if has_bias else None
    parameter = (
        ModelWeightParameter.__new__(ModelWeightParameter, data=weight)
        if parameter_kind == "vllm"
        else weight
    )
    layer = SimpleNamespace(weight=parameter)
    seen = []

    def native(layer, value, b):
        seen.append("native")
        return flag_gems.linear(value, layer.weight, b)

    with torch.inference_mode():
        actual = metax.linear(native, layer, x, bias)
    expected = nn.functional.linear(
        x.cpu().double(),
        weight.cpu().double(),
        None if bias is None else bias.cpu().double(),
    )
    torch.testing.assert_close(actual.cpu().double(), expected, atol=2e-5, rtol=2e-5)
    assert seen == (["native"] if rows > 1024 else [])
    assert layer.weight is parameter
    assert layer.weight.data_ptr() == weight.data_ptr()
    assert module.linear_kernel is original_entry
    if rows <= 1024:
        assert [dict(c.all_kwargs()) for c in original_entry.fn.configs] == before
        assert metax._linear_entry()[0].jit_function is original_entry.jit_function


@pytest.mark.gpu
@pytest.mark.skipif(
    os.getenv("GEMS_VENDOR") != "metax" or not torch.cuda.is_available(),
    reason="requires MetaX GPU",
)
@pytest.mark.parametrize(
    "condition",
    ["gradient", "batch_invariant", "noncontiguous", "other_features", "half"],
)
def test_metax_preserves_native_contract_outside_scope(monkeypatch, condition):
    x = torch.randn(2, 1024, device="cuda")
    weight = torch.randn(1024, 1024, device="cuda")
    if condition == "gradient":
        x.requires_grad_(True)
    if condition == "batch_invariant":
        monkeypatch.setattr(metax.vllm_envs, "VLLM_BATCH_INVARIANT", True)
    if condition == "noncontiguous":
        x = torch.randn(1024, 2, device="cuda").t()
    if condition == "other_features":
        weight = weight[:17]
    if condition == "half":
        x, weight = x.half(), weight.half()
    monkeypatch.setattr(
        metax, "_linear_entry", lambda: pytest.fail("unexpected tuning")
    )
    seen = []

    def native(layer, value, bias):
        seen.append(True)
        return nn.functional.linear(value, layer.weight, bias)

    with torch.enable_grad():
        actual = metax.linear(native, SimpleNamespace(weight=weight), x)
    torch.testing.assert_close(actual, nn.functional.linear(x, weight))
    assert seen == [True]


def test_policy_is_resolved_once_per_model_and_isolated(monkeypatch):
    selected = []
    calls = []

    def provider(name):
        def apply(native, layer, x, bias=None):
            calls.append(name)
            return native(layer, x, bias)

        return apply

    choices = iter([provider("first"), provider("second")])

    def resolve(name):
        selected.append(name)
        return next(choices)

    monkeypatch.setattr(binding, "resolve_op", resolve)
    first, second = nn.Linear(3, 4), nn.Linear(3, 4)
    binding.bind_linear_policy(first)
    binding.bind_linear_policy(second)
    x = torch.randn(2, 3)
    first(x)
    second(x)
    first(x)
    assert selected == ["bge_m3_linear", "bge_m3_linear"]
    assert calls == ["first", "second", "first"]
