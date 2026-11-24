# Coverage tests: vllm_fl/ops/fused_moe/layer.py (CPU only).

from types import SimpleNamespace

import torch

from vllm_fl.ops.fused_moe import layer


def test_public_aliases():
    assert layer.FusedMoEFL is layer.FusedMoEFactoryFL
    assert layer._OrigFusedMoE is layer._OrigFusedMoEFactory
    assert layer._OrigFusedMoEFactory is not None
    assert set(layer.__all__) == {
        "FusedMoEFactoryFL",
        "FusedMoEFL",
        "UnquantizedFusedMoEMethodFL",
    }


def test_unquantized_method_init_selects_backend(monkeypatch):
    def fake_init(self, moe):
        self.moe = moe

    monkeypatch.setattr(layer.UnquantizedFusedMoEMethod, "__init__", fake_init)
    captured = {}

    def select(moe_config):
        captured["moe_config"] = moe_config
        return "backend", "experts"

    monkeypatch.setattr(layer, "select_unquantized_moe_backend_oot", select)

    config = object()
    instance = layer.UnquantizedFusedMoEMethodFL(config)

    assert captured["moe_config"] is config
    assert instance.unquantized_backend == "backend"
    assert instance.experts_cls == "experts"


def test_is_monolithic_branches():
    getter = layer.UnquantizedFusedMoEMethodFL.is_monolithic.fget

    instance = SimpleNamespace(moe_kernel=None, experts_cls=None)
    assert getter(instance) is True

    class Experts:
        @staticmethod
        def is_monolithic():
            return False

    instance = SimpleNamespace(moe_kernel=None, experts_cls=Experts)
    assert getter(instance) is False

    instance = SimpleNamespace(
        moe_kernel=SimpleNamespace(is_monolithic=True), experts_cls=Experts
    )
    assert getter(instance) is True

    instance = SimpleNamespace(
        moe_kernel=SimpleNamespace(is_monolithic=False), experts_cls=Experts
    )
    assert getter(instance) is False


def _runner(quant_method, recorded):
    def replace(replacement):
        recorded.append(replacement)

    return SimpleNamespace(
        routed_experts=SimpleNamespace(quant_method=quant_method),
        moe_config="config",
        _replace_quant_method=replace,
    )


def test_factory_replaces_unquantized_method(monkeypatch):
    recorded = []
    base_method = object.__new__(layer.UnquantizedFusedMoEMethod)
    runner = _runner(base_method, recorded)
    monkeypatch.setattr(layer, "_OrigFusedMoEFactory", lambda *a, **k: runner)
    replacement = object()
    monkeypatch.setattr(
        layer, "UnquantizedFusedMoEMethodFL", lambda config: replacement
    )
    router_calls = []
    monkeypatch.setattr(layer, "replace_router_with_fl", lambda: router_calls.append(True))

    result = layer.FusedMoEFactoryFL(1, 2, flag=True)

    assert result is runner
    assert recorded == [replacement]
    assert router_calls == [True]


def test_factory_preserves_quantized_method(monkeypatch):
    recorded = []
    runner = _runner(object(), recorded)
    monkeypatch.setattr(layer, "_OrigFusedMoEFactory", lambda *a, **k: runner)
    monkeypatch.setattr(
        layer, "UnquantizedFusedMoEMethodFL", lambda config: object()
    )
    router_calls = []
    monkeypatch.setattr(layer, "replace_router_with_fl", lambda: router_calls.append(True))

    assert layer.FusedMoEFactoryFL() is runner
    assert recorded == []
    assert router_calls == [True]
