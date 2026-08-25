# Copyright (c) 2025 BAAI. All rights reserved.

"""Contract tests for the vLLM v0.28.0 MoE factory adaptation."""


def test_factory_reads_quant_method_from_routed_experts(monkeypatch):
    import vllm_fl.ops.fused_moe.layer as layer

    class UpstreamUnquantizedMethod:
        pass

    class RoutedExperts:
        def __init__(self):
            self.quant_method = UpstreamUnquantizedMethod()

    class Runner:
        def __init__(self):
            self.routed_experts = RoutedExperts()
            self.moe_config = object()
            self.replacement = None

        def _replace_quant_method(self, replacement):
            self.replacement = replacement

    runner = Runner()
    replacement = object()

    monkeypatch.setattr(layer, "_OrigFusedMoEFactory", lambda: runner)
    monkeypatch.setattr(
        layer,
        "UnquantizedFusedMoEMethod",
        UpstreamUnquantizedMethod,
    )
    monkeypatch.setattr(
        layer,
        "UnquantizedFusedMoEMethodFL",
        lambda _config: replacement,
    )
    monkeypatch.setattr(layer, "replace_router_with_fl", lambda: None)

    assert layer.FusedMoEFactoryFL() is runner
    assert runner.replacement is replacement


def test_factory_preserves_quantized_method(monkeypatch):
    import vllm_fl.ops.fused_moe.layer as layer

    class UpstreamUnquantizedMethod:
        pass

    class RoutedExperts:
        quant_method = object()

    class Runner:
        routed_experts = RoutedExperts()
        moe_config = object()

        def _replace_quant_method(self, _replacement):
            raise AssertionError("quantized method must not be replaced")

    runner = Runner()
    monkeypatch.setattr(layer, "_OrigFusedMoEFactory", lambda: runner)
    monkeypatch.setattr(
        layer,
        "UnquantizedFusedMoEMethod",
        UpstreamUnquantizedMethod,
    )
    monkeypatch.setattr(layer, "replace_router_with_fl", lambda: None)

    assert layer.FusedMoEFactoryFL() is runner
