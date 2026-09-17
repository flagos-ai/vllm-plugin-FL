# Copyright (c) 2025 BAAI. All rights reserved.

"""Contract tests for the vLLM v0.28.0 MoE factory adaptation."""

from types import SimpleNamespace


def test_factory_patch_does_not_resolve_lazy_module_attributes(monkeypatch):
    import sys
    from types import ModuleType

    import vllm.model_executor.layers.fused_moe as package
    import vllm.model_executor.layers.fused_moe.layer as layer

    import vllm_fl.ops.custom_ops as custom_ops

    original = object()
    replacement = object()
    cached_model = ModuleType("_fl_test_cached_model")
    cached_model.FusedMoEFactory = original
    lazy_module = ModuleType("_fl_test_lazy_module")

    def resolve_attribute(name):
        raise AssertionError(f"Factory patch triggered a lazy import: {name}")

    lazy_module.__getattr__ = resolve_attribute
    monkeypatch.setitem(sys.modules, cached_model.__name__, cached_model)
    monkeypatch.setitem(sys.modules, lazy_module.__name__, lazy_module)
    monkeypatch.setattr(layer, "FusedMoEFactory", original)
    monkeypatch.setattr(package, "FusedMoEFactory", original)
    monkeypatch.setattr(custom_ops, "FusedMoEFactoryFL", replacement)

    custom_ops._patch_fused_moe_factory()
    custom_ops._patch_fused_moe_factory()  # Repeated worker setup is harmless.

    assert cached_model.FusedMoEFactory is replacement
    assert layer.FusedMoEFactory is replacement
    assert package.FusedMoEFactory is replacement


def test_fl_triton_experts_respects_fused_moe_blacklist(monkeypatch):
    import vllm_fl.ops.fused_moe.fused_moe_utils as moe_utils

    platform = SimpleNamespace(
        is_out_of_tree=lambda: True,
        is_cpu=lambda: False,
    )
    monkeypatch.setattr(moe_utils, "current_platform", platform)
    monkeypatch.setattr(moe_utils, "use_flaggems", lambda: True)

    monkeypatch.setattr(moe_utils, "get_oot_blacklist", lambda: [])
    assert moe_utils._should_use_fl_triton_experts()

    monkeypatch.setattr(
        moe_utils,
        "get_oot_blacklist",
        lambda: ["fused_moe"],
    )
    assert not moe_utils._should_use_fl_triton_experts()


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
