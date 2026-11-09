"""K3 library providers use ordinary registry initialization and policy selection."""

import pytest

import vllm_fl.kimi_k3_config as profile
from vllm_fl.dispatch.backends.flaggems import register_ops
from vllm_fl.dispatch.backends.flaggems.impl.kimi_k3_routing import grouped_topk
from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, OpImpl


def register_profile(registry, monkeypatch, enabled=True):
    monkeypatch.setattr(profile, "is_kimi_k3_profile", lambda: enabled)
    register_ops.register_builtins(registry)


def test_k3_provider_is_scoped_and_preserves_generic_registration(monkeypatch):
    registry = OpRegistry()
    register_profile(registry, monkeypatch)
    provider = registry.get_implementation(
        "grouped_topk", "flagos.kimi_k3.grouped_topk"
    )
    assert provider.fn is grouped_topk
    assert provider.kind == BackendImplKind.DEFAULT
    assert len(registry.get_implementations("grouped_topk")) >= 2


def test_normal_profile_has_no_k3_override(monkeypatch):
    registry = OpRegistry()
    register_profile(registry, monkeypatch, False)
    assert (
        registry.get_implementation("grouped_topk", "flagos.kimi_k3.grouped_topk")
        is None
    )
    assert registry.get_implementations("grouped_topk")


def test_conflicting_k3_provider_is_rejected(monkeypatch):
    registry = OpRegistry()
    registry.register_impl(
        OpImpl(
            "grouped_topk",
            "flagos.kimi_k3.grouped_topk",
            BackendImplKind.DEFAULT,
            lambda: None,
        )
    )
    with pytest.raises(RuntimeError, match="Conflicting K3"):
        register_profile(registry, monkeypatch)


def test_disabled_grouped_topk_does_not_add_model_override(monkeypatch):
    monkeypatch.setattr(register_ops, "use_flaggems_op", lambda name: False)
    registry = OpRegistry()
    register_profile(registry, monkeypatch)
    assert (
        registry.get_implementation("grouped_topk", "flagos.kimi_k3.grouped_topk")
        is None
    )


@pytest.mark.parametrize(
    "stage", ["quant", "mla", "custom", "kernel", "cache", "model", "sampler"]
)
def test_k3_registration_failure_restores_previous_entries(monkeypatch, stage):
    from functools import wraps

    import torch

    from vllm.model_executor import custom_op
    from vllm.model_executor.layers import quantization

    from vllm_fl.patches.kimi_k3 import registration as hook, sampling as sampler_hook

    backend = hook.MLAPrefillBackendEnum.CUSTOM
    old_mla = backend.get_path() if backend.is_overridden() else None
    monkeypatch.setattr(hook, "_REGISTERED", False)
    monkeypatch.setattr(hook, "_LIBRARY", None)
    monkeypatch.setattr(
        quantization,
        "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG",
        dict(quantization._CUSTOMIZED_METHOD_TO_QUANT_CONFIG),
    )
    monkeypatch.setattr(custom_op, "op_registry_oot", dict(custom_op.op_registry_oot))
    monkeypatch.setattr(hook.ModelRegistry, "models", dict(hook.ModelRegistry.models))
    before_quant = dict(quantization._CUSTOMIZED_METHOD_TO_QUANT_CONFIG)
    before_custom = dict(custom_op.op_registry_oot)
    before_models = dict(hook.ModelRegistry.models)
    old_name = vars(hook.K3MMEncoderAttention).get("name", hook._MISSING)
    targets = sampler_hook.prepare()
    previous_filters = [m.apply_top_k_top_p for m in targets]

    def failing(fn):
        @wraps(fn)
        def invoke(*args, **kwargs):
            fn(*args, **kwargs)
            # Fail after the actual registry/dispatcher mutation.
            raise RuntimeError("injected registration failure")

        return invoke

    with monkeypatch.context() as failure:
        if stage == "quant":
            original = hook.register_quantization_config

            @wraps(original)
            def register_quant(name):
                return failing(original(name))

            failure.setattr(hook, "register_quantization_config", register_quant)
        elif stage == "mla":
            failure.setattr(
                hook,
                "register_mla_prefill_backend",
                failing(hook.register_mla_prefill_backend),
            )
        elif stage == "custom":
            failure.setattr(
                hook.CustomOp, "register_oot", failing(hook.CustomOp.register_oot)
            )
        elif stage in ("kernel", "cache"):
            original = torch.library.Library.impl

            @wraps(original)
            def impl(library, name, *args, **kwargs):
                result = original(library, name, *args, **kwargs)
                if name == (
                    "situ_and_mul"
                    if stage == "kernel"
                    else "gather_and_maybe_dequant_cache"
                ):
                    raise RuntimeError("injected registration failure")
                return result

            failure.setattr(torch.library.Library, "impl", impl)
        elif stage == "model":
            failure.setattr(
                hook.ModelRegistry,
                "register_model",
                failing(hook.ModelRegistry.register_model),
            )
        else:
            failure.setattr(sampler_hook, "install", failing(sampler_hook.install))
        with pytest.raises(RuntimeError, match="injected registration failure"):
            hook.register()
    assert not hook._REGISTERED and hook._LIBRARY is None
    assert before_quant == quantization._CUSTOMIZED_METHOD_TO_QUANT_CONFIG
    assert custom_op.op_registry_oot == before_custom
    assert hook.ModelRegistry.models == before_models
    assert vars(hook.K3MMEncoderAttention).get("name", hook._MISSING) == old_name
    assert [m.apply_top_k_top_p for m in targets] == previous_filters
    assert (backend.get_path() if backend.is_overridden() else None) == old_mla
    assert (
        "_C_cache_ops::gather_and_maybe_dequant_cache"
        not in torch._C._dispatch_get_all_op_names()
    )
    for name, _ in hook._KERNELS:
        assert not torch._C._dispatch_has_kernel_for_dispatch_key("_C::" + name, "CUDA")

    try:
        hook.register()
        libraries = hook._LIBRARY
        hook.register()
        assert hook._REGISTERED and hook._LIBRARY is libraries
        assert quantization.get_quantization_config("mxfp4") is hook.K3Mxfp4Config
        assert all(
            torch._C._dispatch_has_kernel_for_dispatch_key("_C::" + name, "CUDA")
            for name, _ in hook._KERNELS
        )
    finally:
        for library in hook._LIBRARY or ():
            library._destroy()
        hook._LIBRARY = None
        hook._REGISTERED = False
        if old_mla is None:
            backend.clear_override()
        else:
            hook.register_mla_prefill_backend(backend, old_mla)
        hook._restore_name(old_name)
        for target, previous in zip(targets, previous_filters):
            target.apply_top_k_top_p = previous
