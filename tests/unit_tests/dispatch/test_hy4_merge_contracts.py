# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU wiring regressions for PPU HY4 coexisting with main's MLA integrations.

Optional runtime imports are isolated. These tests exercise production registry
and selection code, not hardware kernels or vLLM model execution.
"""

import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def isolated_runtime(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    for name in list(sys.modules):
        if (
            name == "vllm_fl"
            or name.startswith(("vllm_fl.", "vllm."))
            or name == "vllm"
        ):
            monkeypatch.delitem(sys.modules, name)

    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    # Preserve real source imports without initializing unrelated runtime SDKs.
    for name in (
        "vllm_fl",
        "vllm_fl.dispatch",
        "vllm_fl.dispatch.backends",
        "vllm_fl.dispatch.backends.flaggems",
        "vllm_fl.dispatch.backends.flaggems.impl",
        "vllm_fl.dispatch.backends.reference",
        "vllm_fl.dispatch.backends.reference.impl",
        "vllm_fl.models",
        "vllm_fl.models.hy4",
        "vllm_fl.attention",
        "vllm_fl.configs",
        "vllm_fl.model_loader",
        "vllm_fl.quantization",
        "vllm_fl.quantization.w8a8",
    ):
        module(name, __path__=[str(root / name.replace(".", "/"))])
    for name in (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.models",
        "vllm.transformers_utils",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
        "vllm.v1.attention.backends.mla",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fused_moe",
        "vllm.model_executor.layers.fused_moe.experts",
    ):
        module(name, __path__=[])
    platform = SimpleNamespace(vendor_name="nvidia", is_cuda=lambda: True)
    module("vllm.platforms", current_platform=platform)
    module("vllm.logger", init_logger=logging.getLogger)
    module(
        "vllm_fl.utils", use_flaggems_op=lambda name: True, use_flaggems=lambda: True
    )
    # Unrelated query-quant custom-op registration must not leak between tests.
    module(
        "vllm_fl.dispatch.backends.flaggems.impl.query_quant",
        flaggems_per_token_group_quant_fp8=lambda *args: None,
    )
    before = set(sys.modules)
    yield SimpleNamespace(module=module, platform=platform)
    for name in set(sys.modules) - before:
        if name.startswith(("vllm_fl.", "vllm.")):
            sys.modules.pop(name, None)


def test_registry_keeps_distinct_topk_contracts(isolated_runtime, monkeypatch):
    module = isolated_runtime.module
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "0")
    registration = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.register_ops"
    )
    reference = importlib.import_module(
        "vllm_fl.dispatch.backends.reference.register_ops"
    )
    registry = importlib.import_module("vllm_fl.dispatch.registry").OpRegistry()
    registration.register_builtins(registry)
    reference.register_builtins(registry)
    calls = []
    module("flaggems_vllm", top_k_per_row_prefill=lambda *args: calls.append(args))
    upstream = registry.get_implementation("top_k_per_row_prefill", "default.flagos")
    args = tuple(object() for _ in range(8))
    upstream.fn(*args)
    assert calls == [args]
    bf16 = registry.get_implementation(
        "bf16_indexer_top_k_per_row_prefill", "reference.torch"
    )
    logits = torch.tensor([[9.0, 2.0, 5.0, 4.0]])
    indices = torch.full((1, 2), -99, dtype=torch.int32)
    bf16.fn(
        logits,
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        indices,
    )
    assert indices.tolist() == [[1, 2]]
    assert (
        registry.get_implementation("top_k_per_row_prefill", "reference.torch") is None
    )
    assert (
        registry.get_implementation(
            "bf16_indexer_top_k_per_row_prefill", "default.flagos"
        )
        is not None
    )


@pytest.mark.parametrize("vendor", ["thead", "nvidia", "ascend"])
def test_hy4_registration_is_platform_scoped(isolated_runtime, vendor):
    module = isolated_runtime.module
    isolated_runtime.platform.vendor_name = vendor
    models, loaders, prefill_calls = {}, {}, []
    model_loader = module(
        "vllm.model_executor.model_loader", _LOAD_FORMAT_TO_MODEL_LOADER=loaders
    )
    model_loader.register_model_loader = lambda name: lambda cls: loaders.setdefault(
        name, cls
    )
    module(
        "vllm.model_executor.models.registry",
        ModelRegistry=SimpleNamespace(
            register_model=lambda name, path: models.__setitem__(name, path)
        ),
    )
    config = module("vllm.transformers_utils.config", _CONFIG_REGISTRY={})
    convertor = module(
        "vllm.transformers_utils.model_arch_config_convertor",
        MODEL_ARCH_CONFIG_CONVERTORS={},
        ModelArchConfigConvertorBase=object,
    )
    main_config, ppu_config, main_loader, ppu_loader, converter = [
        type(n, (), {})
        for n in ("MainConfig", "PPUConfig", "MainLoader", "PPULoader", "Converter")
    ]
    module("vllm_fl.models.hy4.config", HYV4Config=main_config)
    module(
        "vllm_fl.models.hy4.config_convertor", HYV4ModelArchConfigConvertor=converter
    )
    module("vllm_fl.models.hy4.loader", HYV4SafetensorsLoader=main_loader)
    module("vllm_fl.configs.hy_v4", HYV4Config=ppu_config)
    module("vllm_fl.model_loader.hy_v4_loader", HYV4SafetensorsLoader=ppu_loader)
    module(
        "vllm_fl.attention.mla_prefill",
        register_flaggems_mla_prefill=lambda: prefill_calls.append(True),
    )
    registration = importlib.import_module("vllm_fl.models.hy4.registration")
    for _ in range(2):
        assert registration.register_hy_v4_support()
    ppu = vendor == "thead"
    assert models == {
        "HYV4ForCausalLM": "vllm_fl.models.hy_v4:HYV4ForCausalLM"
        if ppu
        else "vllm_fl.models.hy4.model:HYV4ForCausalLM"
    }
    assert config._CONFIG_REGISTRY["hy_v4"] is (ppu_config if ppu else main_config)
    assert loaders["hy4_safetensors"] is (ppu_loader if ppu else main_loader)
    selected_converter = convertor.MODEL_ARCH_CONFIG_CONVERTORS["hy_v4"]
    if ppu:
        instance = selected_converter()
        instance.hf_text_config = SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
        assert instance.get_head_size() == 576
        assert instance.is_deepseek_mla()
    else:
        assert selected_converter is converter
    assert bool(prefill_calls) is not ppu


@pytest.fixture
def prefill_runtime(isolated_runtime):
    module = isolated_runtime.module
    custom = SimpleNamespace(is_overridden=lambda: False, get_path=lambda: None)
    enum = SimpleNamespace(CUSTOM=custom)
    registered = []
    register = lambda *args: registered.append(args)
    module(
        "vllm.v1.attention.backends.mla.prefill",
        __path__=[],
        MLAPrefillBackendEnum=enum,
        register_mla_prefill_backend=register,
    )
    module(
        "vllm.v1.attention.backends.mla.prefill.base",
        MLAPrefillBackend=type("Base", (), {}),
    )
    module(
        "vllm.v1.attention.backends.mla.prefill.registry",
        MLAPrefillBackendEnum=enum,
        register_mla_prefill_backend=register,
    )
    dispatch = sys.modules["vllm_fl.dispatch"]
    dispatch.CachedOp = lambda name: None
    dispatch.resolve_op = lambda name: None
    implementation = importlib.import_module("vllm_fl.attention.mla_prefill")
    return implementation, custom, registered


@pytest.mark.parametrize("vendor", ["thead", "nvidia", "ascend"])
@pytest.mark.parametrize("explicit", [False, True])
def test_ppu_prefill_preserves_other_platforms_and_explicit_selection(
    isolated_runtime, prefill_runtime, vendor, explicit
):
    implementation, custom, registered = prefill_runtime
    isolated_runtime.platform.vendor_name = vendor
    selection = object() if explicit else None
    config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=True),
        attention_config=SimpleNamespace(mla_prefill_backend=selection),
    )
    expected = vendor == "thead" and not explicit
    assert implementation.configure_mla_prefill(config) is expected
    assert bool(registered) is expected
    assert config.attention_config.mla_prefill_backend is (
        custom if expected else selection
    )


def test_ppu_prefill_preserves_foreign_custom_backend(
    isolated_runtime, prefill_runtime
):
    implementation, custom, registered = prefill_runtime
    isolated_runtime.platform.vendor_name = "thead"
    custom.is_overridden = lambda: True
    custom.get_path = lambda: "other.Backend"
    config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=True),
        attention_config=SimpleNamespace(mla_prefill_backend=custom),
    )
    assert not implementation.configure_mla_prefill(config)
    assert registered == []


@pytest.mark.parametrize("vendor", ["thead", "nvidia", "ascend"])
@pytest.mark.parametrize("sparse", [False, True])
def test_legacy_mla_fallback_is_ppu_only(isolated_runtime, vendor, sparse):
    isolated_runtime.platform.vendor_name = vendor
    isolated_runtime.module(
        "vllm.v1.attention.backends.registry", AttentionBackendEnum=object()
    )
    backend = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.flaggems"
    ).FlagGemsBackend()
    if vendor == "thead":
        result = backend.attention_backend(use_mla=True, use_sparse=sparse)
        assert result.endswith("MLASparseFLBackend" if sparse else "MLAFLBackend")
    else:
        with pytest.raises(NotImplementedError, match="model runtime plan"):
            backend.attention_backend(use_mla=True, use_sparse=sparse)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_w8a8_respects_selected_provider_without_retry(
    isolated_runtime, monkeypatch, enabled, failure
):
    module = isolated_runtime.module
    module("vllm.model_executor.layers.fused_moe.modular_kernel")
    module("vllm.model_executor.layers.fused_moe.activation", MoEActivation=object())
    module(
        "vllm.model_executor.layers.fused_moe.experts.triton_moe",
        TritonExperts=type("Experts", (), {}),
    )
    module(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        fused_experts=lambda **kw: None,
    )
    calls = []

    def provider(name):
        def invoke(**kwargs):
            calls.append(name)
            if failure:
                raise RuntimeError("selected provider failed")
            return kwargs["hidden_states"]

        return invoke

    module("flaggems_vllm", fused_experts_impl=provider("specialized"))
    module("flag_gems", fused_experts_impl=provider("generic"))
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1" if enabled else "0")
    experts = importlib.import_module("vllm_fl.quantization.w8a8.moe_experts")
    hidden = object()
    if failure:
        with pytest.raises(RuntimeError, match="selected provider failed"):
            experts._flaggems_fused_experts_impl(hidden_states=hidden)
    else:
        assert experts._flaggems_fused_experts_impl(hidden_states=hidden) is hidden
    assert calls == ["specialized" if enabled else "generic"]


def test_ppu_policy_uses_bf16_contract_names():
    import yaml

    root = Path(__file__).resolve().parents[3]
    policy = yaml.safe_load((root / "vllm_fl/dispatch/config/thead.yaml").read_text())
    for phase in ("prefill", "decode"):
        assert policy["op_backends"][f"bf16_indexer_top_k_per_row_{phase}"] == [
            "flagos",
            "reference",
        ]
