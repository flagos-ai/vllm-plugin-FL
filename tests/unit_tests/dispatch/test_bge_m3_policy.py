# Copyright (c) 2026 BAAI. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.flaggems.impl import bge_m3 as provider
from vllm_fl.dispatch.backends.vendor.metax.impl import bge_m3 as metax
from vllm_fl.dispatch.backends.vendor.metax.metax import MacaBackend
from vllm_fl.patches import bge_m3 as registration


@pytest.mark.parametrize(
    "arch,dtype,mla,sparse,selected",
    [
        ("BgeM3EmbeddingModelFL", torch.float32, False, False, True),
        ("BgeM3EmbeddingModelFL", torch.float16, False, False, False),
        ("OtherModel", torch.float32, False, False, False),
        ("BgeM3EmbeddingModelFL", torch.float32, True, False, False),
        ("BgeM3EmbeddingModelFL", torch.float32, True, True, False),
    ],
)
def test_vendor_selection_is_scoped(monkeypatch, arch, dtype, mla, sparse, selected):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=dtype, hf_config=SimpleNamespace(architectures=[arch])
        )
    )
    monkeypatch.setattr(metax, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    monkeypatch.setattr(
        MacaBackend, "attention_backend", lambda self, **kw: ("native", kw)
    )
    actual = metax.attention_backend(mla, sparse)
    if selected:
        assert actual == "vllm_fl.models.bge_m3.attention.BgeM3AttentionBackend"
    else:
        assert actual == ("native", {"use_mla": mla, "use_sparse": sparse})


def test_no_model_preserves_vendor_default(monkeypatch):
    monkeypatch.setattr(metax, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(MacaBackend, "attention_backend", lambda self, **kw: "native")
    assert metax.attention_backend() == "native"


def test_fgv_disabled_fails_explicitly(monkeypatch):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.float32,
            hf_config=SimpleNamespace(architectures=["BgeM3EmbeddingModelFL"]),
        )
    )
    monkeypatch.setattr(metax, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "0")
    with pytest.raises(RuntimeError, match="requires FlagGems-vllm"):
        metax.attention_backend()
    with pytest.raises(RuntimeError, match="requires FlagGems-vllm"):
        provider.encoder_attention(None, None, None, causal=False, softmax_scale=0.125)


def test_provider_reuses_exact_library_contract(monkeypatch):
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    seen = []
    result = object()

    def kernel(*args, **kwargs):
        seen.append((args, kwargs))
        return result

    monkeypatch.setattr(provider, "_varlen", lambda: kernel)
    q, k, v = object(), object(), object()
    assert (
        provider.encoder_attention(q, k, v, causal=False, softmax_scale=0.125) is result
    )
    assert seen == [((q, k, v), {"causal": False, "softmax_scale": 0.125})]


def test_registration_is_lazy_and_idempotent(monkeypatch):
    from vllm.model_executor.models import ModelRegistry

    architectures = set()
    calls = []
    monkeypatch.setattr(ModelRegistry, "get_supported_archs", lambda: architectures)

    def register(name, path):
        calls.append((name, path))
        architectures.add(name)

    monkeypatch.setattr(ModelRegistry, "register_model", register)
    registration.register_bge_m3_support()
    registration.register_bge_m3_support()
    assert calls == [
        ("BgeM3EmbeddingModelFL", "vllm_fl.models.bge_m3.model:BgeM3EmbeddingModelFL")
    ]
    assert (
        registration.register_bge_m3_processor()
        == "vllm_fl.models.bge_m3.io_processor.BgeM3IOProcessor"
    )


def test_policy_selects_vendor_without_replacing_default(monkeypatch):
    from vllm_fl.dispatch.manager import OpManager
    from vllm_fl.dispatch.policy import SelectionPolicy
    from vllm_fl.dispatch.registry import OpRegistry
    from vllm_fl.dispatch.types import BackendImplKind, OpImpl

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    monkeypatch.setattr(provider, "_varlen", lambda: lambda *a, **k: None)
    registry = OpRegistry()
    registry.register_impl(
        OpImpl(
            "attention_backend",
            "default.flagos",
            BackendImplKind.DEFAULT,
            lambda: "default",
        )
    )
    metax.register_bge_m3_operators(registry, is_platform_available=lambda: True)
    provider.register_bge_m3_operators(registry, allow=lambda name: True)
    manager = OpManager(registry)
    from vllm_fl.dispatch.policy import policy_context

    monkeypatch.setattr(manager, "ensure_initialized", lambda: None)
    with policy_context(SelectionPolicy()):
        assert (
            manager.resolve("attention_backend")
            is registry.get_implementation("attention_backend", "default.flagos").fn
        )
    selected = SelectionPolicy.from_dict(
        strict=True, per_op_order={"attention_backend": ["impl:vendor.metax.bge_m3"]}
    )
    with policy_context(selected):
        assert (
            manager.resolve("attention_backend")
            is registry.get_implementation(
                "attention_backend", "vendor.metax.bge_m3"
            ).fn
        )
    denied = SelectionPolicy.from_dict(
        strict=True,
        per_op_order={"attention_backend": ["impl:vendor.metax.bge_m3"]},
        deny_vendors={"metax"},
    )
    with policy_context(denied), pytest.raises(RuntimeError, match="No implementation"):
        manager.resolve("attention_backend")
    assert (
        registry.get_implementation("bge_m3_encoder_attention", "default.flagos")
        is not None
    )


def test_registration_availability_is_not_shared_between_registries():
    from vllm_fl.dispatch.registry import OpRegistry

    enabled, disabled = OpRegistry(), OpRegistry()
    metax.register_bge_m3_operators(enabled, is_platform_available=lambda: True)
    metax.register_bge_m3_operators(disabled, is_platform_available=lambda: False)
    assert enabled.get_implementation(
        "attention_backend", "vendor.metax.bge_m3"
    ).is_available()
    assert not disabled.get_implementation(
        "attention_backend", "vendor.metax.bge_m3"
    ).is_available()
