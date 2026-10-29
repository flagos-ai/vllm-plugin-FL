"""K3 library providers use ordinary registry initialization and policy selection."""

import pytest

from vllm_fl.dispatch.backends.flaggems import register_ops
from vllm_fl.dispatch.backends.flaggems.impl.kimi_k3_routing import grouped_topk
from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, OpImpl
from vllm_fl.patches.kimi_k3 import profile


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
