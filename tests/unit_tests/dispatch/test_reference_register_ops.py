# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for the reference backend's register_builtins wiring."""

import sys

import pytest

from vllm_fl.dispatch.backends.reference.register_ops import (
    _bind_is_available,
    _reference_w8a8_moe_experts,
    register_builtins,
)
from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Give each test a private registry so registrations never leak."""
    registry = OpRegistry()
    yield registry


class TestRegisterBuiltins:
    """register_builtins populates a fresh registry with reference impls."""

    def test_all_implemented_ops_registered(self, _fresh_registry):
        """Every backend attr present is registered with reference metadata."""
        register_builtins(_fresh_registry)
        ops = set(_fresh_registry.list_operators())
        # 11 backend ops (grouped_topk absent on this build) + w8a8_moe_experts.
        assert "silu_and_mul" in ops
        assert "rms_norm" in ops
        assert "rotary_embedding" in ops
        assert "topk_softmax" in ops
        assert "moe_align_block_size" in ops
        assert "moe_sum" in ops
        assert "dynamic_per_token_quant_int8" in ops
        assert "w8a8_moe_experts" in ops
        assert "attention_backend" in ops
        assert "invoke_fused_moe_triton_kernel" in ops
        assert "gelu_and_mul" in ops

    def test_registered_impl_metadata(self, _fresh_registry):
        """Each impl carries the reference impl_id, kind and priority."""
        register_builtins(_fresh_registry)
        impl = _fresh_registry.get_implementation("silu_and_mul", "reference.torch")
        assert impl is not None
        assert impl.kind == BackendImplKind.REFERENCE
        assert impl.priority == BackendPriority.REFERENCE
        assert impl.vendor is None
        assert impl.is_available() is True

    def test_grouped_topk_absent_does_not_register(self, _fresh_registry):
        """An op missing from the backend (grouped_topk) is skipped silently."""
        register_builtins(_fresh_registry)
        assert "grouped_topk" not in _fresh_registry.list_operators()

    def test_duplicate_registration_raises(self, _fresh_registry):
        """Re-registering the same impl_id is rejected by the registry."""
        register_builtins(_fresh_registry)
        with pytest.raises(ValueError, match="Duplicate impl_id 'reference.torch'"):
            register_builtins(_fresh_registry)

    def test_bound_fn_calls_through_and_reports_availability(self):
        """_bind_is_available wraps transparently and exposes the probe."""
        calls = []

        def fn(a, b=0):
            calls.append((a, b))
            return a + b

        def probe():
            return False

        wrapped = _bind_is_available(fn, probe)
        assert wrapped(1, b=2) == 3
        assert calls == [(1, 2)]
        assert wrapped._is_available is probe
        assert wrapped.__name__ == "fn"


class TestReferenceW8A8MoeExperts:
    """_reference_w8a8_moe_experts activation guard."""

    def test_non_silu_activation_rejected(self):
        """Only SwiGLU ('silu') is supported; anything else raises."""
        with pytest.raises(
            NotImplementedError, match="Reference W8A8 MoE supports SwiGLU only"
        ):
            _reference_w8a8_moe_experts(
                activation="gelu", global_num_experts=8, w1=None, w2=None
            )

    def test_silu_dispatches_to_native_experts(self, monkeypatch):
        """The silu path forwards kwargs to _native_w8a8_fused_experts."""
        import types

        # moe_experts imports vllm at module level; stub the chain it needs.
        vllm_mod = types.ModuleType("vllm")
        vllm_mod.__path__ = []
        me = types.ModuleType("vllm.model_executor")
        me.__path__ = []
        layers = types.ModuleType("vllm.model_executor.layers")
        layers.__path__ = []
        fm = types.ModuleType("vllm.model_executor.layers.fused_moe")
        fm.__path__ = []
        mk = types.ModuleType("vllm.model_executor.layers.fused_moe.modular_kernel")
        act = types.ModuleType("vllm.model_executor.layers.fused_moe.activation")
        act.MoEActivation = object
        triton_moe = types.ModuleType(
            "vllm.model_executor.layers.fused_moe.experts.triton_moe"
        )
        triton_moe.TritonExperts = object
        fused = types.ModuleType("vllm.model_executor.layers.fused_moe.fused_moe")
        fused.fused_experts = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
        monkeypatch.setitem(sys.modules, "vllm.model_executor", me)
        monkeypatch.setitem(sys.modules, "vllm.model_executor.layers", layers)
        monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fused_moe", fm)
        monkeypatch.setitem(
            sys.modules, "vllm.model_executor.layers.fused_moe.modular_kernel", mk
        )
        monkeypatch.setitem(
            sys.modules, "vllm.model_executor.layers.fused_moe.activation", act
        )
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.layers.fused_moe.experts.triton_moe",
            triton_moe,
        )
        monkeypatch.setitem(
            sys.modules, "vllm.model_executor.layers.fused_moe.fused_moe", fused
        )
        vllm_mod.model_executor = me
        me.layers = layers
        layers.fused_moe = fm
        fm.modular_kernel = mk
        fm.activation = act
        fm.fused = fused

        import vllm_fl.quantization.w8a8.moe_experts as moe_experts

        captured = {}

        def fake_native(**kwargs):
            captured.update(kwargs)
            return "sentinel"

        monkeypatch.setattr(
            moe_experts, "_native_w8a8_fused_experts", fake_native, raising=False
        )
        out = _reference_w8a8_moe_experts(
            activation="silu", global_num_experts=4, hidden=16
        )
        assert out == "sentinel"
        assert captured == {"hidden": 16}
