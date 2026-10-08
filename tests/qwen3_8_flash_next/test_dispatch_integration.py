"""Semantic dispatch registration, selection and model isolation."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

from vllm_fl.dispatch import BackendImplKind, OpImpl, OpRegistry
from vllm_fl.dispatch.backends.flaggems.register_ops import register_builtins
from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

FUSED_OP_NAMES = {
    "qsa_mqa_paged",
    "expand_qsa_block_indices",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
    "qsa_compress_groups_with_ratio",
    "ple_state_gather",
    "ple_state_scatter_",
    "gdn_packed_decode",
    "compute_common_slot_mapping",
}


def test_semantic_ops_are_owned_by_flaggems_backend(monkeypatch):
    registration = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.register_ops"
    )
    monkeypatch.setattr(
        registration, "use_flaggems_op", lambda name: name in FUSED_OP_NAMES
    )
    registry = OpRegistry()
    register_builtins(registry)
    assert set(registry.list_operators()) == FUSED_OP_NAMES
    for name in registry.list_operators():
        (impl,) = registry.get_implementations(name)
        assert impl.kind == BackendImplKind.DEFAULT
        assert impl.impl_id == "default.flagos"
        assert impl.vendor is None


def test_fused_operator_availability_is_independent(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    flag_gems = ModuleType("flag_gems")
    flag_gems.fused = SimpleNamespace(ple_state_gather=lambda: None)
    monkeypatch.setitem(sys.modules, "flag_gems", flag_gems)
    monkeypatch.setattr(FlagGemsBackend, "_fused_op_availability", {})

    backend = FlagGemsBackend()
    assert backend.fused_op_is_available("ple_state_gather")
    assert not backend.fused_op_is_available("qsa_mqa_paged")


def test_qwen_gdn_subclass_does_not_modify_upstream():
    upstream = importlib.import_module(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )
    recurrent = importlib.import_module(
        "vllm.model_executor.layers.fla.ops.fused_recurrent"
    )
    method = upstream.QwenGatedDeltaNetAttention._forward_core_decode_non_spec
    kernel = recurrent.fused_recurrent_gated_delta_rule_packed_decode_kernel
    from vllm_fl.models.qwen3_8_flash_next import Qwen38GatedDeltaNetAttention

    assert upstream.QwenGatedDeltaNetAttention._forward_core_decode_non_spec is method
    assert recurrent.fused_recurrent_gated_delta_rule_packed_decode_kernel is kernel
    assert Qwen38GatedDeltaNetAttention._forward_core_decode_non_spec is not method
    assert (
        Qwen38GatedDeltaNetAttention.__init__
        is upstream.QwenGatedDeltaNetAttention.__init__
    )


def test_vendor_override_uses_existing_policy_and_denylists(monkeypatch):
    manager = OpManager()
    manager.ensure_initialized()
    name = "ple_state_gather"
    vendor_fn = lambda *args: "test vendor"
    manager.registry.register_impl(
        OpImpl(
            op_name=name,
            impl_id="vendor.test",
            kind=BackendImplKind.VENDOR,
            vendor="test",
            fn=vendor_fn,
        )
    )
    with policy_context(SelectionPolicy(prefer="vendor")):
        assert manager.resolve(name) is vendor_fn
    with policy_context(
        SelectionPolicy(prefer="vendor", deny_vendors=frozenset({"test"}))
    ):
        assert manager.resolve(name) is not vendor_fn
    with policy_context(
        SelectionPolicy.from_dict(per_op_order={name: ["impl:vendor.test"]})
    ):
        assert manager.resolve(name) is vendor_fn
    # A missing optional FlagGems dependency must leave a vendor implementation
    # selectable without importing any fused kernel during registration.
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend

    monkeypatch.setattr(
        FlagGemsBackend, "_fused_op_availability", {"ple_state_gather": False}
    )
    manager.bump_policy_epoch()
    with policy_context(SelectionPolicy(prefer="flagos")):
        assert manager.resolve(name) is vendor_fn


def test_model_registration_preserves_other_models_and_upstream_kernel():
    from vllm.model_executor.models import registry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from vllm_fl.patches.qwen3_8_flash_next import apply_qwen3_8_flash_next_patches

    original = MODELS_CONFIG_MAP.get("Qwen3_5ForConditionalGeneration")
    apply_qwen3_8_flash_next_patches()
    apply_qwen3_8_flash_next_patches()
    assert MODELS_CONFIG_MAP.get("Qwen3_5ForConditionalGeneration") is original
    for arch in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"):
        assert arch in registry.ModelRegistry.get_supported_archs()
    assert "Qwen4ExpMTP" not in MODELS_CONFIG_MAP


def test_model_registration_preserves_future_upstream_ownership(monkeypatch):
    from vllm.model_executor.models import registry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from vllm_fl.patches.qwen3_8_flash_next import apply_qwen3_8_flash_next_patches

    architecture = "Qwen4ExpForCausalLM"
    upstream_verifier = object()
    registrations = []
    monkeypatch.setitem(MODELS_CONFIG_MAP, architecture, upstream_verifier)
    monkeypatch.setattr(
        registry.ModelRegistry,
        "get_supported_archs",
        lambda: {architecture},
    )
    monkeypatch.setattr(
        registry.ModelRegistry,
        "register_model",
        lambda name, model: registrations.append((name, model)),
    )

    apply_qwen3_8_flash_next_patches()

    assert MODELS_CONFIG_MAP[architecture] is upstream_verifier
    assert architecture not in {name for name, _model in registrations}
