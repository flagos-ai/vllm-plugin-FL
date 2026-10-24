"""Selected profile resolves the original implementations, not substitute kernels."""

import sys
from pathlib import Path

from vllm_fl.dispatch.backends.flaggems.impl.kimi_k3_routing import grouped_topk
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.attention import (
    mla_tp8,
    prefill_mma,
)
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts import (
    grouped_moe,
    tp8_experts,
)
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.kda.interface import (
    chunk_kda_with_fused_gate,
)
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.linear import linear
from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import policy_context, policy_from_config
from vllm_fl.dispatch.types import BackendImplKind
from vllm_fl.patches.kimi_k3 import profile

ROOT = Path(__file__).resolve().parents[2]


def test_profile_resolves_expected_callable_identities(monkeypatch):
    monkeypatch.setattr(profile, "is_kimi_k3_profile", lambda: True)
    manager = OpManager()
    expected = {
        "kimi_k3_prepare_experts": tp8_experts.prepare,
        "kimi_k3_tp8_experts": tp8_experts.run,
        "kimi_k3_grouped_experts": grouped_moe.grouped_experts,
        "kimi_k3_mla_tp8": mla_tp8.run,
        "kimi_k3_prefill_mma": prefill_mma.run,
        "kimi_k3_chunk_kda": chunk_kda_with_fused_gate,
        "kimi_k3_linear": linear.linear,
        "grouped_topk": grouped_topk,
    }
    with policy_context(
        policy_from_config(str(ROOT / "examples/kimi_k3_metax/dispatch.yaml"))
    ):
        for name, fn in expected.items():
            assert manager.resolve(name) is fn
    # The generic vendor capability remains registered, rather than being deleted.
    assert (
        manager.registry.get_implementation("attention_backend", "vendor.metax")
        is not None
    )
    local = manager.registry.get_implementation(
        "kimi_k3_tp8_experts", "vendor.metax.kimi_k3.kimi_k3_tp8_experts"
    )
    assert local.kind == BackendImplKind.VENDOR
    assert not any(
        name == "mcoplib" or name.startswith(("mcoplib.", "vllm_metax"))
        for name in sys.modules
    )


def test_canonical_flaggems_mm_module_is_not_loaded_twice():
    modules = [
        m
        for name, m in sys.modules.items()
        if name.endswith(".ops.mm")
        and getattr(m, "__file__", None)
        and "_metax" in m.__file__
    ]
    assert len({id(m) for m in modules}) == 1
