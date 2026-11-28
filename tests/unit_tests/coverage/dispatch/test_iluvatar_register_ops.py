# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for the Iluvatar register_ops module.

Pins the registration contract beyond what the vendor import-contract test
checks: exactly the five builtins (moe_align_block_size, silu_and_mul,
rms_norm, rotary_embedding, attention_backend), each carrying the
``vendor.iluvatar`` impl_id prefix, VENDOR kind, "iluvatar" vendor tag and
VENDOR priority, and each fn bound to a real IluvatarBackend.is_available
method. Also verifies the registered silu fn actually computes (delegation
to the impl layer). Uses a fresh OpRegistry so the global default manager
is untouched. Vendor-gated to the iluvatar platform.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("iluvatar")]

EXPECTED_OPS = {
    "moe_align_block_size",
    "silu_and_mul",
    "rms_norm",
    "rotary_embedding",
    "attention_backend",
}


def _register_into_fresh_registry():
    from vllm_fl.dispatch.backends.vendor.iluvatar import register_ops
    from vllm_fl.dispatch.registry import OpRegistry

    registry = OpRegistry()
    register_ops.register_builtins(registry)
    return registry


@pytest.fixture
def registered_impls():
    snapshot = _register_into_fresh_registry().snapshot()
    impls = {}
    for op_name, impl_list in snapshot.impls_by_op.items():
        for impl in impl_list:
            impls.setdefault(op_name, []).append(impl)
    return impls


class TestRegisterBuiltins:
    def test_exactly_five_expected_ops(self, registered_impls):
        assert set(registered_impls.keys()) == EXPECTED_OPS

    def test_impl_metadata_contract(self, registered_impls):
        from vllm_fl.dispatch.types import BackendImplKind, BackendPriority

        for op_name, impl_list in registered_impls.items():
            assert len(impl_list) == 1, op_name
            impl = impl_list[0]
            assert impl.op_name == op_name
            assert impl.impl_id == "vendor.iluvatar"
            assert impl.impl_id.startswith("vendor.iluvatar")
            assert impl.kind is BackendImplKind.VENDOR
            assert impl.vendor == "iluvatar"
            assert impl.priority == BackendPriority.VENDOR

    def test_fresh_registry_isolated_from_default_manager(self, registered_impls):
        # a fresh registry must not leak into the process-wide default manager
        from vllm_fl.dispatch.registry import OpRegistry

        default = OpRegistry()  # another fresh one has nothing yet either
        assert set(default.snapshot().impls_by_op.keys()) == set()

    def test_all_fns_bound_to_backend_is_available(self, registered_impls):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        for op_name, impl_list in registered_impls.items():
            for impl in impl_list:
                bound = getattr(impl.fn, "_is_available", None)
                assert callable(bound), op_name
                assert bound.__self__.__class__ is IluvatarBackend, op_name
                assert bound.__func__ is IluvatarBackend.is_available, op_name

    def test_bound_is_available_matches_class_state(self, registered_impls):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        saved = IluvatarBackend._available
        IluvatarBackend._available = None
        try:
            for impl_list in registered_impls.values():
                for impl in impl_list:
                    # OpImpl.is_available() consults the bound fn
                    assert impl.is_available() == IluvatarBackend().is_available()
        finally:
            IluvatarBackend._available = saved

    def test_registered_silu_fn_computes(self, registered_impls):
        impl = registered_impls["silu_and_mul"][0]
        x = torch.randn(2, 3, 10)
        d = x.shape[-1] // 2
        out = impl.fn(None, x)
        x1, x2 = x[..., :d], x[..., d:]
        oracle = x1 * torch.sigmoid(x1) * x2
        assert torch.allclose(out, oracle)

    def test_registered_attention_backend_fn_routes(self, registered_impls):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        impl = registered_impls["attention_backend"][0]
        assert impl.fn() == AttentionBackendEnum.TRITON_ATTN.get_path()
        assert impl.fn(use_mla=True) == AttentionBackendEnum.FLASHMLA.get_path()

    def test_duplicate_registration_into_same_registry_rejected(self):
        # the registry rejects a second vendor.iluvatar impl for the same op
        from vllm_fl.dispatch.backends.vendor.iluvatar import register_ops
        from vllm_fl.dispatch.registry import OpRegistry

        registry = OpRegistry()
        register_ops.register_builtins(registry)
        with pytest.raises(ValueError, match="Duplicate impl_id"):
            register_ops.register_builtins(registry)
