# Copyright (c) 2026 BAAI. All rights reserved.

"""Helper-function tests for vllm_fl.ops._C_ops_registry.

register_op_schemas' schema-definition paths are already covered by
tests/unit_tests/ops/test_op_schema_registry.py; this module targets the
pure helpers around it: the repetition-penalty fallback oracle, the
unavailable-view error paths, the MetaX backend detection, and the
optional-op exclusion list.
"""

import pytest
import torch

import vllm_fl.ops._C_ops_registry as registry
from vllm_fl.ops._C_ops_registry import (
    _OPTIONAL_NATIVE_ONLY_OPS,
    _QUERY_OP_IMPLS,
    _accelerator_view_from_cpu_tensor,
    _apply_repetition_penalties_impl,
    _is_metax_backend,
    _unavailable_cuda_view,
    _unavailable_weak_ref,
)


class TestApplyRepetitionPenalties:
    """Pure-torch fallback for _C::apply_repetition_penalties_."""

    def test_matches_independent_oracle(self):
        """Positive logits divide by the penalty; negative logits multiply."""
        logits = torch.tensor([[2.0, -2.0, 0.0, 1.0]])
        prompt_mask = torch.tensor([[True, False, False, False]])
        output_mask = torch.tensor([[False, False, False, True]])
        penalties = torch.tensor([2.0])

        _apply_repetition_penalties_impl(logits, prompt_mask, output_mask, penalties)

        # Independent oracle: p = 2 where masked else 1; pos→x/p, neg→x*p.
        expected = torch.tensor([[2.0 / 2.0, -2.0, 0.0, 1.0 / 2.0]])
        torch.testing.assert_close(logits, expected, rtol=1e-6, atol=1e-7)

    def test_masked_positions_unmodified(self):
        """Positions in neither mask keep their logits untouched."""
        logits = torch.tensor([[5.0, -5.0]])
        no_mask = torch.tensor([[False, False]])
        penalties = torch.tensor([3.0])
        _apply_repetition_penalties_impl(logits, no_mask, no_mask.clone(), penalties)
        torch.testing.assert_close(logits, torch.tensor([[5.0, -5.0]]))

    def test_penalty_one_is_identity(self):
        """A penalty of 1.0 is a no-op wherever masks are set."""
        logits = torch.randn(3, 6)
        saved = logits.clone()
        mask = torch.ones(3, 6, dtype=torch.bool)
        _apply_repetition_penalties_impl(logits, mask, ~mask, torch.ones(3))
        torch.testing.assert_close(logits, saved, rtol=1e-6, atol=1e-7)

    def test_in_place_mutation_and_row_independence(self):
        """The op mutates in place and applies per-row penalties."""
        logits = torch.tensor([[4.0], [-4.0]])
        mask = torch.ones(2, 1, dtype=torch.bool)
        out = _apply_repetition_penalties_impl(
            logits, mask, ~mask, torch.tensor([2.0, 4.0])
        )
        assert out is None  # in-place convention, no return
        torch.testing.assert_close(logits, torch.tensor([[2.0], [-16.0]]))

    def test_prompt_and_output_masks_combine(self):
        """Either mask being true applies the penalty (logical OR)."""
        logits = torch.tensor([[3.0, 3.0, 3.0]])
        prompt = torch.tensor([[True, False, False]])
        output = torch.tensor([[False, True, False]])
        penalties = torch.tensor([3.0])
        _apply_repetition_penalties_impl(logits, prompt, output, penalties)
        torch.testing.assert_close(logits, torch.tensor([[1.0, 1.0, 3.0]]))


class TestUnavailableViews:
    """Error stubs for ops the empty wheel cannot serve."""

    def test_unavailable_cuda_view_message(self):
        """_unavailable_cuda_view raises the documented RuntimeError."""
        with pytest.raises(RuntimeError, match="do not provide a mapped CUDA view"):
            _unavailable_cuda_view(torch.zeros(2))

    def test_unavailable_weak_ref_message(self):
        """_unavailable_weak_ref points at the CUDA build."""
        with pytest.raises(RuntimeError, match="require the vllm_fl._C extension"):
            _unavailable_weak_ref(torch.zeros(2))

    def test_accelerator_view_non_amd_raises(self, monkeypatch):
        """On a non-AMD vendor the dispatcher falls to the unavailable stub."""
        import sys
        import types

        # Stub vllm.platforms so the fallback probe resolves a plain vendor.
        vllm_mod = types.ModuleType("vllm")
        vllm_mod.__path__ = []
        platforms = types.ModuleType("vllm.platforms")
        platforms.current_platform = types.SimpleNamespace(vendor_name="nvidia")
        vllm_mod.platforms = platforms
        monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
        monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
        # DeviceInfo import fails (no flag_gems) → current_platform fallback.
        with pytest.raises(RuntimeError, match="do not provide a mapped CUDA view"):
            _accelerator_view_from_cpu_tensor(torch.zeros(2))


class TestMetaxDetection:
    """_is_metax_backend environment probe."""

    def test_env_var_shortcut(self, monkeypatch):
        """GEMS_VENDOR=metax selects the MetaX path without any import."""
        monkeypatch.setenv("GEMS_VENDOR", "METAX")
        assert _is_metax_backend() is True

    def test_env_var_unset_and_no_mcoplib(self, monkeypatch):
        """Without the env var and without mcoplib the probe is False."""
        monkeypatch.delenv("GEMS_VENDOR", raising=False)
        monkeypatch.setitem(__import__("sys").modules, "vllm.platforms", None)
        assert _is_metax_backend() is False

    def test_resolved_platform_hook_wins(self, monkeypatch):
        """An already-resolved vllm.platforms._current_platform is consulted."""
        import sys
        import types

        monkeypatch.delenv("GEMS_VENDOR", raising=False)

        class FakePlatform:
            def is_metax(self):
                return True

        platforms = types.ModuleType("vllm.platforms")
        platforms._current_platform = FakePlatform()
        vllm_mod = types.ModuleType("vllm")
        vllm_mod.__path__ = []
        vllm_mod.platforms = platforms
        monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
        monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
        assert _is_metax_backend() is True


class TestQueryOpImpls:
    """The CUDA capability probes registered for the _C namespace."""

    def test_cutlass_probes_report_false(self):
        """Every cutlass capability lambda returns False on empty wheels."""
        for name, fn in _QUERY_OP_IMPLS:
            if name.startswith("cutlass"):
                assert fn((9, 0)) is False, name

    def test_weak_ref_entry_present(self):
        """weak_ref_tensor is wired to the unavailable stub."""
        impls = dict(_QUERY_OP_IMPLS)
        assert impls["weak_ref_tensor"] is _unavailable_weak_ref


class TestOptionalNativeOnlyOps:
    """Schema entries deliberately excluded from empty-wheel registration."""

    def test_exclusion_set_names_exist_in_schemas(self):
        """Every excluded op name is defined by a _C schema entry.

        The exclusion set only has an effect where a schema entry carries
        the same name: each member must appear in SCHEMAS' _C namespace,
        so the registration-skip branch (``continue`` in
        ``register_op_schemas``) actually fires for all of them.
        """
        from vllm_fl.ops._C_ops_schemas import SCHEMAS

        defined = set()
        for namespace, schema in SCHEMAS:
            full_name = schema.split("(")[0]
            op_name = full_name.split(".")[0]
            if namespace == "_C":
                defined.add(op_name)
        # Every excluded op is backed by a schema entry with that name.
        assert _OPTIONAL_NATIVE_ONLY_OPS
        assert _OPTIONAL_NATIVE_ONLY_OPS <= defined, (
            sorted(_OPTIONAL_NATIVE_ONLY_OPS - defined)
        )


class TestCOpsPresentProbe:
    """_C_ops_present must key on the probe names, never on the machine.

    The probe result depends on whether the active build (empty wheel vs
    compiled vllm._C / mcoplib._C) already registered ``torch.ops._C``
    entries, so the environment is pinned via ``torch.ops._C`` fakes rather
    than asserted on the host: optional native-only ops must NOT count as
    "present" (the registration skip must not fire for them), while any
    query/fallback probe name must.
    """

    def test_optional_native_only_ops_do_not_count(self, monkeypatch):
        """Optional ops alone leave _C_ops_present False (no skip)."""

        class _FakeOps:
            dsv3_fused_a_gemm = staticmethod(lambda *a, **k: None)
            hadacore_transform = staticmethod(lambda *a, **k: None)

        monkeypatch.setattr(torch.ops, "_C", _FakeOps(), raising=False)
        assert registry._C_ops_present() is False

    def test_query_probe_name_counts_as_present(self, monkeypatch):
        """A cutlass query name flips the probe to True."""

        class _FakeOps:
            cutlass_scaled_mm_supports_fp8 = staticmethod(lambda cap: False)

        monkeypatch.setattr(torch.ops, "_C", _FakeOps(), raising=False)
        assert registry._C_ops_present() is True

    def test_fallback_probe_name_counts_as_present(self, monkeypatch):
        """The CUDA-fallback name flips the probe to True too."""

        class _FakeOps:
            apply_repetition_penalties_ = staticmethod(lambda *a: None)

        monkeypatch.setattr(torch.ops, "_C", _FakeOps(), raising=False)
        assert registry._C_ops_present() is True

    def test_optional_set_members(self):
        """The exclusion set matches the documented six native-only kernels."""
        expected = frozenset(
            {
                "dsv3_fused_a_gemm",
                "fused_gdn_decode_post_conv_mtp",
                "fused_kda_decode",
                "fused_kda_chunk",
                "hadacore_transform",
                "kimi_k3_attn_res",
            }
        )
        assert expected == _OPTIONAL_NATIVE_ONLY_OPS
