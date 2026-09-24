# Copyright (c) 2026 BAAI. All rights reserved.

"""Capability and idempotence tests for the packed-GDN precision patch."""

from types import SimpleNamespace

import pytest

# The patch imports vLLM's Triton compatibility layer.  Keep this test
# collection-friendly on source-only/CPU developer machines; the actual test
# runs in the vLLM/FlagOS image used for Day0 validation.
pytest.importorskip("vllm")

from vllm_fl.patches import gdn_packed_decode  # noqa: E402


def _vulnerable_kernel():
    # Keep the expression in a function body so inspect.getsource exercises
    # the same path used for a real Triton JIT function.
    beta_val = "tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"
    return beta_val


def _fixed_kernel():
    beta_val = "tl.sigmoid(b_val)"
    return beta_val


def _non_sigmoid_kernel():
    beta_val = "b_val"
    return beta_val


def _target(kernel):
    return SimpleNamespace(
        fused_recurrent_gated_delta_rule_packed_decode_kernel=kernel
    )


def test_kernel_detection_matches_only_legacy_sigmoid_cast():
    assert gdn_packed_decode._kernel_needs_beta_patch(_vulnerable_kernel)
    assert not gdn_packed_decode._kernel_needs_beta_patch(_fixed_kernel)
    assert not gdn_packed_decode._kernel_needs_beta_patch(_non_sigmoid_kernel)


def test_uninspectable_source_prefers_the_fix():
    # vLLM 0.24.0 is known to need the fix, so a stripped wheel must not keep
    # the rounded recurrent update just because the source is unavailable.
    assert gdn_packed_decode._kernel_needs_beta_patch(len)


def test_patch_replaces_vulnerable_kernel_and_is_idempotent(monkeypatch):
    target = _target(_vulnerable_kernel)
    monkeypatch.setattr(
        gdn_packed_decode.importlib, "import_module", lambda _module: target
    )

    assert gdn_packed_decode.patch_vllm_packed_gdn_beta() is True
    replacement = target.fused_recurrent_gated_delta_rule_packed_decode_kernel
    assert replacement is not _vulnerable_kernel
    assert replacement._fl_fp32_beta is True

    # A second registration must not replace the already-installed kernel.
    assert gdn_packed_decode.patch_vllm_packed_gdn_beta() is False
    assert target.fused_recurrent_gated_delta_rule_packed_decode_kernel is replacement


@pytest.mark.parametrize("kernel", [_fixed_kernel, _non_sigmoid_kernel])
def test_patch_preserves_non_vulnerable_kernel(monkeypatch, kernel):
    target = _target(kernel)
    monkeypatch.setattr(
        gdn_packed_decode.importlib, "import_module", lambda _module: target
    )

    assert gdn_packed_decode.patch_vllm_packed_gdn_beta() is False
    assert target.fused_recurrent_gated_delta_rule_packed_decode_kernel is kernel


def test_patch_is_optional_when_fla_module_or_symbol_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        gdn_packed_decode.importlib,
        "import_module",
        lambda _module: SimpleNamespace(),
    )
    assert gdn_packed_decode.patch_vllm_packed_gdn_beta() is False

    def missing_module(_module):
        raise ModuleNotFoundError("FLA is not present in this vLLM build")

    monkeypatch.setattr(gdn_packed_decode.importlib, "import_module", missing_module)
    assert gdn_packed_decode.patch_vllm_packed_gdn_beta() is False


def test_registration_helper_reports_the_patch_result(monkeypatch):
    import vllm_fl

    target = _target(_vulnerable_kernel)
    real_import = gdn_packed_decode.importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == gdn_packed_decode._TARGET_MODULE:
            return target
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(gdn_packed_decode.importlib, "import_module", fake_import)
    assert vllm_fl._register_gdn_packed_decode_patch() is True
