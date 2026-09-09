# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import vllm_fl
from vllm_fl.patches import gdn_packed_decode


def _vulnerable_kernel():
    beta_val = "tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"
    return beta_val


def _fixed_kernel():
    beta_val = "tl.sigmoid(b_val)"
    return beta_val


def test_patch_replaces_vulnerable_kernel(monkeypatch):
    target = SimpleNamespace(
        fused_recurrent_gated_delta_rule_packed_decode_kernel=_vulnerable_kernel
    )
    monkeypatch.setattr(
        gdn_packed_decode.importlib, "import_module", lambda _module: target
    )

    assert gdn_packed_decode.patch_vllm_packed_gdn_beta() is True
    replacement = target.fused_recurrent_gated_delta_rule_packed_decode_kernel
    assert replacement is not _vulnerable_kernel
    assert replacement._fl_fp32_beta is True
    assert gdn_packed_decode.patch_vllm_packed_gdn_beta() is False


def test_patch_preserves_already_fixed_upstream_kernel(monkeypatch):
    target = SimpleNamespace(
        fused_recurrent_gated_delta_rule_packed_decode_kernel=_fixed_kernel
    )
    monkeypatch.setattr(
        gdn_packed_decode.importlib, "import_module", lambda _module: target
    )

    assert gdn_packed_decode.patch_vllm_packed_gdn_beta() is False
    assert target.fused_recurrent_gated_delta_rule_packed_decode_kernel is _fixed_kernel


def test_patch_is_optional_when_symbol_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        gdn_packed_decode.importlib,
        "import_module",
        lambda _module: SimpleNamespace(),
    )

    assert gdn_packed_decode.patch_vllm_packed_gdn_beta() is False


def test_registration_is_optional_when_gdn_module_is_unavailable(monkeypatch):
    def missing_module(_module):
        raise ModuleNotFoundError("vendor vLLM image does not provide FLA")

    monkeypatch.setattr(vllm_fl.importlib, "import_module", missing_module)

    assert vllm_fl._register_gdn_packed_decode_patch() is False


def test_registration_is_not_vendor_gated(monkeypatch):
    calls = []
    patch_module = SimpleNamespace(
        patch_vllm_packed_gdn_beta=lambda: calls.append("patched") or True
    )
    monkeypatch.setattr(
        vllm_fl.importlib,
        "import_module",
        lambda module: patch_module,
    )

    assert vllm_fl._register_gdn_packed_decode_patch() is True
    assert calls == ["patched"]
