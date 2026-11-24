# Copyright (c) 2026 BAAI. All rights reserved.
"""Coverage tests for the packed-GDN and QSA patches."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_fl.patches import gdn_packed_decode as gdn
from vllm_fl.patches import qsa_cache_hint as qsa


# ---------------------------------------------------------------------------
# gdn_packed_decode
# ---------------------------------------------------------------------------
def _vulnerable_kernel():
    return "tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"


def _fixed_kernel():
    return "tl.sigmoid(b_val)"


def test_kernel_needs_beta_patch_detection():
    marked = SimpleNamespace(_fl_fp32_beta=True)
    assert gdn._kernel_needs_beta_patch(marked) is False
    assert gdn._kernel_needs_beta_patch(_vulnerable_kernel) is True
    assert gdn._kernel_needs_beta_patch(_fixed_kernel) is False
    # inspection unavailable -> prefer the corrected kernel
    assert gdn._kernel_needs_beta_patch(object()) is True
    # a jit wrapper exposes ``.fn``; its source is inspected instead
    assert gdn._kernel_needs_beta_patch(
        SimpleNamespace(fn=_vulnerable_kernel)) is True


def test_patch_applies_and_is_idempotent(monkeypatch):
    target = SimpleNamespace(
        fused_recurrent_gated_delta_rule_packed_decode_kernel=_vulnerable_kernel)
    monkeypatch.setattr(gdn.importlib, "import_module", lambda name: target)
    assert gdn.patch_vllm_packed_gdn_beta() is True
    assert target.fused_recurrent_gated_delta_rule_packed_decode_kernel is (
        gdn._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta)
    assert gdn.patch_vllm_packed_gdn_beta() is False


def test_patch_skips_fixed_and_missing(monkeypatch):
    fixed = SimpleNamespace(
        fused_recurrent_gated_delta_rule_packed_decode_kernel=_fixed_kernel)
    monkeypatch.setattr(gdn.importlib, "import_module", lambda name: fixed)
    assert gdn.patch_vllm_packed_gdn_beta() is False

    monkeypatch.setattr(gdn.importlib, "import_module", lambda name: (
        _ for _ in ()).throw(ImportError("no fla")))
    assert gdn.patch_vllm_packed_gdn_beta() is False

    monkeypatch.setattr(gdn.importlib, "import_module",
                        lambda name: SimpleNamespace())
    assert gdn.patch_vllm_packed_gdn_beta() is False


# ---------------------------------------------------------------------------
# qsa_cache_hint: AST rewrite
# ---------------------------------------------------------------------------
_VALID_SRC = (
    "def k(k_cache_ptr, off):\n"
    '    keys = tl.load(k_cache_ptr + off, eviction_policy="evict_first")\n'
    "    return keys\n"
)


def test_without_hint_rewrites_valid_source():
    kernel = SimpleNamespace(src=_VALID_SRC)
    rewritten = qsa._without_k_cache_eviction_hint(kernel)
    assert rewritten is not None
    assert "eviction_policy" not in rewritten
    assert "k_cache_ptr" in rewritten


def test_without_hint_returns_none_when_already_patched():
    src = (
        "def k(k_cache_ptr, off):\n"
        "    keys = tl.load(k_cache_ptr + off)\n"
        "    return keys\n"
    )
    assert qsa._without_k_cache_eviction_hint(SimpleNamespace(src=src)) is None


def test_without_hint_rejects_changed_loads():
    no_load = "def k():\n    return 1\n"
    with pytest.raises(RuntimeError, match="K-cache load changed"):
        qsa._without_k_cache_eviction_hint(SimpleNamespace(src=no_load))

    two_loads = (
        "def k(k_cache_ptr, off):\n"
        '    keys = tl.load(k_cache_ptr + off, eviction_policy="evict_first")\n'
        '    keys = tl.load(k_cache_ptr + off, eviction_policy="evict_first")\n'
        "    return keys\n"
    )
    with pytest.raises(RuntimeError, match="K-cache load changed"):
        qsa._without_k_cache_eviction_hint(SimpleNamespace(src=two_loads))

    other_policy = (
        "def k(k_cache_ptr, off):\n"
        '    keys = tl.load(k_cache_ptr + off, eviction_policy="evict_last")\n'
        "    return keys\n"
    )
    with pytest.raises(RuntimeError, match="eviction policy changed"):
        qsa._without_k_cache_eviction_hint(SimpleNamespace(src=other_policy))


# ---------------------------------------------------------------------------
# qsa_cache_hint: patch entry point
# ---------------------------------------------------------------------------
class _Kernel:
    def __init__(self, src, device_caches=None):
        self.src = src
        self.device_caches = list(device_caches or [])
        self.updated = None

    def _unsafe_update_src(self, src):
        self.updated = src


def _model(arch="Qwen4ExpForCausalLM"):
    return SimpleNamespace(architecture=arch)


def _fake_torch(cap=(9, 0), hip=None):
    return SimpleNamespace(
        version=SimpleNamespace(hip=hip),
        cuda=SimpleNamespace(get_device_capability=lambda device: cap),
    )


def _install_qsa_module(monkeypatch, kernels):
    leaf = "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer"
    parts = leaf.split(".")
    parent = None
    for i in range(2, len(parts) + 1):
        name = ".".join(parts[:i])
        mod = ModuleType(name)
        mod.__path__ = []
        if parent is not None:
            setattr(parent, parts[i - 1], mod)
        monkeypatch.setitem(sys.modules, name, mod)
        parent = mod
    for name, kernel in kernels.items():
        setattr(parent, name, kernel)
    return parent


def test_patch_qsa_early_returns(monkeypatch):
    assert qsa.patch_qsa_cache_hint(_model("OtherArch"), SimpleNamespace(
        type="cuda")) is False

    monkeypatch.setitem(sys.modules, "triton",
                        SimpleNamespace(__version__="3.5.0"))
    assert qsa.patch_qsa_cache_hint(_model(), SimpleNamespace(type="cuda")) is False

    monkeypatch.setitem(sys.modules, "triton",
                        SimpleNamespace(__version__="3.6.0"))

    def missing(_name):
        raise qsa.PackageNotFoundError("no flagtree")

    monkeypatch.setattr(qsa, "version", missing)
    assert qsa.patch_qsa_cache_hint(_model(), SimpleNamespace(type="cuda")) is False

    monkeypatch.setattr(qsa, "version", lambda name: "1.0")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    # not cuda
    assert qsa.patch_qsa_cache_hint(_model(), SimpleNamespace(type="cpu")) is False
    # hip
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(hip="6.0"))
    assert qsa.patch_qsa_cache_hint(_model(), SimpleNamespace(type="cuda")) is False
    # wrong capability
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cap=(8, 0)))
    assert qsa.patch_qsa_cache_hint(_model(), SimpleNamespace(type="cuda")) is False


def test_patch_qsa_success(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton",
                        SimpleNamespace(__version__="3.6.0"))
    monkeypatch.setattr(qsa, "version", lambda name: "1.0")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    kernels = {name: _Kernel(_VALID_SRC) for name in qsa._KERNEL_NAMES}
    _install_qsa_module(monkeypatch, kernels)

    assert qsa.patch_qsa_cache_hint(
        _model(), SimpleNamespace(type="cuda")) is True
    for kernel in kernels.values():
        assert kernel._fl_qsa_original_src == _VALID_SRC
        assert "eviction_policy" not in kernel.updated


def test_patch_qsa_rejects_compiled_kernel(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton",
                        SimpleNamespace(__version__="3.6.0"))
    monkeypatch.setattr(qsa, "version", lambda name: "1.0")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    kernels = {name: _Kernel(_VALID_SRC) for name in qsa._KERNEL_NAMES}
    kernels[qsa._KERNEL_NAMES[0]].device_caches = ["compiled"]
    _install_qsa_module(monkeypatch, kernels)
    with pytest.raises(RuntimeError, match="already compiled"):
        qsa.patch_qsa_cache_hint(_model(), SimpleNamespace(type="cuda"))


def test_patch_qsa_noop_when_already_patched(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton",
                        SimpleNamespace(__version__="3.6.0"))
    monkeypatch.setattr(qsa, "version", lambda name: "1.0")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    patched_src = (
        "def k(k_cache_ptr, off):\n"
        "    keys = tl.load(k_cache_ptr + off)\n"
        "    return keys\n"
    )
    kernels = {name: _Kernel(patched_src) for name in qsa._KERNEL_NAMES}
    _install_qsa_module(monkeypatch, kernels)
    assert qsa.patch_qsa_cache_hint(
        _model(), SimpleNamespace(type="cuda")) is False
    assert all(k.updated is None for k in kernels.values())
