# Copyright (c) 2026 BAAI. All rights reserved.
"""Residual branch coverage for the remaining patches.

Targets not already exercised by w6b/w9b:
  * qsa_cache_hint: one kernel already patched, the other still needs a rewrite
  * gdn_packed_decode: source inspection unavailable (OSError) and marked kernel
  * worker_symbols_032: device-side D2H carrier and packing layout assertion
  * fla_module_relocation: absent ops subpackage, pre-bound ``layers.fla``, and
    an old alias entry that already exists in ``sys.modules``
  * flagtree_target_info: predicate when the live target cannot be read
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_fl.patches import (
    fla_module_relocation as fla,
    flagtree_target_info as fti,
    gdn_packed_decode as gdn,
    qsa_cache_hint as qsa,
    worker_symbols_032 as ws,
)

# ---------------------------------------------------------------------------
# qsa_cache_hint: mixed patched / unpatched kernels
# ---------------------------------------------------------------------------
_QSA_WITH_HINT = (
    "def k(k_cache_ptr, off):\n"
    '    keys = tl.load(k_cache_ptr + off, eviction_policy="evict_first")\n'
    "    return keys\n"
)
_QSA_ALREADY_PATCHED = (
    "def k(k_cache_ptr, off):\n"
    "    keys = tl.load(k_cache_ptr + off)\n"
    "    return keys\n"
)


class _QsaKernel:
    def __init__(self, src, device_caches=None):
        self.src = src
        self.device_caches = list(device_caches or [])
        self.updated = None

    def _unsafe_update_src(self, src):
        self.updated = src


def _install_qsa_module(monkeypatch, kernels):
    leaf = "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer"
    parts = leaf.split(".")
    parent = None
    for index in range(2, len(parts) + 1):
        name = ".".join(parts[:index])
        module = ModuleType(name)
        module.__path__ = []
        if parent is not None:
            setattr(parent, parts[index - 1], module)
        monkeypatch.setitem(sys.modules, name, module)
        parent = module
    for name, kernel in kernels.items():
        setattr(parent, name, kernel)
    return parent


def test_qsa_patches_only_kernel_that_still_has_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="3.6.0"))
    monkeypatch.setattr(qsa, "version", lambda name: "1.0")
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(get_device_capability=lambda device: (9, 0)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    needed, done = qsa._KERNEL_NAMES
    kernels = {needed: _QsaKernel(_QSA_WITH_HINT), done: _QsaKernel(_QSA_ALREADY_PATCHED)}
    _install_qsa_module(monkeypatch, kernels)

    model = SimpleNamespace(architecture="Qwen4ExpForConditionalGeneration")
    assert qsa.patch_qsa_cache_hint(model, SimpleNamespace(type="cuda")) is True
    assert kernels[needed].updated is not None
    assert "eviction_policy" not in kernels[needed].updated
    # The already-patched kernel is skipped (source is None -> continue).
    assert kernels[done].updated is None


# ---------------------------------------------------------------------------
# gdn_packed_decode: inspection branches
# ---------------------------------------------------------------------------
def test_kernel_needs_patch_when_source_unavailable(monkeypatch):
    def unavailable(obj):
        raise OSError("stripped wheel")

    monkeypatch.setattr(gdn.inspect, "getsource", unavailable)
    # Prefer the corrected kernel when the source cannot be inspected.
    assert gdn._kernel_needs_beta_patch(object()) is True


def test_marked_kernel_skips_source_inspection(monkeypatch):
    def boom(obj):
        raise AssertionError("must not inspect a marked kernel")

    monkeypatch.setattr(gdn.inspect, "getsource", boom)
    assert gdn._kernel_needs_beta_patch(SimpleNamespace(_fl_fp32_beta=True)) is False


# ---------------------------------------------------------------------------
# worker_symbols_032: device-side carrier + packing assertion
# ---------------------------------------------------------------------------
class _FakeDeviceTensor:
    def __init__(self, device_type):
        self.device = SimpleNamespace(type=device_type)
        self.copied = None

    def to(self, target, non_blocking=False):
        self.copied = (target, non_blocking)
        return ("cpu-tensor", id(self))


def test_routed_experts_to_cpu_nonblocking_from_device():
    routing = _FakeDeviceTensor("npu")
    slots = _FakeDeviceTensor("npu")
    tensors = ws._RoutedExpertsTensors(routing, slots)

    cpu = tensors.to_cpu_nonblocking()
    assert cpu is not tensors
    assert isinstance(cpu, ws._RoutedExpertsTensors)
    assert routing.copied == ("cpu", True)
    assert slots.copied == ("cpu", True)


def test_reshape_packing_requires_block_first_layout():
    raw = torch.zeros(64, dtype=torch.uint8)
    spec = SimpleNamespace(dtype=torch.float32)
    # Stride order (1, 0, 2) makes inv_order[0] != 0 -> assertion.
    with pytest.raises(AssertionError):
        ws._reshape_attention_kv_cache(
            raw,
            spec,
            (2, 3, 4),
            (1, 0, 2),
            num_blocks=2,
            packing=(0, 16),
        )


# ---------------------------------------------------------------------------
# fla_module_relocation: residual install branches
# ---------------------------------------------------------------------------
def test_install_skips_absent_operators_subpackage(monkeypatch):
    new_prefix = fla._NEW_PREFIX
    new_pkg = ModuleType(new_prefix)
    new_pkg.__path__ = []
    layers = ModuleType("vllm.model_executor.layers")
    modules = {new_prefix: new_pkg, "vllm.model_executor.layers": layers}

    monkeypatch.setattr(fla, "_module_exists", lambda name: name == new_prefix)
    monkeypatch.setattr(fla.importlib, "import_module", lambda name: modules[name])

    def _no_iter(path):
        raise AssertionError("iter_modules must not run for an absent subpackage")

    monkeypatch.setattr(fla.pkgutil, "iter_modules", _no_iter)

    try:
        assert fla.install() is True
        assert sys.modules[fla._OLD_PREFIX] is new_pkg
        assert layers.fla is new_pkg
    finally:
        sys.modules.pop(fla._OLD_PREFIX, None)


def test_install_keeps_existing_layers_fla_attribute(monkeypatch):
    new_prefix = fla._NEW_PREFIX
    new_pkg = ModuleType(new_prefix)
    new_pkg.__path__ = []
    layers = ModuleType("vllm.model_executor.layers")
    layers.fla = "sentinel"
    modules = {new_prefix: new_pkg, "vllm.model_executor.layers": layers}

    monkeypatch.setattr(fla, "_module_exists", lambda name: name == new_prefix)
    monkeypatch.setattr(fla.importlib, "import_module", lambda name: modules[name])

    try:
        assert fla.install() is True
        assert sys.modules[fla._OLD_PREFIX] is new_pkg
        assert layers.fla == "sentinel"
    finally:
        sys.modules.pop(fla._OLD_PREFIX, None)


def test_install_does_not_overwrite_existing_old_alias(monkeypatch):
    new_prefix = fla._NEW_PREFIX
    old_prefix = fla._OLD_PREFIX
    new_pkg = ModuleType(new_prefix)
    new_pkg.__path__ = []
    layers = ModuleType("vllm.model_executor.layers")
    modules = {new_prefix: new_pkg, "vllm.model_executor.layers": layers}

    monkeypatch.setattr(fla, "_module_exists", lambda name: name == new_prefix)
    monkeypatch.setattr(fla.importlib, "import_module", lambda name: modules[name])
    existing = ModuleType(old_prefix)
    monkeypatch.setitem(sys.modules, old_prefix, existing)

    assert fla.install() is True
    # The pre-existing alias keeps precedence (added count stays zero).
    assert sys.modules[old_prefix] is existing
    assert layers.fla is new_pkg


# ---------------------------------------------------------------------------
# flagtree_target_info: predicate with unreadable target
# ---------------------------------------------------------------------------
def test_predicate_false_when_live_target_unavailable():
    def no_target():
        raise RuntimeError("no current target")

    target_info = SimpleNamespace(is_hip=lambda: True, current_target=no_target)
    predicate = fti._make_predicate(target_info, "is_hip_unknown", ("gfx942",))
    assert predicate() is False
