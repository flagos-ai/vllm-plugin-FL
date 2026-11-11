# Copyright (c) 2026 BAAI. All rights reserved.

"""Regression coverage for the vLLM 0.32 empty-build adaptation.

These guard the fixes raised in review of the 0.32 adaptation:
- the MetaX IR-op path must never probe/load ``vllm._C`` (a second extension
  would duplicate-register and abort);
- ``_force_uva_unavailable`` must rebind consumers that already imported
  ``is_uva_available`` by value, not just the canonical module.
"""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")


def _vllm_config():
    from vllm.config.compilation import CompilationMode

    return SimpleNamespace(
        compilation_config=SimpleNamespace(backend="eager", mode=CompilationMode.NONE)
    )


def _capture_default(monkeypatch):
    import vllm.config.kernel as kernel

    captured = {}

    def fake_with_default(default, **kwargs):
        captured["default"] = default
        return default

    monkeypatch.setattr(
        kernel.IrOpPriorityConfig,
        "with_default",
        staticmethod(fake_with_default),
    )
    return captured


def test_metax_ir_priority_never_probes_vllm_c(monkeypatch):
    import vllm_fl.ops._C_ops_registry as reg
    import vllm_fl.platform as platform_mod

    monkeypatch.setattr(reg, "_is_metax_backend", lambda: True)
    captured = _capture_default(monkeypatch)

    # On MetaX neither find_spec nor import_module may be consulted for vllm._C.
    def guard_find_spec(name, *args, **kwargs):
        assert name != "vllm._C", "MetaX path probed vllm._C via find_spec"
        return None

    def guard_import(name, *args, **kwargs):
        assert name != "vllm._C", "MetaX path loaded vllm._C"
        raise AssertionError("unreachable")

    monkeypatch.setattr(importlib.util, "find_spec", guard_find_spec)
    monkeypatch.setattr(importlib, "import_module", guard_import)

    platform_mod.PlatformFL.get_default_ir_op_priority(_vllm_config())
    assert captured["default"] == ["native"]


def test_non_metax_offers_vllm_c_only_when_present(monkeypatch):
    import vllm_fl.ops._C_ops_registry as reg
    import vllm_fl.platform as platform_mod

    monkeypatch.setattr(reg, "_is_metax_backend", lambda: False)

    # Extension present -> vllm_c offered; absent -> native only.
    captured = _capture_default(monkeypatch)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: object())
    platform_mod.PlatformFL.get_default_ir_op_priority(_vllm_config())
    assert captured["default"] == ["vllm_c", "native"]

    captured = _capture_default(monkeypatch)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    platform_mod.PlatformFL.get_default_ir_op_priority(_vllm_config())
    assert captured["default"] == ["native"]


def test_force_uva_rebinds_already_imported_consumer(monkeypatch):
    import vllm.utils.platform_utils as pu

    import vllm_fl.ops._C_ops_registry as reg

    def real_uva() -> bool:
        return True

    # A consumer that captured the name by value before the patch, like 0.32's
    # offloader/model modules do at import.
    consumer = types.ModuleType("fake_uva_consumer")
    consumer.is_uva_available = real_uva
    monkeypatch.setitem(sys.modules, "fake_uva_consumer", consumer)
    monkeypatch.setattr(pu, "is_uva_available", real_uva)

    reg._force_uva_unavailable()

    assert pu.is_uva_available() is False
    assert consumer.is_uva_available() is False
    assert consumer.is_uva_available is pu.is_uva_available
