# Copyright (c) 2026 BAAI. All rights reserved.
"""Extra coverage for vllm_fl/ops/_C_ops_registry.py edge branches."""

import importlib.util
import sys
import uuid
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_fl.ops import _C_ops_registry as registry


def _patch_guards(monkeypatch, schemas):
    monkeypatch.setattr(registry, "_is_metax_backend", lambda: False)
    monkeypatch.setattr(registry, "_C_ops_present", lambda: False)
    monkeypatch.setitem(sys.modules, "vllm._C", None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    schema_module = ModuleType("vllm_fl.ops._C_ops_schemas")
    schema_module.SCHEMAS = schemas
    monkeypatch.setitem(sys.modules, "vllm_fl.ops._C_ops_schemas", schema_module)
    monkeypatch.delattr(registry.register_op_schemas, "_libs", raising=False)


def _destroy_libs():
    for lib in getattr(registry.register_op_schemas, "_libs", {}).values():
        lib._destroy()
    if hasattr(registry.register_op_schemas, "_libs"):
        del registry.register_op_schemas._libs


def test_register_handles_overload_names(monkeypatch):
    uid = uuid.uuid4().hex
    namespace = f"fl_cov_ov_{uid}"
    schema = "foo.bar(Tensor input) -> Tensor"
    _patch_guards(monkeypatch, [(namespace, schema)])
    try:
        registry.register_op_schemas()
        found = torch._C._dispatch_find_schema_or_throw(
            f"{namespace}::foo", "bar"
        )
        assert str(found.schema()).startswith(f"{namespace}::foo.bar(")
        # Second run must see the overload and skip re-definition.
        first = registry.register_op_schemas._libs
        registry.register_op_schemas()
        assert registry.register_op_schemas._libs is first
    finally:
        _destroy_libs()


def test_register_wraps_define_errors(monkeypatch):
    uid = uuid.uuid4().hex
    namespace = f"fl_cov_bad_{uid}"
    _patch_guards(monkeypatch, [(namespace, f"broken_{uid}")])
    try:
        with pytest.raises(RuntimeError, match="Failed to register"):
            registry.register_op_schemas()
    finally:
        _destroy_libs()


def test_register_preloads_mcoplib_when_available(monkeypatch):
    _patch_guards(monkeypatch, [])
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "mcoplib" else None,
    )
    package = ModuleType("mcoplib")
    package.__path__ = []
    extension = ModuleType("mcoplib._C")
    monkeypatch.setitem(sys.modules, "mcoplib", package)
    monkeypatch.setitem(sys.modules, "mcoplib._C", extension)
    try:
        registry.register_op_schemas()
        assert not hasattr(registry.register_op_schemas, "_libs")
    finally:
        _destroy_libs()


def test_register_tolerates_broken_mcoplib_import(monkeypatch):
    _patch_guards(monkeypatch, [])
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "mcoplib" else None,
    )
    for name in ("mcoplib", "mcoplib._C"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        registry.register_op_schemas()
        assert not hasattr(registry.register_op_schemas, "_libs")
    finally:
        _destroy_libs()


def test_is_metax_backend_ignores_platform_without_hook(monkeypatch):
    class Plain:
        pass

    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        SimpleNamespace(_current_platform=Plain()),
    )
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert registry._is_metax_backend() is False


def test_accelerator_view_handles_missing_platform_module(monkeypatch):
    import vllm_fl.utils as fl_utils

    def broken():
        raise RuntimeError("no device info")

    monkeypatch.setattr(fl_utils, "DeviceInfo", broken)
    monkeypatch.setitem(sys.modules, "vllm.platforms", None)
    with pytest.raises(RuntimeError, match="mapped CUDA view"):
        registry._accelerator_view_from_cpu_tensor(torch.zeros(1))


def test_c_ops_present_handles_missing_namespace(monkeypatch):
    monkeypatch.setattr(registry, "torch", SimpleNamespace(ops=SimpleNamespace()))
    assert registry._C_ops_present() is False
