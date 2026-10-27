# Copyright (c) 2026 BAAI. All rights reserved.

"""Schema fallback registration preserves namespaces and native ownership."""

import gc
import importlib.util
import sys
import types
import uuid
from pathlib import Path

import pytest
import torch


def _load_registry(monkeypatch, schemas, *, native=False):
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(
        sys.modules, "vllm._C", types.ModuleType("vllm._C") if native else None
    )
    monkeypatch.setitem(sys.modules, "mcoplib", None)
    schema_module = types.ModuleType("vllm_fl.ops._C_ops_schemas")
    schema_module.SCHEMAS = schemas
    monkeypatch.setitem(sys.modules, schema_module.__name__, schema_module)
    path = (
        Path(__file__).resolve().parents[2] / "vllm_fl" / "ops" / "_C_ops_registry.py"
    )
    spec = importlib.util.spec_from_file_location("op_schema_registry_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registers_same_name_and_overloads_in_separate_namespaces(monkeypatch):
    # Unique namespaces keep this test independent of native schemas already
    # loaded by other tests, without requiring a GPU or the vLLM extension.
    prefix = "schema_test_" + uuid.uuid4().hex
    regular, cache = prefix, prefix + "_cache"
    schemas = [
        (regular, "copy(Tensor input) -> Tensor"),
        (cache, "copy(Tensor input, Tensor! cache) -> ()"),
        (cache, "copy.out(Tensor input, Tensor! out) -> ()"),
        # The optional-native exclusion applies only to the _C namespace.
        (cache, "hadacore_transform(Tensor input) -> Tensor"),
    ]
    existing = torch.library.Library(regular, "FRAGMENT")
    existing.define(schemas[0][1])
    registry = _load_registry(monkeypatch, schemas)
    try:
        registry.register_op_schemas()
        libraries = registry.register_op_schemas._libs
        assert set(libraries) == {regular, cache}
        gc.collect()
        for namespace, schema in schemas:
            full_name = schema.split("(")[0]
            name, _, overload = full_name.partition(".")
            registered = torch._C._dispatch_find_schema_or_throw(
                f"{namespace}::{name}", overload
            ).schema()
            assert str(registered).startswith(f"{namespace}::{full_name}(")
        assert not hasattr(getattr(torch.ops, regular).copy, "out")
        registry.register_op_schemas()
        assert registry.register_op_schemas._libs is libraries
    finally:
        for lib in getattr(registry.register_op_schemas, "_libs", {}).values():
            lib._destroy()
        existing._destroy()


def test_native_extension_skips_all_fallback_registration(monkeypatch):
    registry = _load_registry(
        monkeypatch, [("_C_cache_ops", "invalid schema")], native=True
    )
    registry.register_op_schemas()
    assert not hasattr(registry.register_op_schemas, "_libs")


def test_optional_native_exclusion_is_limited_to_c_namespace(monkeypatch):
    name = "optional_test_" + uuid.uuid4().hex
    cache = "schema_test_cache_" + uuid.uuid4().hex
    schema = f"{name}(Tensor input) -> Tensor"
    registry = _load_registry(monkeypatch, [("_C", schema), (cache, schema)])
    monkeypatch.setattr(
        registry,
        "_OPTIONAL_NATIVE_ONLY_OPS",
        registry._OPTIONAL_NATIVE_ONLY_OPS | {name},
    )
    try:
        registry.register_op_schemas()
        with pytest.raises(RuntimeError):
            torch._C._dispatch_find_schema_or_throw(f"_C::{name}", "")
        registered = torch._C._dispatch_find_schema_or_throw(
            f"{cache}::{name}", ""
        ).schema()
        assert registered.name == f"{cache}::{name}"
    finally:
        for lib in getattr(registry.register_op_schemas, "_libs", {}).values():
            lib._destroy()
