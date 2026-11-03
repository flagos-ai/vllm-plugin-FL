# Copyright (c) 2026 BAAI. All rights reserved.

"""MetaX fills vLLM schema gaps while retaining mcoplib's native kernels."""

import gc
import importlib.machinery
import sys
import types
import uuid

import pytest
import torch

from tests.unit_tests.test_op_schema_registry import _load_registry


def _present_mcoplib(monkeypatch):
    module = types.ModuleType("mcoplib")
    module.__spec__ = importlib.machinery.ModuleSpec(
        "mcoplib", loader=None, is_package=True
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_metax_loads_native_ops_before_filling_schema_gaps(monkeypatch):
    suffix = uuid.uuid4().hex
    native_query = "native_query_" + suffix
    missing_query = "missing_query_" + suffix
    native_cuda = "native_cuda_" + suffix
    composite = "composite_" + suffix
    missing_cuda = "missing_cuda_" + suffix
    optional = "optional_" + suffix
    cache = "metax_cache_" + suffix
    schemas = [
        ("_C", f"{native_query}(int capability) -> bool"),
        ("_C", f"{missing_query}(int capability) -> bool"),
        ("_C", f"{native_cuda}(Tensor input) -> Tensor"),
        ("_C", f"{composite}(Tensor input) -> Tensor"),
        ("_C", f"{missing_cuda}(Tensor input) -> Tensor"),
        ("_C", f"{optional}(Tensor input) -> Tensor"),
        (cache, "copy(Tensor input) -> Tensor"),
    ]
    registry = _load_registry(monkeypatch, schemas)
    _present_mcoplib(monkeypatch)
    monkeypatch.setattr(registry, "_is_metax_backend", lambda: True)
    monkeypatch.setattr(
        registry,
        "_QUERY_OP_IMPLS",
        [
            (native_query, lambda capability: False),
            (missing_query, lambda capability: False),
        ],
    )
    monkeypatch.setattr(
        registry,
        "_CUDA_FALLBACK_IMPLS",
        [
            (name, lambda input: input)
            for name in (native_cuda, composite, missing_cuda)
        ],
    )
    monkeypatch.setattr(registry, "_OPTIONAL_NATIVE_ONLY_OPS", {optional})
    native = torch.library.Library("_C", "FRAGMENT")
    calls = []
    native_cuda_table = None

    def load_native(name):
        nonlocal native_cuda_table
        # No vllm._C probe may occur on MetaX, even if it is importable.
        assert name == "mcoplib._C"
        calls.append(name)
        assert not hasattr(torch.ops._C, missing_query)
        native.define(schemas[0][1])
        native.impl(native_query, lambda capability: True, "CompositeImplicitAutograd")
        native.define(schemas[2][1])
        native.impl(native_cuda, lambda input: input + 2, "CUDA")
        native_cuda_table = torch._C._dispatch_dump_table(f"_C::{native_cuda}")
        native.define(schemas[3][1])
        native.impl(composite, lambda input: input + 3, "CompositeImplicitAutograd")

    monkeypatch.setattr(registry.importlib, "import_module", load_native)
    try:
        registry.register_op_schemas()
        gc.collect()
        assert getattr(torch.ops._C, native_query)(90) is True
        assert getattr(torch.ops._C, missing_query)(90) is False
        torch.testing.assert_close(
            getattr(torch.ops._C, composite)(torch.tensor([1.0])),
            torch.tensor([4.0]),
        )
        assert torch._C._dispatch_dump_table(f"_C::{native_cuda}") == native_cuda_table
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(
            f"_C::{composite}", "CUDA"
        )
        assert torch._C._dispatch_has_kernel_for_dispatch_key(
            f"_C::{missing_cuda}", "CUDA"
        )
        assert not hasattr(torch.ops._C, optional)
        assert torch._C._dispatch_find_schema_or_throw(f"{cache}::copy", "")
        libraries = registry.register_op_schemas._libs
        registry.register_op_schemas()
        assert registry.register_op_schemas._libs is libraries
        assert calls == ["mcoplib._C"]
    finally:
        for lib in getattr(registry.register_op_schemas, "_libs", {}).values():
            lib._destroy()
        native._destroy()


def test_failed_mcoplib_import_does_not_claim_vendor_schemas(monkeypatch):
    name = "metax_failed_import_" + uuid.uuid4().hex
    registry = _load_registry(monkeypatch, [("_C", f"{name}(Tensor input) -> Tensor")])
    _present_mcoplib(monkeypatch)
    monkeypatch.setattr(registry, "_is_metax_backend", lambda: True)

    def fail_native(module_name):
        assert module_name == "mcoplib._C"
        raise ImportError("vendor extension is unavailable")

    monkeypatch.setattr(registry.importlib, "import_module", fail_native)
    with pytest.raises(ImportError, match="vendor extension is unavailable"):
        registry.register_op_schemas()
    assert not hasattr(torch.ops._C, name)
    assert not hasattr(registry.register_op_schemas, "_libs")


def test_metax_without_mcoplib_registers_flagos_fallbacks(monkeypatch):
    name = "metax_flagos_query_" + uuid.uuid4().hex
    registry = _load_registry(monkeypatch, [("_C", f"{name}(int capability) -> bool")])
    monkeypatch.setattr(registry, "_is_metax_backend", lambda: True)
    monkeypatch.setattr(registry, "_QUERY_OP_IMPLS", [(name, lambda capability: False)])
    monkeypatch.setattr(registry, "_CUDA_FALLBACK_IMPLS", [])

    def unexpected_import(module_name):
        raise AssertionError(f"No native extension probe is allowed: {module_name}")

    monkeypatch.setattr(registry.importlib, "import_module", unexpected_import)
    try:
        registry.register_op_schemas()
        assert getattr(torch.ops._C, name)(80) is False
    finally:
        for lib in getattr(registry.register_op_schemas, "_libs", {}).values():
            lib._destroy()
