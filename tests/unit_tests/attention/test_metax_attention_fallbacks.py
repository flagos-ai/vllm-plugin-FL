# Copyright (c) 2026 BAAI. All rights reserved.
"""Dispatch behavior for MACA images with native ops or schema-only wheels."""

import ast
import importlib.metadata
import itertools
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from packaging.version import InvalidVersion, Version

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl/dispatch/backends/vendor/metax/impl/attention"
)
_NAMESPACES = itertools.count()


def _function(path, name, **namespace):
    namespace.setdefault("_get_flaggems_cache_writer", lambda: None)
    if name == "reshape_and_cache_flash":
        namespace.setdefault(
            "_flaggems_cache_layout_supported",
            _function("utils/fa_utils.py", "_flaggems_cache_layout_supported"),
        )
    namespace.setdefault(
        "__name__",
        "vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils.fa_utils",
    )
    namespace.setdefault(
        "__package__", "vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils"
    )
    tree = ast.parse((SOURCE / path).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(SOURCE / path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    ("dispatch_key", "expected"),
    [
        (None, False),
        ("Meta", False),
        ("CUDA", True),
        ("CompositeImplicitAutograd", True),
        ("CompositeExplicitAutograd", True),
        ("CompositeExplicitAutogradNonFunctional", True),
    ],
)
def test_kernel_detection_distinguishes_schemas_meta_and_executable_ops(
    dispatch_key, expected
):
    probe = _function(
        "ops/_kernel_utils.py", "has_cuda_or_composite_kernel", torch=torch
    )
    namespace = f"metax_attention_dispatch_test_{next(_NAMESPACES)}"
    library = torch.library.Library(namespace, "DEF")
    library.define("probe(Tensor x) -> Tensor")
    if dispatch_key:
        library.impl("probe", lambda x: x, dispatch_key)
    assert probe(f"{namespace}::probe") is expected
    assert probe(f"{namespace}::not_registered") is False


def _stub_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("native_available", [False, True])
def test_cache_writer_preserves_native_kernel_or_uses_stride_aware_triton(
    monkeypatch, native_available
):
    native = Mock()
    fallback = Mock()
    detector = Mock(return_value=native_available)
    _stub_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.impl.attention.ops.reshape_and_cache",
        triton_reshape_and_cache_flash=fallback,
    )
    write = _function(
        "utils/fa_utils.py",
        "reshape_and_cache_flash",
        has_cuda_or_composite_kernel=detector,
        ops=SimpleNamespace(reshape_and_cache_flash=native),
    )
    # Views of the 0.32 packed [blocks, heads, tokens, K+V] layout are
    # non-contiguous. The writer must receive their original strides.
    cache = torch.zeros(2, 3, 16, 16)
    key_cache, value_cache = cache.transpose(1, 2).split(8, dim=-1)
    key = torch.randn(4, 3, 8)
    value = torch.randn_like(key)
    slots = torch.tensor([0, 17, -1])
    scale = torch.tensor(1.0)
    arguments = (key, value, key_cache, value_cache, slots, "auto", scale, scale)
    write(*arguments)
    detector.assert_called_once_with("_C_cache_ops::reshape_and_cache_flash")
    selected, other = (native, fallback) if native_available else (fallback, native)
    selected.assert_called_once()
    other.assert_not_called()
    passed = selected.call_args.args
    assert all(
        actual is expected for actual, expected in zip(passed[:5], arguments[:5])
    )
    assert passed[2].stride() == (768, 16, 256, 1)
    assert passed[3].storage_offset() == 8
    assert passed[5] == "auto" and passed[6] is scale and passed[7] is scale


@pytest.mark.parametrize(
    ("native_available", "head_size", "expect_native"),
    [(False, 8, False), (True, 8, True), (True, 7, False)],
)
def test_merge_preserves_native_kernel_constraints_and_falls_back(
    monkeypatch, native_available, head_size, expect_native
):
    native = Mock()
    fallback = Mock()
    detector = Mock(return_value=native_available)
    _stub_module(monkeypatch, "vllm._custom_ops", merge_attn_states=native)
    _stub_module(
        monkeypatch,
        "vllm.v1.attention.ops.triton_merge_attn_states",
        merge_attn_states=fallback,
    )
    merge = _function(
        "ops/merge_attn_states.py",
        "merge_attn_states",
        torch=torch,
        has_cuda_or_composite_kernel=detector,
    )
    prefix = torch.randn(2, 3, head_size, dtype=torch.bfloat16)
    suffix = torch.randn_like(prefix)
    prefix_lse, suffix_lse, output_lse = (torch.randn(3, 2) for _ in range(3))
    output = torch.empty_like(prefix)
    scale = torch.tensor(1.0)
    arguments = (output, prefix, prefix_lse, suffix, suffix_lse, output_lse, 1, scale)
    merge(*arguments)
    selected, other = (native, fallback) if expect_native else (fallback, native)
    selected.assert_called_once()
    other.assert_not_called()
    assert all(
        actual is expected
        for actual, expected in zip(selected.call_args.args[:6], arguments[:6])
    )
    assert selected.call_args.args[6] == 1 and selected.call_args.args[7] is scale
    if head_size == 8:
        detector.assert_called_once_with("_C::merge_attn_states")


def test_native_cache_execution_errors_are_not_hidden_by_fallback(monkeypatch):
    native = Mock(side_effect=RuntimeError("native launch failed"))
    fallback = Mock()
    _stub_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.impl.attention.ops.reshape_and_cache",
        triton_reshape_and_cache_flash=fallback,
    )
    write = _function(
        "utils/fa_utils.py",
        "reshape_and_cache_flash",
        has_cuda_or_composite_kernel=lambda name: True,
        ops=SimpleNamespace(reshape_and_cache_flash=native),
    )
    with pytest.raises(RuntimeError, match="native launch failed"):
        write(None, None, None, None, None, "auto", None, None)
    fallback.assert_not_called()


def _bridge_resolver(package_version, module=None, error=None):
    bridge = ModuleType("flaggems_vllm") if module is None else module
    if module is None:
        bridge.reshape_and_cache_flash = Mock()
    lookup = Mock(return_value=package_version)
    if error is not None:
        lookup.side_effect = error
    loader = Mock(return_value=bridge)
    resolve = _function(
        "utils/fa_utils.py",
        "_get_flaggems_cache_writer",
        _flaggems_cache_state={"eligible": None, "writer": None},
        version=lookup,
        PackageNotFoundError=importlib.metadata.PackageNotFoundError,
        InvalidVersion=InvalidVersion,
        Version=Version,
        importlib=SimpleNamespace(import_module=loader),
    )
    return resolve, lookup, loader, bridge


@pytest.mark.parametrize(
    ("package_version", "eligible"),
    [
        ("0.1.0", False),
        ("0.2.0", False),
        ("0.3.0.dev0", False),
        ("0.3.0.dev1", True),
        ("0.3.0.dev2", True),
        ("0.3.0", True),
        ("0.3.1+metax", True),
        ("not-a-version", False),
    ],
)
def test_bridge_version_threshold_and_capability_are_cached(package_version, eligible):
    resolve, lookup, loader, bridge = _bridge_resolver(package_version)
    expected = bridge.reshape_and_cache_flash if eligible else None
    assert resolve() is expected and resolve() is expected
    lookup.assert_called_once_with("flaggems-vllm")
    assert loader.call_count == int(eligible)


@pytest.mark.parametrize("export", [None, object()])
def test_bridge_requires_callable_export(export):
    bridge = ModuleType("flaggems_vllm")
    bridge.reshape_and_cache_flash = export
    resolve, lookup, loader, _ = _bridge_resolver("0.3.0.dev1", bridge)
    assert resolve() is None and resolve() is None
    lookup.assert_called_once()
    loader.assert_called_once()


def test_missing_bridge_metadata_is_cached_without_importing_optional_package():
    resolve, lookup, loader, _ = _bridge_resolver(
        None, error=importlib.metadata.PackageNotFoundError("flaggems-vllm")
    )
    assert resolve() is None and resolve() is None
    lookup.assert_called_once()
    loader.assert_not_called()


def test_partially_initialized_bridge_is_retried_without_repeating_version_lookup():
    bridge = ModuleType("flaggems_vllm")
    bridge.__spec__ = SimpleNamespace(_initializing=True)
    bridge.reshape_and_cache_flash = Mock()
    resolve, lookup, loader, _ = _bridge_resolver("0.3.0.dev1", bridge)
    assert resolve() is None
    bridge.__spec__._initializing = False
    assert resolve() is bridge.reshape_and_cache_flash
    assert resolve() is bridge.reshape_and_cache_flash
    lookup.assert_called_once()
    assert loader.call_count == 2


def test_partial_import_error_is_retried_instead_of_caching_unavailable():
    resolve, lookup, loader, bridge = _bridge_resolver("0.3.0.dev1")
    loader.side_effect = [
        ImportError("partially initialized module flaggems_vllm"),
        bridge,
    ]
    assert resolve() is None
    assert resolve() is bridge.reshape_and_cache_flash
    lookup.assert_called_once()


@pytest.mark.parametrize(
    "error", [ImportError("bridge unavailable"), OSError("bad ABI")]
)
def test_permanent_bridge_import_error_is_cached(error):
    resolve, lookup, loader, _ = _bridge_resolver("0.3.0.dev1")
    loader.side_effect = error
    assert resolve() is None and resolve() is None
    lookup.assert_called_once()
    loader.assert_called_once()


@pytest.mark.parametrize("native_available", [False, True])
def test_validated_bridge_precedes_bounded_triton_but_preserves_native(
    monkeypatch, native_available
):
    native, bridge, bounded = Mock(), Mock(), Mock()
    getter = Mock(return_value=bridge)
    _stub_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.impl.attention.ops.reshape_and_cache",
        triton_reshape_and_cache_flash=bounded,
    )
    write = _function(
        "utils/fa_utils.py",
        "reshape_and_cache_flash",
        has_cuda_or_composite_kernel=lambda _: native_available,
        _get_flaggems_cache_writer=getter,
        ops=SimpleNamespace(reshape_and_cache_flash=native),
    )
    cache = torch.zeros(2, 2, 16, 3, 8)
    key_cache, value_cache = cache.unbind(1)
    key = torch.randn(4, 3, 8)
    scale = torch.tensor(1.0)
    arguments = (
        key,
        key.clone(),
        key_cache,
        value_cache,
        torch.arange(4),
        "auto",
        scale,
        scale,
    )
    write(*arguments)
    selected, unused = (native, bridge) if native_available else (bridge, native)
    selected.assert_called_once_with(*arguments)
    unused.assert_not_called()
    bounded.assert_not_called()
    if native_available:
        getter.assert_not_called()
    else:
        getter.assert_called_once()


@pytest.mark.parametrize("layout", ["LBNHC", "LBHNC", "last_dim_gap", "five_dim"])
def test_bridge_is_not_used_for_unsupported_intra_page_strides(monkeypatch, layout):
    native, bridge, bounded = Mock(), Mock(), Mock()
    getter = Mock(return_value=bridge)
    _stub_module(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.metax.impl.attention.ops.reshape_and_cache",
        triton_reshape_and_cache_flash=bounded,
    )
    write = _function(
        "utils/fa_utils.py",
        "reshape_and_cache_flash",
        has_cuda_or_composite_kernel=lambda _: False,
        _get_flaggems_cache_writer=getter,
        ops=SimpleNamespace(reshape_and_cache_flash=native),
    )
    if layout == "LBNHC":
        packed = torch.zeros(2, 16, 3, 16)
        key_cache, value_cache = packed.split(8, -1)
    elif layout == "LBHNC":
        packed = torch.zeros(2, 3, 16, 16).transpose(1, 2)
        key_cache, value_cache = packed.split(8, -1)
    elif layout == "last_dim_gap":
        key_cache = torch.zeros(2, 16, 3, 16)[..., ::2]
        value_cache = torch.empty_like(key_cache)
    else:
        key_cache = torch.zeros(2, 3, 2, 16, 4)
        value_cache = torch.empty_like(key_cache)
    key = torch.randn(4, 3, 8)
    scale = torch.tensor(1.0)
    arguments = (
        key,
        key.clone(),
        key_cache,
        value_cache,
        torch.arange(4),
        "auto",
        scale,
        scale,
    )
    write(*arguments)
    bounded.assert_called_once_with(*arguments)
    getter.assert_not_called()
    bridge.assert_not_called()
    native.assert_not_called()
