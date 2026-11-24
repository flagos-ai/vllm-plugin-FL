# Copyright (c) 2026 BAAI. All rights reserved.
"""Residual coverage for ``discovery`` plugin failures and registry edges."""

import sys
import types

import pytest

from vllm_fl.dispatch import discovery as d
from vllm_fl.dispatch.registry import OpRegistry, OpRegistrySnapshot
from vllm_fl.dispatch.types import BackendImplKind, OpImpl


@pytest.fixture(autouse=True)
def _clear_plugins():
    d.clear_discovered_plugins()
    yield
    d.clear_discovered_plugins()


class _EP:
    def __init__(self, name, obj):
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


def test_direct_callable_raises_is_isolated(monkeypatch):
    def boom(registry):
        raise RuntimeError("plugin body failed")

    monkeypatch.setattr(d, "_get_entry_points", lambda: [_EP("bad_callable", boom)])
    assert d.discover_from_entry_points(object()) == 0
    assert d.get_discovered_plugins() == [("bad_callable", "entry_point", False)]


def test_register_function_raises_is_isolated(monkeypatch):
    def boom(registry):
        raise ValueError("register failed")

    plugin = types.SimpleNamespace(vllm_fl_register=boom)
    monkeypatch.setattr(d, "_get_entry_points", lambda: [_EP("bad_register", plugin)])
    assert d.discover_from_entry_points(object()) == 0
    assert d.get_discovered_plugins() == [("bad_register", "entry_point", False)]


def test_none_vllm_fl_register_falls_back_to_register(monkeypatch):
    seen = []
    plugin = types.SimpleNamespace(
        vllm_fl_register=None,
        register=lambda registry: seen.append(registry),
    )
    monkeypatch.setattr(d, "_get_entry_points", lambda: [_EP("fallback_attr", plugin)])
    registry = object()
    assert d.discover_from_entry_points(registry) == 1
    assert seen == [registry]


def test_env_module_import_error_is_recorded(monkeypatch):
    def _boom(name, *args, **kwargs):
        raise ImportError("module missing")

    monkeypatch.setattr(d.importlib, "import_module", _boom)
    monkeypatch.setenv(d.PLUGIN_MODULES_ENV, "missing_plugin")
    assert d.discover_from_env_modules(object()) == 0
    assert d.get_discovered_plugins() == [
        ("missing_plugin", "env_module", False)
    ]


def test_env_module_register_raises_is_isolated(monkeypatch):
    module = types.ModuleType("w27_raising_plugin")

    def boom(registry):
        raise RuntimeError("register exploded")

    module.register = boom
    monkeypatch.setitem(sys.modules, "w27_raising_plugin", module)
    monkeypatch.setenv(d.PLUGIN_MODULES_ENV, "w27_raising_plugin")
    assert d.discover_from_env_modules(object()) == 0
    assert d.get_discovered_plugins() == [
        ("w27_raising_plugin", "env_module", False)
    ]


def test_registry_register_many_empty_and_snapshot_survives_clear():
    reg = OpRegistry()
    reg.register_many([])
    assert reg.list_operators() == []

    impl = OpImpl(
        op_name="silu",
        impl_id="default.impl",
        kind=BackendImplKind.DEFAULT,
        fn=lambda x: x,
    )
    reg.register_impl(impl)
    snapshot = reg.snapshot()
    assert isinstance(snapshot, OpRegistrySnapshot)

    reg.clear()
    assert snapshot.impls_by_op["silu"] == [impl]
    assert reg.get_implementations("silu") == []
