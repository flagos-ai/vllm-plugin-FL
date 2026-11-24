# Copyright (c) 2026 BAAI. All rights reserved.

"""Extra branch coverage for ``vllm_fl.dispatch.discovery``.

Focuses on the plugin source paths not covered by the existing
``test_discovery.py``: entry-point discovery variants, successful
registration, generic import errors and aggregate counting.
"""

import sys
import types

import pytest

from vllm_fl.dispatch import discovery as d


@pytest.fixture(autouse=True)
def _clear_plugins():
    d.clear_discovered_plugins()
    yield
    d.clear_discovered_plugins()


class _SelectEPs:
    def __init__(self, items):
        self._items = items
        self.calls = []

    def select(self, group=None):
        self.calls.append(group)
        return iter(self._items)


class _GetEPs:
    def __init__(self, mapping):
        self._mapping = mapping

    def get(self, group, default=None):
        return self._mapping.get(group, default)


class _PlainEPs:
    pass


class TestGetEntryPoints:
    def test_select_style(self, monkeypatch):
        import importlib.metadata as md

        eps = _SelectEPs(["a"])
        monkeypatch.setattr(md, "entry_points", lambda: eps)
        assert d._get_entry_points() == ["a"]
        assert eps.calls == [d.PLUGIN_GROUP]

    def test_dict_style(self, monkeypatch):
        import importlib.metadata as md

        monkeypatch.setattr(
            md, "entry_points", lambda: {d.PLUGIN_GROUP: ["b"]}
        )
        assert d._get_entry_points() == ["b"]

    def test_get_style_non_dict(self, monkeypatch):
        import importlib.metadata as md

        monkeypatch.setattr(
            md, "entry_points", lambda: _GetEPs({d.PLUGIN_GROUP: ["c"]})
        )
        assert d._get_entry_points() == ["c"]

    def test_get_style_missing_group(self, monkeypatch):
        import importlib.metadata as md

        monkeypatch.setattr(md, "entry_points", lambda: _GetEPs({}))
        assert d._get_entry_points() == []

    def test_unsupported_object(self, monkeypatch):
        import importlib.metadata as md

        monkeypatch.setattr(md, "entry_points", lambda: _PlainEPs())
        assert d._get_entry_points() == []

    def test_entry_points_raises(self, monkeypatch):
        import importlib.metadata as md

        def _boom():
            raise RuntimeError("metadata broken")

        monkeypatch.setattr(md, "entry_points", _boom)
        assert d._get_entry_points() == []

    def test_importlib_metadata_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "importlib.metadata", None)
        monkeypatch.setitem(sys.modules, "importlib_metadata", None)
        assert d._get_entry_points() == []


class _EP:
    def __init__(self, name, obj=None, raises=None):
        self.name = name
        self._obj = obj
        self._raises = raises

    def load(self):
        if self._raises is not None:
            raise self._raises
        return self._obj


class TestDiscoverFromEntryPoints:
    def test_no_entry_points(self, monkeypatch):
        monkeypatch.setattr(d, "_get_entry_points", lambda: [])
        assert d.discover_from_entry_points(object()) == 0

    def test_successful_callable(self, monkeypatch):
        seen = []
        ep = _EP("my_plugin", obj=lambda registry: seen.append(registry))
        monkeypatch.setattr(d, "_get_entry_points", lambda: [ep])
        registry = object()
        assert d.discover_from_entry_points(registry) == 1
        assert seen == [registry]
        assert d.get_discovered_plugins() == [("my_plugin", "entry_point", True)]

    def test_register_returns_no_handler(self, monkeypatch):
        plugin = types.SimpleNamespace()  # not callable, no register hooks
        ep = _EP("noop", obj=plugin)
        monkeypatch.setattr(d, "_get_entry_points", lambda: [ep])
        assert d.discover_from_entry_points(object()) == 0
        assert d.get_discovered_plugins() == [("noop", "entry_point", False)]

    def test_load_raises(self, monkeypatch):
        ep = _EP("broken", raises=RuntimeError("cannot load"))
        monkeypatch.setattr(d, "_get_entry_points", lambda: [ep])
        assert d.discover_from_entry_points(object()) == 0
        assert d.get_discovered_plugins() == [("broken", "entry_point", False)]

    def test_ep_without_name_attr(self, monkeypatch):
        class _BareEP:
            def load(self):
                return lambda registry: None

        monkeypatch.setattr(d, "_get_entry_points", lambda: [_BareEP()])
        assert d.discover_from_entry_points(object()) == 1
        name = d.get_discovered_plugins()[0][0]
        assert "object" in name or "BareEP" in name


class TestDiscoverFromEnvModules:
    def test_success_via_register_attr(self, monkeypatch):
        module = types.ModuleType("fake_plugin_mod")
        seen = []
        module.register = lambda registry: seen.append(registry)
        monkeypatch.setitem(sys.modules, "fake_plugin_mod", module)
        monkeypatch.setenv(d.PLUGIN_MODULES_ENV, " fake_plugin_mod ")
        registry = object()
        assert d.discover_from_env_modules(registry) == 1
        assert seen == [registry]
        assert d.get_discovered_plugins() == [
            ("fake_plugin_mod", "env_module", True)
        ]

    def test_multiple_modules_mixed(self, monkeypatch):
        good = types.ModuleType("good_plugin")
        good.vllm_fl_register = lambda registry: None
        monkeypatch.setitem(sys.modules, "good_plugin", good)

        nohook = types.ModuleType("nohook_plugin")
        monkeypatch.setitem(sys.modules, "nohook_plugin", nohook)

        monkeypatch.setenv(d.PLUGIN_MODULES_ENV, "good_plugin, nohook_plugin")
        assert d.discover_from_env_modules(object()) == 1
        plugins = d.get_discovered_plugins()
        assert ("good_plugin", "env_module", True) in plugins
        assert ("nohook_plugin", "env_module", False) in plugins

    def test_generic_exception_is_caught(self, monkeypatch):
        def _boom(name, *args, **kwargs):
            raise RuntimeError("import exploded")

        monkeypatch.setattr(d.importlib, "import_module", _boom)
        monkeypatch.setenv(d.PLUGIN_MODULES_ENV, "some_module")
        assert d.discover_from_env_modules(object()) == 0
        assert d.get_discovered_plugins() == [
            ("some_module", "env_module", False)
        ]

    def test_blank_entries_ignored(self, monkeypatch):
        def _boom(name, *args, **kwargs):
            raise AssertionError("should not import blank names")

        monkeypatch.setattr(d.importlib, "import_module", _boom)
        monkeypatch.setenv(d.PLUGIN_MODULES_ENV, " , , ")
        assert d.discover_from_env_modules(object()) == 0


class TestDiscoverPluginsAggregate:
    def test_sums_both_sources(self, monkeypatch):
        monkeypatch.setattr(d, "discover_from_entry_points", lambda registry: 2)
        monkeypatch.setattr(d, "discover_from_env_modules", lambda registry: 3)
        assert d.discover_plugins(object()) == 5

    def test_none_registry_skips_sources(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            d, "discover_from_entry_points", lambda registry: called.append(1)
        )
        monkeypatch.setattr(
            d, "discover_from_env_modules", lambda registry: called.append(1)
        )
        assert d.discover_plugins(None) == 0
        assert called == []


def test_clear_discovered_plugins():
    d._discovered_plugins.append(("x", "entry_point", True))
    d.clear_discovered_plugins()
    assert d.get_discovered_plugins() == []
