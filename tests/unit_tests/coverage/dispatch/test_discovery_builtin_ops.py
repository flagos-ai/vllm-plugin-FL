# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary coverage tests for plugin discovery and builtin registration.

Covers ground not exercised by the other supplementary dispatch tests:

- ``vllm_fl.dispatch.discovery``: every entry-point enumeration style
  (select / dict / get fallback), import-error fallbacks, the plugin
  register-function calling convention, and the entry-point / env-module
  discovery flows.
- ``vllm_fl.dispatch.builtin_ops``: vendor backend directory resolution,
  current-platform vendor detection, vendor backend auto-registration, and
  the ``register_builtins`` orchestration order.
- ``vllm_fl.dispatch.config.utils``: platform name detection, NVIDIA Hopper
  arch-config selection, and config path aliasing.
"""

from __future__ import annotations

import sys
import types

import pytest

import vllm_fl.dispatch.builtin_ops as builtin_ops
import vllm_fl.dispatch.config.utils as config_utils
import vllm_fl.dispatch.discovery as discovery


@pytest.fixture(autouse=True)
def _reset_discovered_plugins():
    """Keep the module-level discovered-plugin list isolated per test."""
    discovery.clear_discovered_plugins()
    yield
    discovery.clear_discovered_plugins()


def _make_vllm_platforms(monkeypatch, current_platform=None):
    """Install a fake ``vllm.platforms`` module into ``sys.modules``.

    When *current_platform* is None the attribute is omitted entirely so that
    ``from vllm.platforms import current_platform`` raises ImportError.
    """
    stub = types.ModuleType("vllm.platforms")
    if current_platform is not None:
        stub.current_platform = current_platform
    monkeypatch.setitem(sys.modules, "vllm.platforms", stub)


class FakeEntryPoint:
    """Minimal entry-point double exposing name and load()."""

    def __init__(self, name, obj):
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


class FakeEntryPointError:
    """Entry point whose load() raises, exercising the failure path."""

    def __init__(self, name, exc):
        self.name = name
        self._exc = exc

    def load(self):
        raise self._exc


class SelectableEps:
    """Entry-point container mimicking the Python 3.10+ SelectableGroups."""

    def __init__(self, groups):
        self._groups = groups

    def select(self, group):
        return self._groups.get(group, [])


class GettableEps:
    """Entry-point container with only a get() method (legacy fallback)."""

    def __init__(self, groups):
        self._groups = groups

    def get(self, key, default=None):
        return self._groups.get(key, default)


class TestGetEntryPoints:
    """_get_entry_points across container shapes and import failures."""

    def test_select_style_returns_group(self, monkeypatch):
        """Python 3.10+ selectable containers yield the plugin group list."""
        ep = FakeEntryPoint("alpha", None)
        monkeypatch.setattr(
            discovery.importlib.metadata,
            "entry_points",
            lambda: SelectableEps({discovery.PLUGIN_GROUP: [ep]}),
        )
        assert discovery._get_entry_points() == [ep]

    def test_dict_style_returns_group(self, monkeypatch):
        """Plain-dict containers return the group's entry-point list."""
        ep = FakeEntryPoint("beta", None)
        monkeypatch.setattr(
            discovery.importlib.metadata,
            "entry_points",
            lambda: {discovery.PLUGIN_GROUP: [ep]},
        )
        assert discovery._get_entry_points() == [ep]

    def test_get_fallback_returns_group(self, monkeypatch):
        """Containers with only get() fall back to dict-like access."""
        ep = FakeEntryPoint("gamma", None)
        monkeypatch.setattr(
            discovery.importlib.metadata,
            "entry_points",
            lambda: GettableEps({discovery.PLUGIN_GROUP: [ep]}),
        )
        assert discovery._get_entry_points() == [ep]

    def test_unsupported_container_returns_empty(self, monkeypatch):
        """Containers without select/dict/get behaviour yield an empty list."""

        class Odd:
            pass

        monkeypatch.setattr(discovery.importlib.metadata, "entry_points", lambda: Odd())
        assert discovery._get_entry_points() == []

    def test_entry_points_error_returns_empty(self, monkeypatch):
        """Exceptions raised while enumerating entry points degrade to []."""

        def boom():
            raise RuntimeError("metadata exploded")

        monkeypatch.setattr(discovery.importlib.metadata, "entry_points", boom)
        assert discovery._get_entry_points() == []


class TestCallRegisterFunction:
    """_call_register_function for each plugin object convention."""

    def _registry(self):
        class Registry:
            def __init__(self):
                self.calls = []

        return Registry()

    def test_classes_skip_direct_call_and_use_register(self):
        """Classes are not called directly; register is looked up instead."""

        class Plugin:
            @staticmethod
            def vllm_fl_register(reg):
                reg.calls.append("class-register")

        reg = self._registry()
        assert discovery._call_register_function(Plugin, reg, "src") is True
        assert reg.calls == ["class-register"]

    def test_falls_back_to_register_attribute(self):
        """vllm_fl_register takes precedence over the register attribute."""
        obj = types.SimpleNamespace()
        obj.vllm_fl_register = lambda reg: reg.calls.append("preferred")
        obj.register = lambda reg: reg.calls.append("fallback")
        reg = self._registry()
        assert discovery._call_register_function(obj, reg, "src") is True
        assert reg.calls == ["preferred"]

    def test_register_attribute_used_when_canonical_missing(self):
        """The plain ``register`` attribute is used as a fallback hook."""
        obj = types.SimpleNamespace()
        obj.register = lambda reg: reg.calls.append("fallback")
        reg = self._registry()
        assert discovery._call_register_function(obj, reg, "src") is True
        assert reg.calls == ["fallback"]


class TestDiscoverFromEntryPoints:
    """discover_from_entry_points bookkeeping."""

    def test_success_and_failure_are_tracked(self, monkeypatch):
        """Successful and failing entry points are counted and recorded."""

        def good(reg):
            reg.append("good")

        reg = []
        eps = [
            FakeEntryPoint("good", good),
            FakeEntryPoint("obj", types.SimpleNamespace()),
            FakeEntryPointError("bad", ImportError("missing")),
        ]
        monkeypatch.setattr(discovery, "_get_entry_points", lambda: eps)
        loaded = discovery.discover_from_entry_points(reg)
        assert loaded == 1
        assert discovery.get_discovered_plugins() == [
            ("good", "entry_point", True),
            ("obj", "entry_point", False),
            ("bad", "entry_point", False),
        ]

    def test_empty_entry_points_short_circuits(self, monkeypatch):
        """No entry points means zero loads and no bookkeeping."""
        monkeypatch.setattr(discovery, "_get_entry_points", lambda: [])
        assert discovery.discover_from_entry_points(object()) == 0
        assert discovery.get_discovered_plugins() == []


class TestDiscoverFromEnvModules:
    """discover_from_env_modules via the VLLM_FL_PLUGIN_MODULES variable."""

    def _plugin_mod(self, monkeypatch, name, registered):
        mod = types.ModuleType(name)
        mod.register = lambda reg: registered.append(name)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    def test_env_modules_load_in_order(self, monkeypatch):
        """Comma-separated modules are imported and registered in order."""
        registered = []
        self._plugin_mod(monkeypatch, "supp_plugin_a", registered)
        self._plugin_mod(monkeypatch, "supp_plugin_b", registered)
        monkeypatch.setenv(
            discovery.PLUGIN_MODULES_ENV, " supp_plugin_a , supp_plugin_b "
        )
        assert discovery.discover_from_env_modules(object()) == 2
        assert registered == ["supp_plugin_a", "supp_plugin_b"]
        assert discovery.get_discovered_plugins() == [
            ("supp_plugin_a", "env_module", True),
            ("supp_plugin_b", "env_module", True),
        ]

    def test_unknown_module_records_failure(self, monkeypatch):
        """Unimportable module names are recorded as failed plugins."""
        monkeypatch.setenv(discovery.PLUGIN_MODULES_ENV, "supp_no_such_plugin_xyz")
        assert discovery.discover_from_env_modules(object()) == 0
        assert discovery.get_discovered_plugins() == [
            ("supp_no_such_plugin_xyz", "env_module", False)
        ]

    def test_module_without_register_records_failure(self, monkeypatch):
        """A module lacking any register hook counts as a failure."""
        monkeypatch.setitem(
            sys.modules, "supp_silent_plugin", types.ModuleType("supp_silent")
        )
        monkeypatch.setenv(discovery.PLUGIN_MODULES_ENV, "supp_silent_plugin")
        assert discovery.discover_from_env_modules(object()) == 0
        assert discovery.get_discovered_plugins() == [
            ("supp_silent_plugin", "env_module", False)
        ]

    def test_blank_env_var_is_a_no_op(self, monkeypatch):
        """An unset or whitespace-only env var loads nothing."""
        monkeypatch.setenv(discovery.PLUGIN_MODULES_ENV, "   ")
        assert discovery.discover_from_env_modules(object()) == 0
        assert discovery.get_discovered_plugins() == []

    def test_discover_plugins_aggregates_sources(self, monkeypatch):
        """discover_plugins sums entry-point and env-module loads."""
        reg = []
        monkeypatch.setattr(
            discovery,
            "_get_entry_points",
            lambda: [FakeEntryPoint("ep", lambda r: r.append("ep"))],
        )
        monkeypatch.setenv(discovery.PLUGIN_MODULES_ENV, "")
        assert discovery.discover_plugins(reg) == 1
        assert reg == ["ep"]
        # Registry of None short-circuits to zero loads.
        assert discovery.discover_plugins(None) == 0


class TestFindVendorBackendDir:
    """_find_vendor_backend_dir resolution rules."""

    def test_vendor_name_preferred_over_device_fields(self):
        """vendor_name wins over device_name / device_type candidates."""
        assert (
            builtin_ops._find_vendor_backend_dir("nvidia", {"nvidia", "cuda"})
            == "nvidia"
        )

    def test_device_name_and_type_fallbacks(self):
        """When the vendor dir is absent, device_name then type are used."""
        # metax: device_name "metax" absent here, device_type "cuda" present.
        assert builtin_ops._find_vendor_backend_dir("metax", {"cuda"}) == "cuda"
        # ascend: device_name "npu" present.
        assert builtin_ops._find_vendor_backend_dir("ascend", {"npu"}) == "npu"

    def test_no_candidate_dirs_returns_none(self):
        """None is returned when no candidate directory exists."""
        assert builtin_ops._find_vendor_backend_dir("nvidia", {"unrelated"}) is None


class TestGetCurrentVendorBackendDirs:
    """_get_current_vendor_backend_dirs platform probing."""

    def test_platform_with_vendor_name(self, monkeypatch):
        """A string vendor_name on current_platform resolves to its dir."""

        class Platform:
            vendor_name = "nvidia"

        _make_vllm_platforms(monkeypatch, Platform())
        assert builtin_ops._get_current_vendor_backend_dirs({"nvidia"}) == "nvidia"

    def test_non_string_vendor_name_returns_none(self, monkeypatch):
        """A missing / non-string vendor_name yields None (no crash)."""

        class Platform:
            vendor_name = None

        _make_vllm_platforms(monkeypatch, Platform())
        assert builtin_ops._get_current_vendor_backend_dirs({"nvidia"}) is None

    def test_platform_import_failure_raises_runtime_error(self, monkeypatch):
        """Import errors from vllm.platforms surface as RuntimeError."""
        _make_vllm_platforms(monkeypatch, None)
        with pytest.raises(RuntimeError, match="current_platform"):
            builtin_ops._get_current_vendor_backend_dirs({"nvidia"})


class TestRegisterVendorBackends:
    """_register_vendor_backends filesystem scanning and import."""

    def _scan_with_dirs(self, monkeypatch, tmp_path, dirs):
        """Point _VENDOR_BACKENDS_DIR at tmp_path with *dirs* created."""
        for name, files in dirs.items():
            d = tmp_path / name
            d.mkdir()
            for f in files:
                (d / f).write_text("", encoding="utf-8")

        monkeypatch.setattr(builtin_ops, "_VENDOR_BACKENDS_DIR", str(tmp_path))
        return tmp_path

    def test_missing_dir_returns_early(self, monkeypatch, tmp_path):
        """A missing vendor directory logs and returns without error."""
        monkeypatch.setattr(
            builtin_ops, "_VENDOR_BACKENDS_DIR", str(tmp_path / "absent")
        )
        builtin_ops._register_vendor_backends(object())

    def test_undetectable_vendor_skips_registration(self, monkeypatch, tmp_path):
        """No resolvable current vendor means nothing is imported."""
        self._scan_with_dirs(monkeypatch, tmp_path, {"nvidia": ["register_ops.py"]})
        imported = []

        def fake_import(name, package=None):
            imported.append(name)
            raise AssertionError("should not import")

        monkeypatch.setattr(
            "vllm_fl.dispatch.builtin_ops.importlib.import_module", fake_import
        )
        monkeypatch.setattr(
            builtin_ops, "_get_current_vendor_backend_dirs", lambda dirs: None
        )
        builtin_ops._register_vendor_backends(object())
        assert imported == []

    def test_matching_vendor_registers_ops(self, monkeypatch, tmp_path):
        """The current vendor's register_ops.register_builtins is invoked."""
        self._scan_with_dirs(monkeypatch, tmp_path, {"nvidia": ["register_ops.py"]})

        class Platform:
            vendor_name = "nvidia"

        _make_vllm_platforms(monkeypatch, Platform())
        calls = []

        def fake_import(name, package=None):
            assert name == ".backends.vendor.nvidia.register_ops"
            assert package == "vllm_fl.dispatch"
            return types.SimpleNamespace(
                register_builtins=lambda reg: calls.append(reg)
            )

        monkeypatch.setattr(
            "vllm_fl.dispatch.builtin_ops.importlib.import_module", fake_import
        )
        sentinel = object()
        builtin_ops._register_vendor_backends(sentinel)
        assert calls == [sentinel]

    def test_module_without_register_builtins_is_skipped(self, monkeypatch, tmp_path):
        """A register_ops module lacking register_builtins is tolerated."""
        self._scan_with_dirs(monkeypatch, tmp_path, {"nvidia": ["register_ops.py"]})

        class Platform:
            vendor_name = "nvidia"

        _make_vllm_platforms(monkeypatch, Platform())
        monkeypatch.setattr(
            "vllm_fl.dispatch.builtin_ops.importlib.import_module",
            lambda name, package=None: types.SimpleNamespace(),
        )
        builtin_ops._register_vendor_backends(object())

    def test_import_failure_is_swallowed(self, monkeypatch, tmp_path):
        """An import error inside vendor registration is logged, not raised."""
        self._scan_with_dirs(monkeypatch, tmp_path, {"nvidia": ["register_ops.py"]})

        class Platform:
            vendor_name = "nvidia"

        _make_vllm_platforms(monkeypatch, Platform())

        def boom(name, package=None):
            raise ImportError("vendor gone")

        monkeypatch.setattr(
            "vllm_fl.dispatch.builtin_ops.importlib.import_module", boom
        )
        builtin_ops._register_vendor_backends(object())

    def test_dir_without_register_ops_is_skipped(self, monkeypatch, tmp_path):
        """Vendor directories lacking register_ops.py are ignored."""
        self._scan_with_dirs(monkeypatch, tmp_path, {"nvidia": []})

        class Platform:
            vendor_name = "nvidia"

        _make_vllm_platforms(monkeypatch, Platform())
        imported = []

        def fake_import(name, package=None):
            imported.append(name)
            return types.SimpleNamespace()

        monkeypatch.setattr(
            "vllm_fl.dispatch.builtin_ops.importlib.import_module", fake_import
        )
        builtin_ops._register_vendor_backends(object())
        assert imported == []


class TestRegisterBuiltins:
    """register_builtins orchestration over the backend sources."""

    def test_order_of_backend_registration(self, monkeypatch):
        """FlagGems and Reference register before vendor and plugin stages."""
        order = []
        monkeypatch.setitem(
            sys.modules,
            "vllm_fl.dispatch.backends.flaggems.register_ops",
            types.SimpleNamespace(
                register_builtins=lambda reg: order.append("flaggems")
            ),
        )
        monkeypatch.setitem(
            sys.modules,
            "vllm_fl.dispatch.backends.reference.register_ops",
            types.SimpleNamespace(
                register_builtins=lambda reg: order.append("reference")
            ),
        )
        monkeypatch.setattr(
            builtin_ops, "_register_vendor_backends", lambda reg: order.append("vendor")
        )
        monkeypatch.setattr(
            discovery,
            "discover_plugins",
            lambda reg: order.append("plugins") or 1,
        )
        builtin_ops.register_builtins(object())
        assert order == ["flaggems", "reference", "vendor", "plugins"]

    def test_flaggems_failure_still_registers_reference(self, monkeypatch):
        """A FlagGems registration failure does not block later stages."""
        order = []

        def broken(reg):
            raise RuntimeError("fg down")

        monkeypatch.setitem(
            sys.modules,
            "vllm_fl.dispatch.backends.flaggems.register_ops",
            types.SimpleNamespace(register_builtins=broken),
        )
        monkeypatch.setitem(
            sys.modules,
            "vllm_fl.dispatch.backends.reference.register_ops",
            types.SimpleNamespace(
                register_builtins=lambda reg: order.append("reference")
            ),
        )
        monkeypatch.setattr(builtin_ops, "_register_vendor_backends", lambda reg: None)
        monkeypatch.setattr(discovery, "discover_plugins", lambda reg: 0)
        builtin_ops.register_builtins(object())
        assert order == ["reference"]


class TestConfigUtilsPlatform:
    """config.utils platform detection and arch-config selection."""

    def test_platform_name_from_current_platform(self, monkeypatch):
        """vendor_name from vllm.platforms is returned verbatim."""

        class Platform:
            vendor_name = "ascend"

        _make_vllm_platforms(monkeypatch, Platform())
        assert config_utils.get_platform_name() == "ascend"

    def test_platform_name_env_override_used_on_import_failure(self, monkeypatch):
        """Without vllm.platforms the VLLM_FL_PLATFORM override applies."""
        _make_vllm_platforms(monkeypatch, None)
        monkeypatch.setenv("VLLM_FL_PLATFORM", "  CUDA ")
        assert config_utils.get_platform_name() == "cuda"

    def test_platform_name_unknown_when_no_signal(self, monkeypatch):
        """Missing platform module and env var default to 'unknown'."""
        _make_vllm_platforms(monkeypatch, None)
        monkeypatch.delenv("VLLM_FL_PLATFORM", raising=False)
        assert config_utils.get_platform_name() == "unknown"

    def test_arch_config_requires_nvidia(self, monkeypatch):
        """Non-nvidia platforms never select an architecture config."""
        monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "1")
        assert config_utils._get_arch_config_name("ascend") is None

    def test_arch_config_disabled_by_default(self, monkeypatch):
        """The Hopper opt-in env var defaults to off."""
        monkeypatch.delenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", raising=False)
        assert config_utils._get_arch_config_name("nvidia") is None

    def test_arch_config_hopper_when_capability_9(self, monkeypatch):
        """Major capability 9 with the opt-in selects nvidia_hopper."""

        class Capability:
            major = 9

        class Platform:
            @staticmethod
            def get_device_capability():
                return Capability()

        _make_vllm_platforms(monkeypatch, Platform())
        monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "true")
        assert config_utils._get_arch_config_name("nvidia") == "nvidia_hopper"

    def test_arch_config_capability_exception_returns_none(self, monkeypatch):
        """Device-discovery failures degrade to the vendor-wide config."""

        class Platform:
            @staticmethod
            def get_device_capability():
                raise RuntimeError("no device")

        _make_vllm_platforms(monkeypatch, Platform())
        monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "1")
        assert config_utils._get_arch_config_name("nvidia") is None

    def test_arch_config_non_hopper_capability(self, monkeypatch):
        """Major capability other than 9 keeps the plain nvidia config."""

        class Capability:
            major = 8

        class Platform:
            @staticmethod
            def get_device_capability():
                return Capability()

        _make_vllm_platforms(monkeypatch, Platform())
        monkeypatch.setenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", "1")
        assert config_utils._get_arch_config_name("nvidia") is None

    def test_config_path_aliases_mthreads_and_gcu(self):
        """mthreads maps to musa and gcu maps to enflame in config paths."""
        musa = config_utils.get_config_path("mthreads")
        assert musa is not None and musa.name == "musa.yaml"
        enflame = config_utils.get_config_path("gcu")
        assert enflame is not None and enflame.name == "enflame.yaml"

    def test_config_path_unknown_platform_returns_none(self):
        """Platforms without a config file resolve to None."""
        assert config_utils.get_config_path("does_not_exist_xyz") is None
