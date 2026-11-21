# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for remaining ``vllm_fl.ops._C_ops_registry`` branches.

The core suite (``tests/unit_tests/test_op_schema_registry.py``) drives the
fresh-loaded module through the happy-path registration, the native-extension
skip and the optional-op exclusion.  The supplementary helpers file covers the
repetition-penalty oracle and the non-AMD view stub.  Still uncovered:

* ``_accelerator_view_from_cpu_tensor`` AMD delegation (with the
  ``_log_amd_host_view_once`` one-shot log) and its double-fallback vendor
  probe when both DeviceInfo and current_platform raise;
* ``_is_metax_backend`` resolved-platform hook raising, missing hook, and the
  ``find_spec`` mcoplib probe (present / raising);
* ``register_op_schemas`` mcoplib preload branch (successful import and
  ImportError warning), the empty-SCHEMAS warning, and the schema-define
  RuntimeError chain.

The fresh-load + unique-namespace pattern follows the core suite's
``_load_registry``; the module under test is loaded through
``spec_from_file_location`` with stubbed ``sys.modules`` entries so real
extensions and a possibly-registered ``torch.ops._C`` never interfere.
"""

import gc
import importlib.util
import logging
import sys
import types
import uuid
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.supplementary]

_REGISTRY_PATH = (
    Path(__file__).resolve().parents[4] / "vllm_fl" / "ops" / "_C_ops_registry.py"
)


def _make_module(name: str, **attrs):
    module = types.ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_registry(monkeypatch, schemas, *, native=False):
    """Load a fresh registry copy against stubbed vllm/mcoplib/schema modules.

    Mirrors the core suite's loader: ``vllm._C`` is importable when *native*
    else a None entry (ImportError); ``mcoplib`` is absent by default.
    """
    vllm = _make_module("vllm")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(
        sys.modules, "vllm._C", types.ModuleType("vllm._C") if native else None
    )
    monkeypatch.setitem(sys.modules, "mcoplib", None)
    schema_module = _make_module("vllm_fl.ops._C_ops_schemas", SCHEMAS=schemas)
    monkeypatch.setitem(sys.modules, schema_module.__name__, schema_module)
    spec = importlib.util.spec_from_file_location(
        "supp_c_ops_registry_" + uuid.uuid4().hex[:8], _REGISTRY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_C_ops_present", lambda: False)
    monkeypatch.setattr(module, "_is_metax_backend", lambda: False)
    return module


def _destroy(libs):
    """Destroy torch Library objects created by a registration run."""
    for lib in (libs or {}).values():
        lib._destroy()


class TestAcceleratorViewAmd:
    """``_accelerator_view_from_cpu_tensor`` AMD path and vendor probe."""

    def _install_vllm_platforms(self, monkeypatch, current_platform=None):
        vllm = _make_module("vllm")
        platforms = _make_module("vllm.platforms")
        if current_platform is not None:
            platforms.current_platform = current_platform
        vllm.platforms = platforms
        monkeypatch.setitem(sys.modules, "vllm", vllm)
        monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    def test_amd_delegates_to_vendor_view(self, monkeypatch, caplog):
        """The AMD vendor resolves to the ROCm zero-copy host mapping.

        The vendor is pinned through the ``vllm_fl.utils.DeviceInfo`` seam —
        the same attribute the unavailable-view tests below patch — because
        with flag_gems installed the real DeviceInfo answers with the host
        vendor (``nvidia`` on the CUDA CI container) instead of raising, and
        the call would fall through to the unavailable-view stub. The
        ``current_platform`` stub keeps steering the fallback probe.
        """
        calls = []

        def fake_amd_view(tensor):
            calls.append(tensor)
            return "amd-view"

        amd_mod = _make_module(
            "vllm_fl.dispatch.backends.vendor.amd.impl.uva",
            get_cuda_view_from_cpu_tensor_amd=fake_amd_view,
        )
        saved = sys.modules.get(amd_mod.__name__)
        sys.modules[amd_mod.__name__] = amd_mod
        import vllm_fl.utils as utils

        class _AmdDeviceInfo:
            def __init__(self):
                self.vendor_name = "amd"

        monkeypatch.setattr(utils, "DeviceInfo", _AmdDeviceInfo)
        self._install_vllm_platforms(
            monkeypatch, types.SimpleNamespace(vendor_name="amd")
        )
        try:
            import vllm_fl.ops._C_ops_registry as registry

            monkeypatch.setattr(registry, "_log_amd_host_view_once", lambda: None)
            tensor = torch.zeros(2)
            with caplog.at_level(logging.INFO, logger=registry.logger.name):
                result = registry._accelerator_view_from_cpu_tensor(tensor)
        finally:
            if saved is None:
                sys.modules.pop(amd_mod.__name__, None)
            else:
                sys.modules[amd_mod.__name__] = saved
        assert result == "amd-view"
        assert calls == [tensor]

    def test_amd_logs_once_message(self, monkeypatch):
        """The one-shot INFO log names the ROCm host mapping."""
        import vllm_fl.ops._C_ops_registry as registry

        registry._log_amd_host_view_once.cache_clear()
        records = []

        class _Handler(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Handler()
        logger = logging.getLogger(registry.__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            registry._log_amd_host_view_once()
            registry._log_amd_host_view_once()  # second call is cached
        finally:
            logger.removeHandler(handler)
        assert records == [
            (
                "Using the AMD ROCm zero-copy host mapping for "
                "_C::get_cuda_view_from_cpu_tensor."
            )
        ]

    def test_both_probes_failing_raises_unavailable(self, monkeypatch):
        """When DeviceInfo and current_platform both raise, the stub raises."""
        import vllm_fl.utils as utils

        class _ExplodingDeviceInfo:
            def __init__(self):
                raise RuntimeError("flag_gems missing")

        monkeypatch.setattr(utils, "DeviceInfo", _ExplodingDeviceInfo)
        # current_platform import fails: vllm.platforms is a None entry.
        monkeypatch.setitem(sys.modules, "vllm.platforms", None)
        import vllm_fl.ops._C_ops_registry as registry

        with pytest.raises(RuntimeError, match="do not provide a mapped CUDA view"):
            registry._accelerator_view_from_cpu_tensor(torch.zeros(2))

    def test_no_vendor_attribute_raises_unavailable(self, monkeypatch):
        """A current_platform without vendor_name falls to the stub."""
        import vllm_fl.utils as utils

        class _ExplodingDeviceInfo:
            def __init__(self):
                raise RuntimeError("flag_gems missing")

        monkeypatch.setattr(utils, "DeviceInfo", _ExplodingDeviceInfo)
        vllm = _make_module("vllm")
        platforms = _make_module("vllm.platforms")
        platforms.current_platform = types.SimpleNamespace()
        vllm.platforms = platforms
        monkeypatch.setitem(sys.modules, "vllm", vllm)
        monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
        import vllm_fl.ops._C_ops_registry as registry

        with pytest.raises(RuntimeError, match="do not provide a mapped CUDA view"):
            registry._accelerator_view_from_cpu_tensor(torch.zeros(2))


class TestIsMetaxBackendProbe:
    """``_is_metax_backend`` resolved-hook and find_spec branches."""

    def _fresh_registry(self, monkeypatch):
        spec = importlib.util.spec_from_file_location(
            "supp_metax_probe_" + uuid.uuid4().hex[:8], _REGISTRY_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_resolved_hook_raising_falls_through(self, monkeypatch):
        """A raising is_metax hook degrades to the env/spec probes."""

        class _Platform:
            @staticmethod
            def is_metax(self):
                raise RuntimeError("platform not ready")

        platforms = _make_module("vllm.platforms")
        platforms._current_platform = _Platform()
        monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
        monkeypatch.delenv("GEMS_VENDOR", raising=False)
        monkeypatch.setitem(sys.modules, "mcoplib", None)
        registry = self._fresh_registry(monkeypatch)
        assert registry._is_metax_backend() is False

    def test_resolved_hook_non_callable_falls_through(self, monkeypatch):
        """A non-callable is_metax attribute degrades to the env/spec probes."""

        class _Platform:
            is_metax = "not-callable"

        platforms = _make_module("vllm.platforms")
        platforms._current_platform = _Platform()
        monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
        monkeypatch.delenv("GEMS_VENDOR", raising=False)
        monkeypatch.setitem(sys.modules, "mcoplib", None)
        registry = self._fresh_registry(monkeypatch)
        assert registry._is_metax_backend() is False

    def test_find_spec_mcoplib_present(self, monkeypatch):
        """An importable mcoplib package makes the probe True."""
        platforms = _make_module("vllm.platforms")
        monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
        monkeypatch.delenv("GEMS_VENDOR", raising=False)
        monkeypatch.setitem(sys.modules, "mcoplib", None)
        # find_spec consults sys.modules first; emulate presence via a finder.
        saved_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name == "mcoplib":
                # Simulate the real finder discovering the stub package.
                class _Spec:
                    pass

                return _Spec()
            return saved_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        registry = self._fresh_registry(monkeypatch)
        assert registry._is_metax_backend() is True

    def test_find_spec_raising_returns_false(self, monkeypatch):
        """A raising find_spec degrades to False."""
        platforms = _make_module("vllm.platforms")
        monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
        monkeypatch.delenv("GEMS_VENDOR", raising=False)

        def raising_find_spec(name, *args, **kwargs):
            raise ValueError(" finder broken")

        monkeypatch.setattr(importlib.util, "find_spec", raising_find_spec)
        registry = self._fresh_registry(monkeypatch)
        assert registry._is_metax_backend() is False


class TestRegisterOpSchemasCorners:
    """``register_op_schemas`` mcoplib / empty / define-failure branches."""

    def test_mcoplib_preload_success(self, monkeypatch):
        """An importable mcoplib._C is preloaded before schema definition."""
        prefix = "supp_mcop_preload_" + uuid.uuid4().hex[:8]
        schemas = [(prefix, "copy(Tensor input) -> Tensor")]
        registry = _load_registry(monkeypatch, schemas)
        # Remove the None mcoplib entry and make find_spec see it.
        monkeypatch.setitem(sys.modules, "mcoplib", _make_module("mcoplib"))
        monkeypatch.setitem(sys.modules, "mcoplib._C", _make_module("mcoplib._C"))
        saved_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name == "mcoplib":
                return object()  # truthy spec
            return saved_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        try:
            registry.register_op_schemas()
            assert prefix in registry.register_op_schemas._libs
        finally:
            _destroy(registry.register_op_schemas._libs)

    def test_mcoplib_preload_import_error_warns(self, monkeypatch, caplog):
        """An ImportError from mcoplib._C logs a warning and continues."""
        prefix = "supp_mcop_fail_" + uuid.uuid4().hex[:8]
        schemas = [(prefix, "copy(Tensor input) -> Tensor")]
        registry = _load_registry(monkeypatch, schemas)
        monkeypatch.setitem(sys.modules, "mcoplib", _make_module("mcoplib"))
        # mcoplib._C as a None entry raises ImportError on import.
        monkeypatch.setitem(sys.modules, "mcoplib._C", None)
        saved_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name == "mcoplib":
                return object()
            return saved_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        try:
            with caplog.at_level(logging.WARNING, logger=registry.logger.name):
                registry.register_op_schemas()
            assert any(
                "Failed to import mcoplib._C" in r.message for r in caplog.records
            )
            assert prefix in registry.register_op_schemas._libs
        finally:
            _destroy(registry.register_op_schemas._libs)

    def test_empty_schemas_warns(self, monkeypatch, caplog):
        """An empty SCHEMAS list logs the torch.compile warning and stops."""
        registry = _load_registry(monkeypatch, [])
        try:
            with caplog.at_level(logging.WARNING, logger=registry.logger.name):
                registry.register_op_schemas()
            assert any("No op schemas found" in r.message for r in caplog.records)
            assert getattr(registry.register_op_schemas, "_libs", None) is None
        finally:
            _destroy(getattr(registry.register_op_schemas, "_libs", None))

    def test_invalid_schema_raises_runtime_error(self, monkeypatch):
        """A schema torch cannot define raises the chained RuntimeError."""
        prefix = "supp_bad_schema_" + uuid.uuid4().hex[:8]
        # Missing return type: torch.library rejects this.
        schemas = [(prefix, "copy(Tensor input)")]
        registry = _load_registry(monkeypatch, schemas)
        try:
            with pytest.raises(RuntimeError, match="Failed to register"):
                registry.register_op_schemas()
        finally:
            _destroy(getattr(registry.register_op_schemas, "_libs", None))
        gc.collect()
