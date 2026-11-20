# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary CPU tests for ``vllm_fl/__init__.py`` (plugin bootstrap).

The plugin's top-level module runs its FlagGems/Triton compatibility patch and
the torch.float4 sentinel at import time, and exposes the vendor-branching
registration helpers vLLM's entry point calls.  Import-time behaviour needs a
*fresh copy* per scenario: ``spec_from_file_location`` under an alias package
name with ``submodule_search_locations`` pointing at the real ``vllm_fl``
directory (so ``from . import version`` resolves to the real version module
while every other import goes through whatever stubs the test installs).  The
registration helpers are plain functions, so one shared fresh copy serves those
tests.

Two suite lessons drive the stub discipline here:

* names a fixture restores in its teardown are never touched by
  ``monkeypatch.setitem`` (the undo would re-delete the restored module) —
  stubs installed against restored names use manual try/finally restores;
* a replaced ``builtins.__import__`` must be SELECTIVE (only the target names
  are intercepted, everything else delegates to the real importer) and must be
  restored in a ``finally`` block, otherwise pytest's own machinery breaks.
"""

import contextlib
import importlib
import importlib.util
import logging
import os
import platform
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.supplementary

_ROOT = Path(__file__).resolve().parents[3]
_INIT_SOURCE = _ROOT / "vllm_fl" / "__init__.py"
_VLLM_FL_DIR = _ROOT / "vllm_fl"
_ALIAS = "supp_toplevel_alias"

_TRITON_NAMES = ["triton", "triton.language", "triton.knobs"]


class _RecordingHandler(logging.Handler):
    """``logging.Handler`` double recording (level, message) pairs."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


def _make_module(name: str, **attrs: Any) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _snapshot(names):
    return {name: sys.modules.get(name) for name in names}


def _restore(saved):
    for name, old in saved.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


def _load_fresh(alias: str = _ALIAS) -> ModuleType:
    """Load and EXECUTE a fresh copy of ``vllm_fl/__init__.py`` as ``alias``."""
    spec = importlib.util.spec_from_file_location(
        alias, _INIT_SOURCE, submodule_search_locations=[str(_VLLM_FL_DIR)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module


def _pop_alias(alias: str) -> None:
    """Remove an alias module and its executed submodule entries."""
    for name in [n for n in sys.modules if n == alias or n.startswith(alias + ".")]:
        sys.modules.pop(name, None)


@contextlib.contextmanager
def _installed_stubs(stubs: dict):
    """Manually install/restore sys.modules entries (suite lesson: no
    monkeypatch.setitem on names a fixture teardown may restore)."""
    names = list(stubs)
    saved = _snapshot(names)
    for name, mod in stubs.items():
        sys.modules[name] = mod
    try:
        yield stubs
    finally:
        _restore(saved)


def _vendor_device_info(vendor_name: str):
    class _DeviceInfo:
        """DeviceInfo double whose probes report the configured vendor."""

        def __init__(self):
            self.vendor_name = vendor_name

    return _DeviceInfo


# ---------------------------------------------------------------------------
# Import-time behaviour: each test loads its own fresh copy.
# ---------------------------------------------------------------------------


class TestTritonCompatPatch:
    """``_patch_flag_gems_triton_import_compat`` import-time branches."""

    def test_sentinel_and_synthetic_knobs(self):
        """A Triton lacking map_elementwise and knobs gets both shims."""
        triton = _make_module("triton")
        tl = _make_module("triton.language")
        triton.language = tl
        saved = _snapshot(_TRITON_NAMES)
        sys.modules["triton"] = triton
        sys.modules["triton.language"] = tl
        # On hosts with real Triton installed, a real triton.knobs sits in
        # sys.modules and importlib would return it instead of synthesizing
        # the shim — pop it so the "knobs missing" scenario is forced.  The
        # snapshot above restores it afterwards.
        sys.modules.pop("triton.knobs", None)
        alias = _ALIAS + "_tr"
        try:
            module = _load_fresh(alias)
            assert hasattr(module, "register")
            assert hasattr(tl, "map_elementwise")
            sentinel = tl.map_elementwise
            assert sentinel.__name__ == "map_elementwise"
            assert sentinel.__triton_builtin__ is True
            with pytest.raises(NotImplementedError, match="Kunlunxin"):
                sentinel()
            knobs = sys.modules.get("triton.knobs")
            assert knobs is not None
            assert knobs.autotuning.adjust_block_size is True
            assert triton.knobs is knobs
        finally:
            _pop_alias(alias)
            _restore(saved)

    def test_wrong_name_modulenotfound_reraised(self, monkeypatch):
        """A ModuleNotFoundError for a different module propagates."""
        real_import_module = importlib.import_module

        def raise_other(name):
            # Delegate to the real importer: the fresh copy's import of
            # triton.knobs must fail with the WRONG name to exercise the
            # re-raise branch (line 42).
            if name == "triton.knobs":
                raise ModuleNotFoundError("nope", name="other.module")
            return real_import_module(name)

        monkeypatch.setattr(importlib, "import_module", raise_other)
        triton = _make_module("triton")
        tl = _make_module("triton.language")
        tl.map_elementwise = lambda *a, **k: None
        triton.language = tl
        saved = _snapshot(_TRITON_NAMES)
        sys.modules["triton"] = triton
        sys.modules["triton.language"] = tl
        # Ensure triton.knobs is NOT resolvable, so import_module is reached.
        sys.modules.pop("triton.knobs", None)
        alias = _ALIAS + "_wrongname"
        try:
            with pytest.raises(ModuleNotFoundError, match="nope"):
                _load_fresh(alias)
        finally:
            _pop_alias(alias)
            _restore(saved)

    def test_triton_import_failure_returns(self):
        """A missing Triton makes the patch a no-op; the module still loads."""
        saved = _snapshot(_TRITON_NAMES)
        sys.modules["triton"] = None  # `import triton` raises ImportError
        alias = _ALIAS + "_noimport"
        try:
            module = _load_fresh(alias)
            assert hasattr(module, "register")
        finally:
            _pop_alias(alias)
            _restore(saved)

    def test_knobs_present_left_alone(self):
        """An already importable triton.knobs is not replaced."""
        triton = _make_module("triton")
        tl = _make_module("triton.language")
        tl.map_elementwise = lambda *a, **k: None
        triton.language = tl
        real_knobs = _make_module("triton.knobs", MARK="real")
        saved = _snapshot(_TRITON_NAMES)
        sys.modules["triton"] = triton
        sys.modules["triton.language"] = tl
        sys.modules["triton.knobs"] = real_knobs
        alias = _ALIAS + "_knobspresent"
        try:
            _load_fresh(alias)
            assert sys.modules["triton.knobs"] is real_knobs
            assert sys.modules["triton.knobs"].MARK == "real"
        finally:
            _pop_alias(alias)
            _restore(saved)


class TestFloat4Sentinel:
    """The torch.float4 sentinel injected at import time."""

    def test_sentinel_injected_when_missing(self):
        """A torch build without float4 gets uint8 bound in its place."""
        import torch

        saved = getattr(torch, "float4_e2m1fn_x2", None)
        if saved is not None:
            del torch.float4_e2m1fn_x2
        alias = _ALIAS + "_f4"
        try:
            module = _load_fresh(alias)
            assert hasattr(module, "register")
            assert torch.float4_e2m1fn_x2 is torch.uint8
        finally:
            _pop_alias(alias)
            if saved is None:
                if hasattr(torch, "float4_e2m1fn_x2"):
                    del torch.float4_e2m1fn_x2
            else:
                torch.float4_e2m1fn_x2 = saved

    def test_existing_float4_kept(self):
        """A torch build shipping float4 keeps its own dtype."""
        import torch

        if not hasattr(torch, "float4_e2m1fn_x2"):
            pytest.skip("torch build without float4_e2m1fn_x2")
        alias = _ALIAS + "_f4keep"
        try:
            module = _load_fresh(alias)
            assert hasattr(module, "register")
            assert torch.float4_e2m1fn_x2 is not torch.uint8
        finally:
            _pop_alias(alias)


class TestModuleGetattr:
    """``__getattr__`` on the freshly loaded plugin package."""

    def test_distributed_lazy_import(self):
        """The distributed attribute resolves to the aliased subpackage."""
        alias = _ALIAS + "_getattr"
        sub_name = alias + ".distributed"
        try:
            module = _load_fresh(alias)
            distributed = module.distributed
            assert isinstance(distributed, ModuleType)
            assert sys.modules[sub_name] is distributed
        finally:
            _pop_alias(alias)

    def test_unknown_attribute_raises(self):
        """Any other attribute name raises AttributeError."""
        alias = _ALIAS + "_getattr2"
        try:
            module = _load_fresh(alias)
            with pytest.raises(AttributeError, match="has no attribute"):
                module.definitely_not_here  # noqa: B018 (attribute access)
        finally:
            _pop_alias(alias)


# ---------------------------------------------------------------------------
# Function-call helpers: a shared fresh copy plus targeted stubs.
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_module():
    """A fresh, executed copy of the plugin top-level module."""
    module = _load_fresh(_ALIAS + "_shared")
    yield module
    _pop_alias(_ALIAS + "_shared")


def _capture(module: str, func, *args):
    """Run ``func`` while recording records emitted on ``logging.getLogger``.

    The plugin logger defaults to the root logger's WARNING threshold, so the
    DEBUG records several helpers emit would be filtered before reaching any
    handler; the logger's level is lowered for the duration and restored.
    """
    handler = _RecordingHandler()
    logger = logging.getLogger(module)
    saved_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        result = func(*args)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved_level)
    return result, handler.records


class TestRegisterFlagcxConnector:
    """``_register_flagcx_connector`` registry bookkeeping."""

    _FACTORY = "vllm.distributed.kv_transfer.kv_connector.factory"

    def test_registers_both_aliases(self, fresh_module):
        """FlagCXConnector and FlagcxConnector are registered once each."""
        registered = []

        class _Factory:
            _registry = set()

            @classmethod
            def register_connector(cls, name, module_path, cls_name):
                registered.append((name, module_path, cls_name))
                cls._registry.add(name)

        factory_mod = _make_module(self._FACTORY, KVConnectorFactory=_Factory)
        with _installed_stubs({self._FACTORY: factory_mod}):
            fresh_module._register_flagcx_connector()
        assert registered == [
            (
                "FlagCXConnector",
                "vllm_fl.distributed.kv_transfer.flagcx_connector",
                "FlagCXConnector",
            ),
            (
                "FlagcxConnector",
                "vllm_fl.distributed.kv_transfer.flagcx_connector",
                "FlagCXConnector",
            ),
        ]

    def test_already_registered_skips(self, fresh_module):
        """Pre-registered aliases are not re-registered."""

        class _Factory:
            _registry = {"FlagCXConnector", "FlagcxConnector"}

            @classmethod
            def register_connector(cls, *args):
                raise AssertionError("must not be called")

        factory_mod = _make_module(self._FACTORY, KVConnectorFactory=_Factory)
        with _installed_stubs({self._FACTORY: factory_mod}):
            fresh_module._register_flagcx_connector()


class TestPatchCustomOps:
    """``_patch_custom_ops`` extension probe order.

    The code under test probes extensions through
    ``importlib.import_module``, which returns an existing ``sys.modules``
    entry unchanged and raises ``ImportError`` for a ``None`` entry — so the
    probes are steered purely through ``sys.modules`` state.
    """

    _REGISTRY = "vllm_fl.ops._C_ops_registry"
    _EXTS = ["vllm._C", "vllm._C_stable_libtorch", "vllm_fl._C"]

    def test_metax_backend_skips_probe(self, fresh_module):
        """A MetaX backend answers True and skips the vllm._C probe."""
        calls = []
        registry_mod = _make_module(
            self._REGISTRY,
            _is_metax_backend=lambda: True,
            register_op_schemas=lambda: calls.append("schemas"),
        )
        with _installed_stubs({self._REGISTRY: registry_mod}):
            _, records = _capture(
                fresh_module.logger.name, fresh_module._patch_custom_ops
            )
        assert calls == []
        assert any("MetaX backend" in msg for _, msg in records)

    def test_vllm_c_present_returns_early(self, fresh_module):
        """An importable vllm._C short-circuits the fallback path."""
        calls = []
        registry_mod = _make_module(
            self._REGISTRY,
            _is_metax_backend=lambda: False,
            register_op_schemas=lambda: calls.append("schemas"),
        )
        stubs = {name: None for name in self._EXTS}
        stubs["vllm._C"] = _make_module("vllm._C")
        stubs[self._REGISTRY] = registry_mod
        with _installed_stubs(stubs):
            fresh_module._patch_custom_ops()
        assert calls == []

    def test_fallback_registers_schemas(self, fresh_module):
        """No vLLM extension present: vllm_fl._C is probed, schemas register."""
        calls = []
        registry_mod = _make_module(
            self._REGISTRY,
            _is_metax_backend=lambda: False,
            register_op_schemas=lambda: calls.append("schemas"),
        )
        stubs = {name: None for name in self._EXTS}
        stubs[self._REGISTRY] = registry_mod
        with _installed_stubs(stubs):
            _, records = _capture(
                fresh_module.logger.name, fresh_module._patch_custom_ops
            )
        assert calls == ["schemas"]
        assert any("Failed to import vllm_fl._C" in msg for _, msg in records)


class TestMetaxCompat:
    """``_install_metax_compat`` shim installation."""

    _SHIMS = "vllm_fl.dispatch.backends.vendor.metax.shims"

    def test_active_installs(self, fresh_module):
        """An active MetaX runtime gets the shims installed."""
        calls = []

        class _Install:
            pass

        shims_mod = _make_module(
            self._SHIMS,
            install=lambda: calls.append("install"),
            metax_active=lambda: True,
        )
        with _installed_stubs({self._SHIMS: shims_mod}):
            fresh_module._install_metax_compat()
        assert calls == ["install"]

    def test_inactive_or_failing_warns(self, fresh_module):
        """A failed shim import logs a warning and never raises."""
        shims_mod = None
        # Import failure via a None entry (ImportError on `from ... import`).
        with _installed_stubs({self._SHIMS: shims_mod}):
            _, records = _capture(
                fresh_module.logger.name, fresh_module._install_metax_compat
            )
        assert any(
            "MetaX compatibility shims not installed" in msg for _, msg in records
        )


class TestVendorDeviceHooks:
    """``_init_vendor_device`` per-vendor branches."""

    _MUSA_PATCH = "vllm_fl.dispatch.backends.vendor.musa.patch"
    _KUNLUNXIN_PATCH = (
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils"
    )
    _FLAGTREE = "vllm_fl.patches.flagtree_target_info"
    _UTILS = "vllm_fl.utils"

    def _with_utils(self, vendor):
        utils_mod = _make_module(self._UTILS, DeviceInfo=_vendor_device_info(vendor))
        return utils_mod

    def test_mthreads_failure_raises(self, fresh_module):
        """A failing MUSA accelerator patch aborts registration."""

        class _Error(Exception):
            pass

        musa_mod = _make_module(
            self._MUSA_PATCH,
            patch_accelerator_missing_attrs=lambda: False,
        )
        stubs = {
            self._MUSA_PATCH: musa_mod,
            self._UTILS: self._with_utils("mthreads"),
        }
        with _installed_stubs(stubs), pytest.raises(RuntimeError, match="MUSA"):
            fresh_module._init_vendor_device()

    def test_mthreads_success(self, fresh_module):
        """A successful MUSA patch completes initialization."""
        musa_mod = _make_module(
            self._MUSA_PATCH,
            patch_accelerator_missing_attrs=lambda: True,
        )
        stubs = {
            self._MUSA_PATCH: musa_mod,
            self._UTILS: self._with_utils("mthreads"),
        }
        with _installed_stubs(stubs):
            fresh_module._init_vendor_device()

    def test_kunlunxin_xpu_patch(self, fresh_module):
        """Kunlunxin runs the XPU device patch before model imports."""
        calls = []
        kunlunxin_mod = _make_module(
            self._KUNLUNXIN_PATCH, _patch_xpu_get_device=lambda: calls.append("xpu")
        )
        stubs = {
            self._KUNLUNXIN_PATCH: kunlunxin_mod,
            self._UTILS: self._with_utils("kunlunxin"),
        }
        with _installed_stubs(stubs):
            fresh_module._init_vendor_device()
        assert calls == ["xpu"]

    def test_amd_flagtree_applied(self, fresh_module):
        """AMD runs the FlagTree target-info patch before model imports."""
        calls = []
        flagtree_mod = _make_module(
            self._FLAGTREE, apply_flagtree_target_info=lambda: calls.append("amd")
        )
        stubs = {
            self._FLAGTREE: flagtree_mod,
            self._UTILS: self._with_utils("amd"),
        }
        with _installed_stubs(stubs):
            fresh_module._init_vendor_device()
        assert calls == ["amd"]


class TestKunlunxinFlaAlias:
    """``_install_kunlunxin_fla_alias`` lock and failure handling."""

    _KUNLUNXIN_PATCH = (
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils"
    )
    _RELOCATION = "vllm_fl.patches.fla_module_relocation"

    def test_installs_alias_and_fixes_vars(self, fresh_module):
        """A successful run installs the FLA alias then fixes platform vars."""
        calls = []
        kunlunxin_mod = _make_module(
            self._KUNLUNXIN_PATCH,
            _fix_platform_vars=lambda: calls.append("fix"),
        )
        relocation_mod = _make_module(
            self._RELOCATION, install=lambda: calls.append("install")
        )
        stubs = {
            self._KUNLUNXIN_PATCH: kunlunxin_mod,
            self._RELOCATION: relocation_mod,
        }
        with _installed_stubs(stubs):
            fresh_module._install_kunlunxin_fla_alias()
        assert calls == ["install", "fix"]
        assert not fresh_module._kunlunxin_fla_alias_lock.locked()

    def test_lock_held_returns_silently(self, fresh_module):
        """A re-entered call returns without touching the lock."""
        with fresh_module._kunlunxin_fla_alias_lock:
            fresh_module._install_kunlunxin_fla_alias()

    def test_import_failure_logs_debug(self, fresh_module):
        """An unavailable FLA alias logs at DEBUG and releases the lock."""
        stubs = {self._RELOCATION: None}
        with _installed_stubs(stubs):
            _, records = _capture(
                fresh_module.logger.name,
                fresh_module._install_kunlunxin_fla_alias,
            )
        assert any("FLA module alias unavailable" in msg for _, msg in records)
        assert not fresh_module._kunlunxin_fla_alias_lock.locked()


class TestRegister:
    """``register()`` vendor matrix."""

    _UTILS = "vllm_fl.utils"
    _KUNLUNXIN_PATCH = (
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils"
    )
    _RELOCATION = "vllm_fl.patches.fla_module_relocation"
    _METAX_SHIMS = "vllm_fl.dispatch.backends.vendor.metax.shims"
    _FLAGCX_FACTORY = "vllm.distributed.kv_transfer.kv_connector.factory"

    def _env_view(self):
        """Detached plain-dict stand-in for os.environ (direct writes)."""
        return {k: v for k, v in os.environ.items()}

    def test_arm_cpu_short_circuits(self, fresh_module, monkeypatch):
        """An ARM CPU target returns vLLM's own CPU platform untouched."""
        monkeypatch.setattr(platform, "machine", lambda: "aarch64")
        cpu_plugin = _make_module("vllm.platforms", cpu_platform_plugin=lambda: "cpu")
        utils_mod = _make_module(self._UTILS, DeviceInfo=_vendor_device_info("nvidia"))
        stubs = {
            "vllm.platforms": cpu_plugin,
            self._UTILS: utils_mod,
            self._METAX_SHIMS: None,
        }
        with _installed_stubs(stubs):
            result, records = _capture(fresh_module.logger.name, fresh_module.register)
        assert result == "cpu"
        assert any("ARM64 CPU target" in msg for _, msg in records)

    def test_kunlunxin_env_and_alias(self, fresh_module, monkeypatch):
        """Kunlunxin forces MOE padding off and installs the FLA alias."""
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        env = self._env_view()
        monkeypatch.setattr(os, "environ", env)

        calls = []
        utils_mod = _make_module(
            self._UTILS,
            DeviceInfo=_vendor_device_info("kunlunxin"),
            get_op_config=lambda: "cfg",
        )
        kunlunxin_mod = _make_module(
            self._KUNLUNXIN_PATCH, _patch_xpu_get_device=lambda: calls.append("xpu")
        )
        relocation_mod = _make_module(
            self._RELOCATION, install=lambda: calls.append("fla")
        )
        stubs = {
            self._UTILS: utils_mod,
            self._KUNLUNXIN_PATCH: kunlunxin_mod,
            self._RELOCATION: relocation_mod,
            self._METAX_SHIMS: None,
        }
        # _install_kunlunxin_fla_alias imports _fix_platform_vars from the
        # same kunlunxin patch module; give it a recording double.
        kunlunxin_mod._fix_platform_vars = lambda: calls.append("fix")
        with _installed_stubs(stubs):
            result = fresh_module.register()
        assert env["VLLM_MOE_SKIP_PADDING"] == "0"
        assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
        assert (
            "VLLM_USE_V2_MODEL_RUNNER" not in env
            or env["VLLM_USE_V2_MODEL_RUNNER"] != "0"
        )
        assert calls == ["xpu", "fla", "fix"]
        assert result == "vllm_fl.platform.PlatformFL"

    def test_nvidia_returns_native_path(self, fresh_module, monkeypatch):
        """NVIDIA keeps vLLM's native extensions and its own platform."""
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        env = self._env_view()
        monkeypatch.setattr(os, "environ", env)
        utils_mod = _make_module(
            self._UTILS,
            DeviceInfo=_vendor_device_info("nvidia"),
            get_op_config=lambda: "cfg",
        )
        stubs = {
            self._UTILS: utils_mod,
            self._METAX_SHIMS: None,
        }
        with _installed_stubs(stubs):
            result = fresh_module.register()
        assert result == "vllm_fl.nvidia_platform.NvidiaPlatformFL"
        assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"

    def test_other_vendor_patches_ops(self, fresh_module, monkeypatch):
        """Non-NVIDIA vendors install the stub patches before returning."""
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        env = self._env_view()
        monkeypatch.setattr(os, "environ", env)
        calls = []
        utils_mod = _make_module(
            self._UTILS,
            DeviceInfo=_vendor_device_info("gcu"),
            get_op_config=lambda: "cfg",
        )
        stubs = {
            self._UTILS: utils_mod,
            self._METAX_SHIMS: None,
        }
        # Stub the functions register() itself calls for non-nvidia vendors.
        fresh_module._patch_custom_ops = lambda: calls.append("custom_ops")
        fresh_module._patch_flash_attn_import = lambda: calls.append("flash_attn")
        with _installed_stubs(stubs):
            result = fresh_module.register()
        assert calls == ["custom_ops", "flash_attn"]
        assert env["VLLM_USE_V2_MODEL_RUNNER"] == "0"
        assert result == "vllm_fl.platform.PlatformFL"


class TestRegisterQuantLinear:
    """``register_quant_linear`` device gating."""

    def _utils_stub(self):
        return _make_module(
            "vllm_fl.quantization.quant_linear", add_oot_quant_kernel=lambda: None
        )

    def test_musa_skips(self, fresh_module):
        """MUSA's linear kernels are unavailable; nothing is registered."""
        platforms_mod = _make_module(
            "vllm.platforms", current_platform=SimpleNamespace(device_type="musa")
        )
        with _installed_stubs({"vllm.platforms": platforms_mod}):
            fresh_module.register_quant_linear()

    def test_gcu_skips(self, fresh_module):
        """GCU's linear kernels are unavailable; nothing is registered."""
        platforms_mod = _make_module(
            "vllm.platforms", current_platform=SimpleNamespace(device_type="gcu")
        )
        with _installed_stubs({"vllm.platforms": platforms_mod}):
            fresh_module.register_quant_linear()

    def test_cuda_registers(self, fresh_module):
        """CUDA registers the OOT quant kernel."""
        calls = []
        quant_mod = _make_module(
            "vllm_fl.quantization.quant_linear",
            add_oot_quant_kernel=lambda: calls.append("oot"),
        )
        platforms_mod = _make_module(
            "vllm.platforms", current_platform=SimpleNamespace(device_type="cuda")
        )
        stubs = {
            "vllm.platforms": platforms_mod,
            "vllm_fl.quantization.quant_linear": quant_mod,
        }
        with _installed_stubs(stubs):
            fresh_module.register_quant_linear()
        assert calls == ["oot"]


class TestRegisterRouter:
    """``register_router`` OOT gating."""

    _UTILS = "vllm_fl.utils"
    _ROUTER = "vllm_fl.ops.fused_moe.router"

    def test_musa_skips(self, fresh_module):
        """MUSA's fused MoE import chain is unavailable; nothing happens."""
        platforms_mod = _make_module(
            "vllm.platforms", current_platform=SimpleNamespace(device_type="musa")
        )
        with _installed_stubs({"vllm.platforms": platforms_mod}):
            fresh_module.register_router()

    def test_oot_disabled_skips(self, fresh_module):
        """A disabled OOT flag leaves the router untouched."""
        platforms_mod = _make_module(
            "vllm.platforms", current_platform=SimpleNamespace(device_type="cuda")
        )
        utils_mod = _make_module(self._UTILS, is_oot_enabled=lambda: False)
        with _installed_stubs(
            {"vllm.platforms": platforms_mod, self._UTILS: utils_mod}
        ):
            fresh_module.register_router()

    def test_oot_enabled_replaces_router(self, fresh_module):
        """An enabled OOT flag installs the FL fused-MoE router."""
        calls = []
        platforms_mod = _make_module(
            "vllm.platforms", current_platform=SimpleNamespace(device_type="cuda")
        )
        utils_mod = _make_module(self._UTILS, is_oot_enabled=lambda: True)
        router_mod = _make_module(
            self._ROUTER, replace_router_with_fl=lambda: calls.append("router")
        )
        stubs = {
            "vllm.platforms": platforms_mod,
            self._UTILS: utils_mod,
            self._ROUTER: router_mod,
        }
        with _installed_stubs(stubs):
            fresh_module.register_router()
        assert calls == ["router"]


class TestGdnPackedDecodePatch:
    """``_register_gdn_packed_decode_patch`` capability probe."""

    _PATCH = "vllm_fl.patches.gdn_packed_decode"

    def test_missing_module_returns_false(self, fresh_module):
        """A build without the patch module reports unavailability."""
        with _installed_stubs({self._PATCH: None}):
            _, records = _capture(
                fresh_module.logger.name,
                fresh_module._register_gdn_packed_decode_patch,
            )
        assert any("unavailable" in msg for _, msg in records)

    def test_missing_symbol_returns_false(self, fresh_module):
        """A build without the patch symbol reports unavailability."""
        patch_mod = _make_module(self._PATCH)  # no patch_vllm_packed_gdn_beta
        with _installed_stubs({self._PATCH: patch_mod}):
            result, records = _capture(
                fresh_module.logger.name,
                fresh_module._register_gdn_packed_decode_patch,
            )
        assert result is False
        assert any("unavailable" in msg for _, msg in records)

    def test_patch_function_invoked(self, fresh_module):
        """A capable build runs the packed GDN patch."""
        calls = []
        patch_mod = _make_module(
            self._PATCH, patch_vllm_packed_gdn_beta=lambda: calls.append("gdn") or True
        )
        with _installed_stubs({self._PATCH: patch_mod}):
            result = fresh_module._register_gdn_packed_decode_patch()
        assert result is True
        assert calls == ["gdn"]


class TestArmCpuPlatform:
    """``_arm_cpu_platform`` machine gating."""

    def test_x86_returns_none(self, fresh_module, monkeypatch):
        """A non-AArch64 host has no ARM CPU platform."""
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        assert fresh_module._arm_cpu_platform() is None

    def test_arm64_returns_plugin(self, fresh_module, monkeypatch):
        """An ARM64 host resolves vLLM's CPU platform plugin."""
        monkeypatch.setattr(platform, "machine", lambda: "arm64")
        plugin_mod = _make_module(
            "vllm.platforms", cpu_platform_plugin=lambda: "cpu-plugin"
        )
        with _installed_stubs({"vllm.platforms": plugin_mod}):
            assert fresh_module._arm_cpu_platform() == "cpu-plugin"


class TestRegisterModel:
    """``register_model`` vendor matrix."""

    _UTILS = "vllm_fl.utils"
    _METAX_SHIMS = "vllm_fl.dispatch.backends.vendor.metax.shims"
    _FLAGCX_FACTORY = "vllm.distributed.kv_transfer.kv_connector.factory"
    _QUANT = "vllm_fl.quantization.quant_linear"
    _GDN = "vllm_fl.patches.gdn_packed_decode"

    def _platforms(self, device_type):
        return _make_module(
            "vllm.platforms",
            current_platform=SimpleNamespace(device_type=device_type),
            cpu_platform_plugin=lambda: "cpu-plugin",
        )

    def _base_stubs(self, vendor):
        """Stubs every register_model path needs past the first imports."""
        return {
            "vllm.platforms": self._platforms("cpu"),
            self._UTILS: _make_module(
                self._UTILS,
                DeviceInfo=_vendor_device_info(vendor),
                is_oot_enabled=lambda: False,
            ),
            self._METAX_SHIMS: None,
            self._FLAGCX_FACTORY: _make_module(
                self._FLAGCX_FACTORY,
                KVConnectorFactory=SimpleNamespace(
                    _registry=set(), register_connector=lambda *a: None
                ),
            ),
            self._QUANT: _make_module(self._QUANT, add_oot_quant_kernel=lambda: None),
        }

    def _gdn_stub(self, calls):
        return _make_module(
            self._GDN, patch_vllm_packed_gdn_beta=lambda: calls.append("gdn")
        )

    def test_arm_cpu_flag_gems_missing_warns_and_returns(
        self, fresh_module, monkeypatch
    ):
        """An ARM CPU target without FlagGems disables integration quietly."""
        monkeypatch.setattr(platform, "machine", lambda: "aarch64")
        stubs = self._base_stubs("nvidia")
        stubs["vllm_fl.patches.arm_cpu_gdn"] = _make_module(
            "vllm_fl.patches.arm_cpu_gdn",
            apply_arm_cpu_gdn_state_indices_patch=lambda: None,
        )
        stubs["flag_gems"] = None  # `import flag_gems` raises ModuleNotFoundError
        with _installed_stubs(stubs):
            _, records = _capture(fresh_module.logger.name, fresh_module.register_model)
        assert any("FlagGems is not installed" in msg for _, msg in records)

    def test_arm_cpu_wrong_flag_gems_vendor(self, fresh_module, monkeypatch):
        """FlagGems built for another vendor leaves the ARM path untouched."""
        monkeypatch.setattr(platform, "machine", lambda: "aarch64")
        stubs = self._base_stubs("nvidia")
        stubs["vllm_fl.patches.arm_cpu_gdn"] = _make_module(
            "vllm_fl.patches.arm_cpu_gdn",
            apply_arm_cpu_gdn_state_indices_patch=lambda: None,
        )
        stubs["flag_gems"] = _make_module("flag_gems", vendor_name="cuda")
        with _installed_stubs(stubs):
            _, records = _capture(fresh_module.logger.name, fresh_module.register_model)
        assert any("not 'arm'" in msg for _, msg in records)

    def test_arm_cpu_packed_w4a8_installed(self, fresh_module, monkeypatch):
        """A matching FlagGems installs the ARM packed W4A8 integration."""
        monkeypatch.setattr(platform, "machine", lambda: "aarch64")
        calls = []
        stubs = self._base_stubs("nvidia")
        stubs["vllm_fl.patches.arm_cpu_gdn"] = _make_module(
            "vllm_fl.patches.arm_cpu_gdn",
            apply_arm_cpu_gdn_state_indices_patch=lambda: None,
        )
        stubs["flag_gems"] = _make_module("flag_gems", vendor_name="arm")
        stubs["vllm_fl.quantization.arm_cpu_w4a8"] = _make_module(
            "vllm_fl.quantization.arm_cpu_w4a8",
            install_arm_cpu_packed_w4a8=lambda: calls.append("w4a8"),
        )
        with _installed_stubs(stubs):
            fresh_module.register_model()
        assert calls == ["w4a8"]

    def test_iluvatar_installs_cleanup(self, fresh_module, monkeypatch):
        """Iluvatar registers its worker shutdown cleanup hook."""
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        calls = []
        stubs = self._base_stubs("iluvatar")
        stubs["vllm.platforms"] = self._platforms("cuda")
        stubs["vllm_fl.dispatch.backends.vendor.iluvatar.shutdown"] = _make_module(
            "vllm_fl.dispatch.backends.vendor.iluvatar.shutdown",
            install_worker_cleanup=lambda: calls.append("cleanup"),
        )
        stubs[self._GDN] = self._gdn_stub(calls)
        with _installed_stubs(stubs):
            fresh_module.register_model()
        assert calls == ["cleanup", "gdn"]

    def test_nvidia_minicpmo_patch(self, fresh_module, monkeypatch):
        """NVIDIA applies the MiniCPM-o audio budget patch."""
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        calls = []
        stubs = self._base_stubs("nvidia")
        stubs["vllm.platforms"] = self._platforms("cuda")
        stubs["vllm_fl.patches.minicpmo_audio"] = _make_module(
            "vllm_fl.patches.minicpmo_audio",
            patch_minicpmo_audio_budget=lambda: calls.append("minicpmo"),
        )
        with _installed_stubs(stubs):
            fresh_module.register_model()
        assert calls == ["minicpmo"]

    def test_ascend_registers_models_and_modelslim(self, fresh_module, monkeypatch):
        """Ascend registers its DeepSeek/GLM models and ModelSlim config."""
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        calls = []

        class _Registry:
            @staticmethod
            def register_model(name, path):
                calls.append((name, path))

        stubs = self._base_stubs("ascend")
        stubs["vllm.platforms"] = self._platforms("npu")
        stubs["vllm.model_executor.models"] = _make_module(
            "vllm.model_executor.models", ModelRegistry=_Registry
        )
        stubs["vllm_fl.quantization.modelslim_w8a8"] = _make_module(
            "vllm_fl.quantization.modelslim_w8a8"
        )
        stubs[self._GDN] = self._gdn_stub(calls)
        with _installed_stubs(stubs):
            fresh_module.register_model()
        assert (
            "DeepseekV4ForCausalLM",
            "vllm_fl.models.deepseek_v4:DeepseekV4ForCausalLM",
        ) in calls
        assert (
            "GlmMoeDsaForCausalLM",
            "vllm_fl.models.glm_moe_dsa:GlmMoeDsaForCausalLM",
        ) in calls

    def test_generic_vendor_registers_flagcx(self, fresh_module, monkeypatch):
        """The default path registers the FlagCX connector and kernels."""
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
        calls = []
        factory = _make_module(
            self._FLAGCX_FACTORY,
            KVConnectorFactory=SimpleNamespace(
                _registry=set(),
                register_connector=lambda *a: calls.append(("flagcx", a[0])),
            ),
        )
        stubs = self._base_stubs("gcu")
        stubs[self._FLAGCX_FACTORY] = factory
        stubs[self._GDN] = self._gdn_stub(calls)
        with _installed_stubs(stubs):
            fresh_module.register_model()
        assert ("flagcx", "FlagCXConnector") in calls
        assert ("flagcx", "FlagcxConnector") in calls
