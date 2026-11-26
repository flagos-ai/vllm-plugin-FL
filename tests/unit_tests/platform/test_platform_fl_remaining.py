# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary CPU tests for ``vllm_fl/platform.py`` (remaining branches).

Third companion to ``test_platform_fl_methods.py`` and
``test_platform_fl_device_introspection.py``: the same ``platform_env``
pattern (stub vllm graph + ``spec_from_file_location`` load) drives the
methods those two files leave uncovered -- ``get_attn_backend_cls`` dispatch,
MetaX/MUSA ``import_kernels`` branches, the MetaX
``apply_config_platform_defaults`` graph defaults, the ViT backend selection
matrix, the AMD static-graph wrapper, the Ascend hybrid-block alignment,
``pre_register_and_update`` and the ``_vllm_c_is_registered_provider`` edge
cases.

Import-boundary lesson (shared with the sibling files): inside a fixture that
pops ``sys.modules`` entries in its teardown, ``monkeypatch.setitem`` on those
exact names must never be used (the undo re-deletes the restored real
modules). Stubs for lazy imports that only run inside a method under test are
installed with plain ``sys.modules`` writes and a manual try/finally restore
here so no fixture-managed name is ever touched.
"""

import importlib.util
import importlib.util as iu
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]
_PLATFORM_SOURCE = _ROOT / "vllm_fl" / "platform.py"
_PLATFORM_MODULE = "supp_platform_fl_remaining_under_test"

_STUB_ENTRIES = [
    "vllm",
    "vllm.logger",
    "vllm.platforms",
    "vllm.platforms.interface",
    "vllm.platforms.rocm",
    "vllm.v1.attention.backends.registry",
    "vllm.config",
    "vllm.config.compilation",
    "vllm.config.kernel",
    "vllm.ir",
    "vllm.ir.op",
    "vllm.ir.ops",
    "vllm_fl.utils",
]


class _FakeLogger:
    """``vllm.logger.init_logger`` stand-in recording level calls."""

    def __init__(self):
        self.records = []

    def _log(self, level, msg, *args, **kwargs):
        self.records.append((level, msg % args if args else msg))

    def info(self, msg, *args, **kwargs):
        self._log("info", msg, *args)

    def debug(self, msg, *args, **kwargs):
        self._log("debug", msg, *args)

    def warning(self, msg, *args, **kwargs):
        self._log("warning", msg, *args)

    def exception(self, msg, *args, **kwargs):
        self._log("exception", msg, *args)

    def info_once(self, msg, *args, **kwargs):
        self._log("info_once", msg, *args)

    def warning_once(self, msg, *args, **kwargs):
        self._log("warning_once", msg, *args)

    def error(self, msg, *args, **kwargs):
        self._log("error", msg, *args)


def _make_module(name: str, **attrs: Any) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _DeviceInfoStub:
    """Vendor-neutral DeviceInfo double: probes report nothing."""

    vendor_name = ""
    device_type = ""
    device = SimpleNamespace(__class__=SimpleNamespace(__name__="NoDevice"))
    dispatch_key = "CUDA"
    torch_device_fn = torch.cuda


class _BrokenDataclassMeta(type):
    """Metaclass whose ``fields`` raises TypeError when called via ``fields()``."""

    @property
    def __dataclass_fields__(cls):
        raise TypeError("not a dataclass")


class _NotADataclass(metaclass=_BrokenDataclassMeta):
    """Stand-in that makes ``dataclasses.fields()`` raise TypeError."""


class _IrOp:
    def __init__(self, impls):
        self.impls = list(impls)


class _CapabilityStub:
    """DeviceCapability double with major >= 8 for the FLASH_ATTN probe."""

    major = 8
    minor = 0


class _Member:
    """An enum member value that also answers get_class()."""

    def __init__(self, value):
        self._value = value

    def get_class(self):
        raise ImportError("no backend class")

    def __eq__(self, other):
        return self._value == other

    def __hash__(self):
        return hash(self._value)

    def __repr__(self):
        return repr(self._value)


@pytest.fixture
def platform_env(monkeypatch):
    """Load platform.py against a stub vllm graph; yield (module, logger)."""
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.setenv("GEMS_VENDOR", "nvidia")

    logger = _FakeLogger()
    saved = {name: sys.modules.get(name) for name in _STUB_ENTRIES}

    class _Capability:
        def __init__(self, major=8, minor=0):
            self.major = major
            self.minor = minor

        def as_version_str(self):
            return f"{self.major}.{self.minor}"

    class _PlatformEnum:
        CUDA = "cuda"
        OOT = "oot"
        CPU = "cpu"

    class _BasePlatform:
        """Stub of vllm.platforms.Platform: records super() delegations."""

        dist_backend = "nccl"
        super_calls = []

        @classmethod
        def import_kernels(cls):
            cls.super_calls.append("import_kernels")

        @classmethod
        def _align_hybrid_block_size(cls, vllm_config, backend_cls):
            cls.super_calls.append("_align_hybrid_block_size")

        @classmethod
        def get_device_capability(cls, device_id=0):
            return _Capability()

        @classmethod
        def has_device_capability(cls, major):
            return major <= 8

    class _RocmPlatform:
        @staticmethod
        def get_supported_vit_attn_backends():
            return ["rocm-sdpa", "rocm-flash"]

        @staticmethod
        def get_vit_attn_backend(head_size, dtype, backend):
            return "rocm-vit"

    class _AttentionBackendEnum:
        TORCH_SDPA = "torch_sdpa"
        FLASH_ATTN = "flash_attn"

        def get_class(self):
            raise ImportError("no backend class")

    import dataclasses

    @dataclasses.dataclass
    class _IrOpPriorityConfig:
        rms_norm: Any = None
        fused_add_rms_norm: Any = None

        @classmethod
        def with_default(cls, default, **kwargs):
            instance = cls()
            instance.default = list(default)
            for key, value in kwargs.items():
                setattr(instance, key, list(value))
            return instance

    vllm_pkg = _make_module("vllm")
    logger_mod = _make_module("vllm.logger", init_logger=lambda name: logger)
    platforms_mod = _make_module(
        "vllm.platforms", Platform=_BasePlatform, PlatformEnum=_PlatformEnum
    )
    interface_mod = _make_module(
        "vllm.platforms.interface", DeviceCapability=_Capability
    )
    rocm_mod = _make_module("vllm.platforms.rocm", RocmPlatform=_RocmPlatform)
    registry_mod = _make_module(
        "vllm.v1.attention.backends.registry",
        AttentionBackendEnum=_AttentionBackendEnum,
    )
    config_mod = _make_module(
        "vllm.config",
        CUDAGraphMode=SimpleNamespace(NONE="none", PIECEWISE="piecewise"),
    )
    compilation_mod = _make_module(
        "vllm.config.compilation", CompilationMode=SimpleNamespace(NONE="NONE")
    )
    kernel_mod = _make_module(
        "vllm.config.kernel", IrOpPriorityConfig=_IrOpPriorityConfig
    )
    ir_mod = _make_module("vllm.ir")
    ir_op_mod = _make_module("vllm.ir.op", IrOp=_IrOp)
    ir_ops_mod = _make_module(
        "vllm.ir.ops",
        rms_norm=_IrOp(["native", "vllm_c"]),
        fused_add_rms_norm=_IrOp(["native", "vllm_c"]),
    )
    ir_mod.op = ir_op_mod
    ir_mod.ops = ir_ops_mod

    utils_stub = _make_module("vllm_fl.utils")
    utils_stub.DeviceInfo = _DeviceInfoStub
    utils_stub._env_vendor_name = lambda: "nvidia"
    utils_stub.get_device_control_env_var = lambda vendor: "CUDA_VISIBLE_DEVICES"
    utils_stub.get_device_type = lambda vendor: "cuda"

    stubs = {
        "vllm": vllm_pkg,
        "vllm.logger": logger_mod,
        "vllm.platforms": platforms_mod,
        "vllm.platforms.interface": interface_mod,
        "vllm.platforms.rocm": rocm_mod,
        "vllm.v1.attention.backends.registry": registry_mod,
        "vllm.config": config_mod,
        "vllm.config.compilation": compilation_mod,
        "vllm.config.kernel": kernel_mod,
        "vllm.ir": ir_mod,
        "vllm.ir.op": ir_op_mod,
        "vllm.ir.ops": ir_ops_mod,
        "vllm_fl.utils": utils_stub,
    }
    for name, mod in stubs.items():
        sys.modules[name] = mod

    spec = importlib.util.spec_from_file_location(_PLATFORM_MODULE, _PLATFORM_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PLATFORM_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_PLATFORM_MODULE, None)
        for name in _STUB_ENTRIES:
            if saved[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved[name]
        raise
    yield SimpleNamespace(module=module, logger=logger, stubs=stubs, base=_BasePlatform)
    sys.modules.pop(_PLATFORM_MODULE, None)
    for name in _STUB_ENTRIES:
        if saved[name] is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved[name]


def _install_lazy_stubs(entries):
    """Install method-lazy stub modules; return an undo callable.

    The names given here are never part of ``_STUB_ENTRIES`` (the fixture only
    manages the module-level vllm graph), so a plain dict write plus manual
    restore is the safe pattern -- the suite's critical lesson forbids
    ``monkeypatch.setitem`` only for names a fixture's teardown also restores,
    but keeping the same manual discipline everywhere avoids any doubt.
    """
    saved = {name: sys.modules.get(name) for name in entries}
    for name, mod in entries.items():
        sys.modules[name] = mod
    return saved


def _restore_lazy_stubs(saved):
    for name, old in saved.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


def _vendor_patch_stub(fullname, marker, calls):
    """A ``vllm_fl.dispatch.backends.vendor.*`` stub recording the marker."""
    return _make_module(fullname, MARK=marker, CALLS=calls)


class TestGetAttnBackendCls:
    """``get_attn_backend_cls`` routes through the dispatch system."""

    def _dispatch_stub(self, calls):
        stubs = {}
        for name in (
            "vllm_fl",
            "vllm_fl.dispatch",
        ):
            stubs[name] = _make_module(name)
        dispatch = stubs["vllm_fl.dispatch"]

        def call_op(op_name, *args, **kwargs):
            calls.append((op_name, args, kwargs))
            return "stub.backend.FLAttention"

        dispatch.call_op = call_op
        return stubs

    def test_dispatches_via_call_op(self, platform_env):
        """The backend path is whatever ``call_op('attention_backend')`` returns."""
        cls = platform_env.module.PlatformFL
        calls = []
        saved = _install_lazy_stubs(self._dispatch_stub(calls))
        try:
            selector = SimpleNamespace(use_mla=True, use_sparse=False)
            result = cls.get_attn_backend_cls("selected", selector)
        finally:
            _restore_lazy_stubs(saved)
        assert result == "stub.backend.FLAttention"
        assert calls == [
            ("attention_backend", (), {"use_mla": True, "use_sparse": False})
        ]

    def test_amd_non_sparse_dispatches(self, platform_env, monkeypatch):
        """AMD without sparse MLA still goes through dispatch."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        calls = []
        saved = _install_lazy_stubs(self._dispatch_stub(calls))
        try:
            selector = SimpleNamespace(use_mla=False, use_sparse=False)
            result = cls.get_attn_backend_cls("selected", selector)
        finally:
            _restore_lazy_stubs(saved)
        assert result == "stub.backend.FLAttention"
        assert calls == [
            ("attention_backend", (), {"use_mla": False, "use_sparse": False})
        ]

    def test_logging_records_backend(self, platform_env):
        """Both the once-logger and the plain logger see the backend path."""
        cls = platform_env.module.PlatformFL
        calls = []
        saved = _install_lazy_stubs(self._dispatch_stub(calls))
        try:
            cls.get_attn_backend_cls(
                "selected", SimpleNamespace(use_mla=False, use_sparse=False)
            )
        finally:
            _restore_lazy_stubs(saved)
        levels = {level for level, _ in platform_env.logger.records}
        assert {"info", "info_once"} <= levels
        assert any(
            "stub.backend.FLAttention" in msg for _, msg in platform_env.logger.records
        )


class TestImportKernelsVendorBranches:
    """``import_kernels`` MetaX and MUSA branches."""

    def test_metax_imports_mcoplib_and_patches(self, platform_env, monkeypatch):
        """MetaX imports both mcoplib extensions and the patch package."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "metax", raising=False)
        monkeypatch.setattr(cls, "device_type", "cuda", raising=False)
        imported = []

        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            # Only intercept the vendor names the source imports; everything
            # else (stdlib, pytest machinery running mid-test) must go
            # through the real importer -- a blank ModuleType in sys.modules
            # for e.g. ``inspect`` breaks monkeypatch teardown and pytest's
            # own failure reporting.
            if not (
                name.startswith("mcoplib")
                or name == "vllm_fl.dispatch.backends.vendor.metax.patches"
            ):
                return real_import(name, *args, **kwargs)
            imported.append(name)
            module = _make_module(name)
            sys.modules[name] = module
            return module

        monkeypatch.setattr(builtins, "__import__", fake_import)
        try:
            cls.import_kernels()
        finally:
            # Always restore the importer, even when the call under test
            # raises: leaving the fake installed poisons every later import
            # in the process (including pytest's failure reporting).
            builtins.__import__ = real_import
        assert "mcoplib._C" in imported
        assert "mcoplib._moe_C" in imported
        assert "vllm_fl.dispatch.backends.vendor.metax.patches" in imported

    def test_metax_mcoplib_failure_warns(self, platform_env, monkeypatch):
        """A failing mcoplib import logs a warning per extension."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "metax", raising=False)
        monkeypatch.setattr(cls, "device_type", "cuda", raising=False)

        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.startswith("mcoplib"):
                raise ImportError("no mcoplib")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        try:
            cls.import_kernels()
        finally:
            builtins.__import__ = real_import
        assert any("mcoplib._C" in msg for _, msg in platform_env.logger.records)
        assert any("mcoplib._moe_C" in msg for _, msg in platform_env.logger.records)

    def test_metax_patch_failure_warns(self, platform_env, monkeypatch):
        """A failing metax patch import is reported with its exception."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "metax", raising=False)
        monkeypatch.setattr(cls, "device_type", "cuda", raising=False)

        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "vllm_fl.dispatch.backends.vendor.metax.patches":
                raise RuntimeError("circular import")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        try:
            cls.import_kernels()
        finally:
            builtins.__import__ = real_import
        assert any("maca patches" in msg for _, msg in platform_env.logger.records)

    def test_musa_applies_patches(self, platform_env, monkeypatch):
        """The MUSA branch imports and runs ``apply_musa_patches``."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "mthreads", raising=False)
        monkeypatch.setattr(cls, "device_type", "musa", raising=False)
        calls = []

        patch_mod = _make_module(
            "vllm_fl.dispatch.backends.vendor.musa.patch",
            apply_musa_patches=lambda: calls.append("applied"),
        )
        entries = {
            "vllm_fl.dispatch.backends.vendor.musa.patch": patch_mod,
        }
        saved = _install_lazy_stubs(entries)
        try:
            cls.import_kernels()
        finally:
            _restore_lazy_stubs(saved)
        assert calls == ["applied"]
        assert platform_env.base.super_calls  # non-metax delegates to base

    def test_musa_patch_failure_warns(self, platform_env, monkeypatch):
        """A failing MUSA patch application logs the failure."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "mthreads", raising=False)
        monkeypatch.setattr(cls, "device_type", "musa", raising=False)

        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "vllm_fl.dispatch.backends.vendor.musa.patch":
                raise ImportError("no musa patch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        try:
            cls.import_kernels()
        finally:
            builtins.__import__ = real_import
        assert any("MUSA patches" in msg for _, msg in platform_env.logger.records)


class TestApplyConfigPlatformDefaultsMetax:
    """MetaX graph defaults in ``apply_config_platform_defaults``."""

    def _metax(self, platform_env, monkeypatch):
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "metax", raising=False)
        monkeypatch.setattr(cls, "device_type", "cuda", raising=False)
        return cls

    def _patch_entries(self, ensure_result=None, calls=None):
        if calls is None:
            calls = []
        if ensure_result is None:
            ensure = lambda: calls.append("ensure") or {}
        else:
            ensure = lambda: calls.append("ensure") or ensure_result
        patches_pkg = _make_module(
            "vllm_fl.dispatch.backends.vendor.metax.patches",
            ensure_patches=ensure,
        )
        return {"vllm_fl.dispatch.backends.vendor.metax.patches": patches_pkg}

    def test_failed_patches_warned(self, platform_env, monkeypatch):
        """``ensure_patches`` failures are logged with their names."""
        cls = self._metax(platform_env, monkeypatch)
        error = RuntimeError("boom")
        saved = _install_lazy_stubs(
            self._patch_entries(ensure_result={"gdn_linear_attn": error})
        )
        try:
            cfg = SimpleNamespace(compilation_config=SimpleNamespace(mode="NONE"))
            cls.apply_config_platform_defaults(cfg)
        finally:
            _restore_lazy_stubs(saved)
        assert any(
            level == "warning" and "gdn_linear_attn" in msg
            for level, msg in platform_env.logger.records
        )

    def test_missing_compilation_config_returns(self, platform_env, monkeypatch):
        """A config without compilation_config stops before graph defaults."""
        cls = self._metax(platform_env, monkeypatch)
        calls = []
        saved = _install_lazy_stubs(self._patch_entries(calls=calls))
        try:
            cls.apply_config_platform_defaults(SimpleNamespace())
        finally:
            _restore_lazy_stubs(saved)
        assert calls == ["ensure"]

    def test_none_mode_returns_early(self, platform_env, monkeypatch):
        """CompilationMode.NONE keeps the eager run untouched."""
        cls = self._metax(platform_env, monkeypatch)
        saved = _install_lazy_stubs(self._patch_entries())
        try:
            cfg = SimpleNamespace(compilation_config=SimpleNamespace(mode="NONE"))
            cls.apply_config_platform_defaults(cfg)
        finally:
            _restore_lazy_stubs(saved)
        assert cfg.compilation_config.mode == "NONE"
        assert not hasattr(cfg.compilation_config, "cudagraph_mode")

    def test_explicit_cudagraph_mode_wins(self, platform_env, monkeypatch):
        """An explicit cudagraph_mode is not overridden."""
        cls = self._metax(platform_env, monkeypatch)
        saved = _install_lazy_stubs(self._patch_entries())
        try:
            cfg = SimpleNamespace(
                compilation_config=SimpleNamespace(
                    mode="PIECEWISE", cudagraph_mode="FULL"
                )
            )
            cls.apply_config_platform_defaults(cfg)
        finally:
            _restore_lazy_stubs(saved)
        assert cfg.compilation_config.cudagraph_mode == "FULL"
        assert not hasattr(cfg.compilation_config, "cudagraph_capture_sizes")

    def test_piecewise_defaults_installed(self, platform_env, monkeypatch):
        """PIECEWISE with no explicit mode gets capture sizes and guards."""
        cls = self._metax(platform_env, monkeypatch)
        cfg = SimpleNamespace(
            compilation_config=SimpleNamespace(
                mode="PIECEWISE",
                cudagraph_mode=None,
                cudagraph_capture_sizes=None,
                cudagraph_mm_encoder=True,
                pass_config=SimpleNamespace(fuse_rope_kvcache_cat_mla=True),
            )
        )
        saved = _install_lazy_stubs(self._patch_entries())
        try:
            cls.apply_config_platform_defaults(cfg)
        finally:
            _restore_lazy_stubs(saved)
        cc = cfg.compilation_config
        assert cc.cudagraph_mode == "piecewise"
        assert cc.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 32, 64, 128, 256]
        assert cc.cudagraph_mm_encoder is False
        assert cc.pass_config.fuse_rope_kvcache_cat_mla is False

    def test_existing_capture_sizes_kept(self, platform_env, monkeypatch):
        """Preconfigured capture sizes are left alone."""
        cls = self._metax(platform_env, monkeypatch)
        cfg = SimpleNamespace(
            compilation_config=SimpleNamespace(
                mode="PIECEWISE",
                cudagraph_mode=None,
                cudagraph_capture_sizes=[512],
                pass_config=None,
            )
        )
        saved = _install_lazy_stubs(self._patch_entries())
        try:
            cls.apply_config_platform_defaults(cfg)
        finally:
            _restore_lazy_stubs(saved)
        assert cfg.compilation_config.cudagraph_capture_sizes == [512]

    def test_kunlunxin_forces_eager(self, platform_env, monkeypatch):
        """The kunlunxin branch sets CompilationMode.NONE and returns."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_type", "cuda", raising=False)
        monkeypatch.setattr(cls, "vendor_name", "kunlunxin", raising=False)
        stub_compilation = sys.modules["vllm.config.compilation"]
        modes = SimpleNamespace(NONE="NONE")
        monkeypatch.setattr(stub_compilation, "CompilationMode", modes)
        cfg = SimpleNamespace(compilation_config=SimpleNamespace(mode="PIECEWISE"))
        cls.apply_config_platform_defaults(cfg)
        assert cfg.compilation_config.mode == "NONE"


class TestVitAttentionBackendMatrix:
    """Remaining ViT backend selection branches."""

    def _attention_stubs(self):
        utils_mod = _make_module(
            "vllm_fl.attention.utils", patch_mm_encoder_attention=lambda: None
        )
        attention_pkg = _make_module("vllm_fl.attention")
        attention_pkg.utils = utils_mod
        return {
            "vllm_fl.attention": attention_pkg,
            "vllm_fl.attention.utils": utils_mod,
        }

    def _gcu_stubs(self, head_ok=True, dtype_ok=True):
        class _GcuBackend:
            MARK = "gcu"

            @staticmethod
            def supports_head_size(head_size):
                return head_ok

            @staticmethod
            def supports_dtype(dtype):
                return dtype_ok

        attention_mod = _make_module(
            "vllm_fl.dispatch.backends.vendor.gcu.impl.attention",
            AttentionGCUBackend=_GcuBackend,
        )
        return {
            "vllm_fl.dispatch.backends.vendor.gcu.impl.attention": attention_mod,
        }, _GcuBackend

    def test_supported_amd_rocm_import_fails(self, platform_env, monkeypatch):
        """AMD without an importable RocmPlatform falls back to TORCH_SDPA."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        # The fixture installs the rocm stub; make the import fail by hand
        # and restore before the fixture's own teardown.
        saved = sys.modules["vllm.platforms.rocm"]
        sys.modules["vllm.platforms.rocm"] = None
        try:
            assert cls.get_supported_vit_attn_backends() == ["torch_sdpa"]
        finally:
            sys.modules["vllm.platforms.rocm"] = saved
        assert any(
            "RocmPlatform unavailable" in msg for _, msg in platform_env.logger.records
        )

    def test_vit_gcu_explicit_unsupported_raises(self, platform_env, monkeypatch):
        """GCU with an explicit FLASH_ATTN but bad head size raises."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "gcu", raising=False)
        gcu_entries, _ = self._gcu_stubs(head_ok=False)
        entries = {**self._attention_stubs(), **gcu_entries}
        saved = _install_lazy_stubs(entries)
        try:
            with pytest.raises(ValueError, match="Enflame vision FlashAttention"):
                cls.get_vit_attn_backend(8, torch.float16, "flash_attn")
        finally:
            _restore_lazy_stubs(saved)

    def test_vit_gcu_explicit_supported(self, platform_env, monkeypatch):
        """GCU with a supported explicit backend returns it unchanged."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "gcu", raising=False)
        gcu_entries, _ = self._gcu_stubs()
        saved = _install_lazy_stubs({**self._attention_stubs(), **gcu_entries})
        try:
            result = cls.get_vit_attn_backend(64, torch.bfloat16, "flash_attn")
        finally:
            _restore_lazy_stubs(saved)
        assert result == "flash_attn"
        assert any(level == "info_once" for level, _ in platform_env.logger.records)

    def test_vit_gcu_default_probe_flash(self, platform_env, monkeypatch):
        """GCU default probing picks FLASH_ATTN when supported."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "enflame", raising=False)
        gcu_entries, _ = self._gcu_stubs()
        saved = _install_lazy_stubs({**self._attention_stubs(), **gcu_entries})
        try:
            result = cls.get_vit_attn_backend(64, torch.float16)
        finally:
            _restore_lazy_stubs(saved)
        assert result == "flash_attn"

    def test_vit_gcu_default_probe_sdpa(self, platform_env, monkeypatch):
        """GCU default probing falls back to TORCH_SDPA when unsupported."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "enflame", raising=False)
        gcu_entries, _ = self._gcu_stubs(dtype_ok=False)
        saved = _install_lazy_stubs({**self._attention_stubs(), **gcu_entries})
        try:
            result = cls.get_vit_attn_backend(64, torch.float32)
        finally:
            _restore_lazy_stubs(saved)
        assert result == "torch_sdpa"

    def test_vit_amd_rocm_import_fails(self, platform_env, monkeypatch):
        """AMD default selection with RocmPlatform unavailable falls back."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        saved = _install_lazy_stubs(self._attention_stubs())
        rocm_saved = sys.modules["vllm.platforms.rocm"]
        sys.modules["vllm.platforms.rocm"] = None
        try:
            result = cls.get_vit_attn_backend(64, torch.bfloat16)
        finally:
            sys.modules["vllm.platforms.rocm"] = rocm_saved
            _restore_lazy_stubs(saved)
        assert result == "torch_sdpa"
        assert any(
            "defaulting AMD ViT attn backend" in msg
            for _, msg in platform_env.logger.records
        )

    def test_vit_cc8_flash_import_fails(self, platform_env, monkeypatch):
        """cc>=8 with a failing FLASH_ATTN import falls back to TORCH_SDPA."""
        cls = platform_env.module.PlatformFL
        # A stubbed capability keeps the probe off the real torch.cuda
        # (absent on CPU-only hosts); major 8 enters the FLASH_ATTN branch.
        # Swap the enum stub for one whose members raise ImportError from
        # get_class(), covering the except-ImportError fallback.
        monkeypatch.setattr(
            cls, "get_device_capability", lambda device_id=0: _CapabilityStub()
        )
        # platform.py bound AttentionBackendEnum at import time; patch the
        # module attribute it imported into its own namespace.
        monkeypatch.setattr(
            platform_env.module,
            "AttentionBackendEnum",
            SimpleNamespace(TORCH_SDPA="torch_sdpa", FLASH_ATTN=_Member("flash_attn")),
        )
        saved = _install_lazy_stubs(self._attention_stubs())
        try:
            result = cls.get_vit_attn_backend(64, torch.bfloat16)
        finally:
            _restore_lazy_stubs(saved)
        assert result == "torch_sdpa"

    def test_vit_low_capability_sdpa(self, platform_env, monkeypatch):
        """Below cc 8 the default selection is TORCH_SDPA directly."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "get_device_capability", lambda device_id=0: None)
        saved = _install_lazy_stubs(self._attention_stubs())
        try:
            assert cls.get_vit_attn_backend(64, torch.bfloat16) == "torch_sdpa"
        finally:
            _restore_lazy_stubs(saved)


class TestStaticGraphWrapperAmd:
    """``get_static_graph_wrapper_cls`` AMD branch."""

    def test_amd_v2_runner_uses_upstream(self, platform_env, monkeypatch):
        """AMD + V2 model runner returns the upstream CUDAGraphWrapper."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        config_mod = platform_env.stubs["vllm.config"]
        monkeypatch.setattr(
            config_mod,
            "get_current_vllm_config_or_none",
            lambda: SimpleNamespace(use_v2_model_runner=True),
            raising=False,
        )
        assert (
            cls.get_static_graph_wrapper_cls()
            == "vllm.compilation.cuda_graph.CUDAGraphWrapper"
        )

    def test_amd_v1_runner_uses_fl(self, platform_env, monkeypatch):
        """AMD on the FL runner keeps the platform GraphWrapper."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        config_mod = platform_env.stubs["vllm.config"]
        monkeypatch.setattr(
            config_mod,
            "get_current_vllm_config_or_none",
            lambda: SimpleNamespace(use_v2_model_runner=False),
            raising=False,
        )
        assert cls.get_static_graph_wrapper_cls().endswith("graph.GraphWrapper")

    def test_amd_no_config_uses_fl(self, platform_env, monkeypatch):
        """AMD with no current config keeps the platform GraphWrapper."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        config_mod = platform_env.stubs["vllm.config"]
        monkeypatch.setattr(
            config_mod,
            "get_current_vllm_config_or_none",
            lambda: None,
            raising=False,
        )
        assert cls.get_static_graph_wrapper_cls().endswith("graph.GraphWrapper")


class TestAlignHybridBlockSizeAscend:
    """``_align_hybrid_block_size`` Ascend branch."""

    def _entries(self, uses_dense, calls):
        hybrid_mod = _make_module(
            "vllm_fl.kv_cache.ascend.hybrid",
            align_hybrid_block_size=lambda cfg: calls.append(("align", cfg)),
            uses_dense_hybrid_cache=lambda model_cfg: (
                calls.append(("probe", model_cfg)) or uses_dense
            ),
        )
        return {"vllm_fl.kv_cache.ascend.hybrid": hybrid_mod}

    def test_ascend_dense_cache_aligns(self, platform_env, monkeypatch):
        """A dense hybrid cache delegates to the Ascend aligner."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "ascend", raising=False)
        calls = []
        cfg = SimpleNamespace(model_config="model-cfg")
        saved = _install_lazy_stubs(self._entries(True, calls))
        try:
            cls._align_hybrid_block_size(cfg, "backend")
        finally:
            _restore_lazy_stubs(saved)
        assert calls == [("probe", "model-cfg"), ("align", cfg)]
        assert "_align_hybrid_block_size" not in platform_env.base.super_calls

    def test_ascend_sparse_cache_defers_to_super(self, platform_env, monkeypatch):
        """A sparse hybrid cache keeps the base-class alignment."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "ascend", raising=False)
        calls = []
        platform_env.base.super_calls.clear()
        saved = _install_lazy_stubs(self._entries(False, calls))
        try:
            cls._align_hybrid_block_size(
                SimpleNamespace(model_config="model-cfg"), "backend"
            )
        finally:
            _restore_lazy_stubs(saved)
        assert "_align_hybrid_block_size" in platform_env.base.super_calls

    def test_non_ascend_defers_to_super(self, platform_env, monkeypatch):
        """Non-Ascend vendors use the base alignment directly."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "nvidia", raising=False)
        platform_env.base.super_calls.clear()
        cls._align_hybrid_block_size(SimpleNamespace(), "backend")
        assert "_align_hybrid_block_size" in platform_env.base.super_calls


class TestPreRegisterAndUpdate:
    """``pre_register_and_update`` per-vendor branches."""

    def test_npu_no_env_quiet(self, platform_env, monkeypatch):
        """NPU without the env var stays quiet and still forces 0."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "device_name", "npu", raising=False)
        monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
        ascend_pkg = _make_module("vllm_fl.dispatch.backends.vendor.ascend")
        detached = {
            k: v for k, v in os.environ.items() if k != "VLLM_USE_BREAKABLE_CUDAGRAPH"
        }
        monkeypatch.setattr(os, "environ", detached)
        saved = _install_lazy_stubs(
            {"vllm_fl.dispatch.backends.vendor.ascend": ascend_pkg}
        )
        try:
            cls.pre_register_and_update()
        finally:
            _restore_lazy_stubs(saved)
        assert detached["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"
        assert not any(
            level == "warning_once" for level, _ in platform_env.logger.records
        )

    def test_iluvatar_patches_applied(self, platform_env, monkeypatch):
        """Iluvatar runs the chained-or and MoE router patches."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "iluvatar", raising=False)
        monkeypatch.setattr(cls, "device_name", "cuda", raising=False)
        calls = []
        iluvatar_mod = _make_module(
            "vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar",
            patch_triton_chained_or_for_iluvatar=lambda: calls.append("chained-or"),
        )
        router_mod = _make_module(
            "vllm_fl.dispatch.backends.vendor.iluvatar.moe_router",
            patch_moe_router=lambda: calls.append("moe-router"),
        )
        saved = _install_lazy_stubs(
            {
                "vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar": iluvatar_mod,
                "vllm_fl.dispatch.backends.vendor.iluvatar.moe_router": router_mod,
            }
        )
        try:
            cls.pre_register_and_update()
        finally:
            _restore_lazy_stubs(saved)
        assert calls == ["chained-or", "moe-router"]

    def test_amd_disables_rocm_kernels(self, platform_env, monkeypatch):
        """AMD vendors get the ROCm native-kernel toggle applied."""
        cls = platform_env.module.PlatformFL
        monkeypatch.setattr(cls, "vendor_name", "amd", raising=False)
        monkeypatch.setattr(cls, "device_name", "cuda", raising=False)
        detached = dict(os.environ)
        monkeypatch.setattr(os, "environ", detached)
        monkeypatch.setattr(iu, "find_spec", lambda name: None)
        cls.pre_register_and_update()
        assert detached.get("VLLM_ROCM_USE_SKINNY_GEMM") == "0"


class TestVllmCAvailable:
    """``_vllm_c_available`` import-error branch."""

    def test_finder_raises_false(self, platform_env, monkeypatch):
        """An ImportError/ValueError from the finder resolves to False."""

        def boom(name):
            raise ValueError("parent not in sys.modules")

        monkeypatch.setattr(iu, "find_spec", boom)
        assert platform_env.module._vllm_c_available() is False

    def test_finder_import_error_false(self, platform_env, monkeypatch):
        """An ImportError from the finder also resolves to False."""

        def boom(name):
            raise ImportError("no vllm._C")

        monkeypatch.setattr(iu, "find_spec", boom)
        assert platform_env.module._vllm_c_available() is False


class TestUseFlagcxCallable:
    """``_use_flagcx`` callable check on the kunlunxin import path."""

    def test_kunlunxin_non_callable_false(self, platform_env, monkeypatch):
        """A non-callable FLAGCXLibrary keeps flagcx off."""
        mod = platform_env.module
        monkeypatch.delenv("FLAGCX_PATH", raising=False)
        flagcx_api = _make_module("flagcx.api", FLAGCXLibrary="not-callable")
        saved = _install_lazy_stubs({"flagcx.api": flagcx_api})
        try:
            assert mod._use_flagcx("kunlunxin") is False
        finally:
            _restore_lazy_stubs(saved)

    def test_kunlunxin_callable_true(self, platform_env, monkeypatch):
        """A callable FLAGCXLibrary switches flagcx on."""
        mod = platform_env.module
        monkeypatch.delenv("FLAGCX_PATH", raising=False)

        def library():
            return None

        flagcx_api = _make_module("flagcx.api", FLAGCXLibrary=library)
        saved = _install_lazy_stubs({"flagcx.api": flagcx_api})
        try:
            assert mod._use_flagcx("kunlunxin") is True
        finally:
            _restore_lazy_stubs(saved)


class TestVllmCIsRegisteredProviderEdges:
    """``_vllm_c_is_registered_provider`` failure shapes."""

    def test_fields_type_error_false(self, platform_env, monkeypatch):
        """A non-dataclass config makes fields() raise TypeError -> False."""
        kernel_mod = platform_env.stubs["vllm.config.kernel"]
        saved_attr = kernel_mod.IrOpPriorityConfig
        kernel_mod.IrOpPriorityConfig = _NotADataclass
        try:
            assert (
                platform_env.module.PlatformFL._vllm_c_is_registered_provider() is False
            )
        finally:
            kernel_mod.IrOpPriorityConfig = saved_attr

    def test_empty_fields_false(self, platform_env, monkeypatch):
        """A dataclass with no fields answers False."""
        kernel_mod = platform_env.stubs["vllm.config.kernel"]
        saved_attr = kernel_mod.IrOpPriorityConfig

        class _Empty:
            pass

        import dataclasses

        kernel_mod.IrOpPriorityConfig = dataclasses.dataclass(_Empty)
        try:
            assert (
                platform_env.module.PlatformFL._vllm_c_is_registered_provider() is False
            )
        finally:
            kernel_mod.IrOpPriorityConfig = saved_attr

    def test_non_irop_attribute_false(self, platform_env, monkeypatch):
        """An op entry that is not an IrOp answers False."""
        # Rename a field so getattr(ir_ops, ...) misses and hits a non-IrOp.
        ir_ops = platform_env.stubs["vllm.ir.ops"]
        saved_op = ir_ops.rms_norm
        ir_ops.rms_norm = "not-an-op"
        try:
            assert (
                platform_env.module.PlatformFL._vllm_c_is_registered_provider() is False
            )
        finally:
            ir_ops.rms_norm = saved_op

    def test_impls_access_exception_false(self, platform_env, monkeypatch):
        """An impls membership that raises answers False."""
        ir_ops = platform_env.stubs["vllm.ir.ops"]

        class _ExplodingImpls:
            def __contains__(self, item):
                raise RuntimeError("impls broken")

        class _BrokenOp:
            impls = _ExplodingImpls()

        saved_op = ir_ops.rms_norm
        ir_ops.rms_norm = _BrokenOp()
        try:
            assert (
                platform_env.module.PlatformFL._vllm_c_is_registered_provider() is False
            )
        finally:
            ir_ops.rms_norm = saved_op

    def test_all_impls_present_true(self, platform_env):
        """Every field carrying vllm_c answers True."""
        assert platform_env.module.PlatformFL._vllm_c_is_registered_provider() is True


class TestImportFailureTolerated:
    """Module-level helper behaviours on the loaded copy."""

    def test_class_body_error_logged_when_no_vendor(self, monkeypatch):
        """A fully empty vendor probe logs the class-body error line."""
        monkeypatch.delenv("GEMS_VENDOR", raising=False)
        monkeypatch.delenv("FLAGCX_PATH", raising=False)
        logger = _FakeLogger()
        saved = {name: sys.modules.get(name) for name in _STUB_ENTRIES}

        class _Capability:
            major = 8
            minor = 0

        class _PlatformEnum:
            CUDA = "cuda"
            OOT = "oot"
            CPU = "cpu"

        class _BasePlatform:
            dist_backend = "nccl"

            @classmethod
            def get_device_capability(cls, device_id=0):
                return _Capability()

        class _EmptyInfo:
            vendor_name = ""
            device_type = ""
            device = SimpleNamespace(__class__=SimpleNamespace(__name__="NoDevice"))
            dispatch_key = "CUDA"
            torch_device_fn = torch.cuda

        vllm_pkg = _make_module("vllm")
        logger_mod = _make_module("vllm.logger", init_logger=lambda name: logger)
        platforms_mod = _make_module(
            "vllm.platforms", Platform=_BasePlatform, PlatformEnum=_PlatformEnum
        )
        interface_mod = _make_module(
            "vllm.platforms.interface", DeviceCapability=_Capability
        )
        rocm_mod = _make_module("vllm.platforms.rocm")
        registry_mod = _make_module(
            "vllm.v1.attention.backends.registry",
            AttentionBackendEnum=SimpleNamespace(TORCH_SDPA="t", FLASH_ATTN="f"),
        )
        config_mod = _make_module("vllm.config", CUDAGraphMode=SimpleNamespace())
        compilation_mod = _make_module(
            "vllm.config.compilation", CompilationMode=SimpleNamespace()
        )
        kernel_mod = _make_module(
            "vllm.config.kernel", IrOpPriorityConfig=SimpleNamespace()
        )
        ir_mod = _make_module("vllm.ir")
        ir_op_mod = _make_module("vllm.ir.op", IrOp=_IrOp)
        ir_ops_mod = _make_module("vllm.ir.ops")
        utils_stub = _make_module("vllm_fl.utils")
        utils_stub.DeviceInfo = _EmptyInfo
        utils_stub._env_vendor_name = lambda: ""
        utils_stub.get_device_control_env_var = lambda vendor: "CUDA_VISIBLE_DEVICES"
        utils_stub.get_device_type = lambda vendor: "cuda"

        stubs = {
            "vllm": vllm_pkg,
            "vllm.logger": logger_mod,
            "vllm.platforms": platforms_mod,
            "vllm.platforms.interface": interface_mod,
            "vllm.platforms.rocm": rocm_mod,
            "vllm.v1.attention.backends.registry": registry_mod,
            "vllm.config": config_mod,
            "vllm.config.compilation": compilation_mod,
            "vllm.config.kernel": kernel_mod,
            "vllm.ir": ir_mod,
            "vllm.ir.op": ir_op_mod,
            "vllm.ir.ops": ir_ops_mod,
            "vllm_fl.utils": utils_stub,
        }
        for name, mod in stubs.items():
            sys.modules[name] = mod

        module_name = _PLATFORM_MODULE + "_empty_vendor"
        spec = importlib.util.spec_from_file_location(module_name, _PLATFORM_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
            for name in _STUB_ENTRIES:
                if saved[name] is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = saved[name]
        assert any(
            level == "error" and "No vendor/device_type resolved" in msg
            for level, msg in logger.records
        )
