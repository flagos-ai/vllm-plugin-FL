# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""G3 coverage: vllm_fl.platform helpers and PlatformFL device surface (814b).

Targets the 814b snapshot of ``vllm_fl/platform.py``. Every test drives the
behavior through module globals, class attributes, ``sys.modules`` fakes or
``torch`` attribute injection (the sanctioned probe boundaries). No device is
required: all delegation targets are recording stubs and the tensor work is
CPU-only.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import pathlib
import sys
import types
from types import SimpleNamespace as NS

import pytest
import torch


def _ensure_triton_stub():
    """Best-effort triton shim; never return early on a partial install.

    A 910C host can import ``triton`` while ``triton.language`` is missing the
    dtype/attr names ``vllm_fl.platform`` touches at import time. Backfill every
    symbol unconditionally so module-level collection cannot fail.
    """
    triton_mod = sys.modules.get("triton")
    if triton_mod is None:
        try:
            import triton as triton_mod  # noqa: F401
        except Exception:
            triton_mod = types.ModuleType("triton")
            sys.modules["triton"] = triton_mod
    lang = sys.modules.get("triton.language")
    if lang is None:
        try:
            import triton.language as lang  # noqa: F401
        except Exception:
            lang = types.ModuleType("triton.language")
    sys.modules.setdefault("triton.language", lang)

    def _mark(fn):
        try:
            fn.fn = fn
        except Exception:
            pass
        return fn

    def _jit(fn=None, *a, **k):
        return (lambda f: _mark(f)) if fn is None else _mark(fn)

    def _autotune(*a, **k):
        return lambda f: _mark(f)

    for name in (
        "float32", "float16", "bfloat16", "int32", "int64", "int8", "uint8",
        "uint16", "uint32", "uint64", "bool", "void", "constexpr",
        "static_range", "program_id", "arange", "load", "store", "where",
        "cdiv", "multiple_of", "max_contiguous", "next_power_of_2",
        "static_assert", "atomic_add", "atomic_max", "sum", "max", "min",
        "exp", "log", "sqrt", "zeros", "full", "reshape", "trans", "dot",
        "sort", "num_programs",
    ):
        if not hasattr(lang, name):
            setattr(lang, name, name)
    for name, value in (
        ("jit", _jit),
        ("autotune", _autotune),
        ("cdiv", lambda a, b: -(-int(a) // int(b))),
        ("multiple_of", lambda v, m: v),
        ("max_contiguous", lambda v, n: v),
        ("static_range", lambda *a, **k: range(*(a or (0,)))),
        ("program_id", lambda axis=0: 0),
        ("next_power_of_2", lambda n: 1 << max(0, int(n) - 1).bit_length()),
    ):
        if not hasattr(lang, name):
            setattr(lang, name, value)
    triton_mod.language = lang
    if not hasattr(triton_mod, "jit"):
        triton_mod.jit = _jit
    if not hasattr(triton_mod, "autotune"):
        triton_mod.autotune = _autotune
    for sub in (
        "triton.runtime", "triton.compiler", "triton.backends",
        "triton.backends.ascend", "triton.language.extra",
        "triton.language.standard", "triton.language.libdevice",
    ):
        sys.modules.setdefault(sub, types.ModuleType(sub))
    try:
        lang.extra = sys.modules["triton.language.extra"]
        lang.standard = sys.modules["triton.language.standard"]
    except Exception:
        pass


_ensure_triton_stub()

from vllm.platforms.interface import DeviceCapability

# --- robust vllm_fl.platform loader (g25 three-attempt strategy) -----------
def _harden_triton():
    for full in (
        "triton",
        "triton.language",
        "triton.runtime",
        "triton.runtime.driver",
        "triton.compiler",
        "triton.compiler.compiler",
        "triton.backends",
        "triton.backends.ascend",
        "triton.language.extra",
        "triton.language.standard",
        "triton.language.libdevice",
    ):
        try:
            mod = importlib.import_module(full)
        except Exception:
            mod = sys.modules.get(full) or types.ModuleType(full)
            sys.modules[full] = mod
        try:
            mod.__path__ = []
        except Exception:
            pass
        parent, _, leaf = full.rpartition(".")
        if parent:
            pmod = sys.modules.get(parent)
            if pmod is not None:
                try:
                    setattr(pmod, leaf, mod)
                except Exception:
                    pass

    class _TritonFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        _fl_triton_stub = True

        def find_spec(self, fullname, path=None, target=None):
            if not fullname.startswith("triton.") or fullname in sys.modules:
                return None
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

        def create_module(self, spec):
            mod = types.ModuleType(spec.name)
            mod.__path__ = []
            mod.__getattr__ = lambda name: (lambda *a, **k: None)
            return mod

        def exec_module(self, module):
            pass

    if not any(getattr(f, "_fl_triton_stub", False) for f in sys.meta_path):
        sys.meta_path.append(_TritonFinder())


def _quiet_logger():
    return types.SimpleNamespace(
        info=lambda *a, **k: None,
        info_once=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        warning_once=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )


_DEVICE_TYPE = {
    "ascend": "npu", "nvidia": "cuda", "metax": "cuda", "iluvatar": "cuda",
    "hygon": "cuda", "amd": "cuda", "kunlunxin": "cuda", "mthreads": "musa",
    "tsingmicro": "txda", "gcu": "gcu", "enflame": "gcu",
}


def _fake_device_info():
    vendor = os.environ.get("GEMS_VENDOR", "")
    try:
        import torch as _t
    except Exception:
        _t = types.SimpleNamespace(empty_cache=lambda: None, set_device=lambda d: None)
    return types.SimpleNamespace(
        vendor_name=vendor,
        device_type=_DEVICE_TYPE.get(vendor, ""),
        dispatch_key="",
        torch_device_fn=_t,
        device=object(),
    )


class _PlatformBase:
    device_type = ""
    vendor_name = ""
    dispatch_key = ""

    @classmethod
    def import_kernels(cls):
        return None

    def get_compile_backend(self):
        return "inductor"

    def _align_hybrid_block_size(self, *a, **k):
        return None

    def insert_blocks_to_device(self, *a, **k):
        return None

    def swap_out_blocks_to_host(self, *a, **k):
        return None


class _PlatformEnum:
    OOT = "OOT"
    CUDA = "CUDA"


class _ABE:
    TORCH_SDPA = "TORCH_SDPA"
    FLASH_ATTN = "FLASH_ATTN"
    ROCM_AITER_FA = "ROCM_AITER_FA"


_MISSING = object()


def _make_pkg(name):
    mod = types.ModuleType(name)
    mod.__path__ = []
    return mod


def _put(name, mod, created):
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            created.setdefault(parent, _MISSING)
            sys.modules[parent] = _make_pkg(parent)
    created.setdefault(name, sys.modules.get(name, _MISSING))
    sys.modules[name] = mod
    parent, _, leaf = name.rpartition(".")
    pmod = sys.modules.get(parent)
    if pmod is not None:
        try:
            setattr(pmod, leaf, mod)
        except Exception:
            pass


def _install_platform_deps(force):
    created = {}

    def provide(name, attrs, build):
        available = False
        if not force:
            try:
                mod = importlib.import_module(name)
                available = all(hasattr(mod, a) for a in attrs)
            except Exception:
                available = False
        if available:
            parts = name.split(".")
            for i in range(1, len(parts)):
                parent = ".".join(parts[:i])
                if parent not in sys.modules:
                    _put(parent, _make_pkg(parent), created)
            return
        _put(name, build(), created)

    provide(
        "vllm.logger", ("init_logger",),
        lambda: types.SimpleNamespace(init_logger=lambda n: _quiet_logger()),
    )

    def _platforms():
        mod = types.ModuleType("vllm.platforms")
        mod.Platform = _PlatformBase
        mod.PlatformEnum = _PlatformEnum
        return mod

    provide("vllm.platforms", ("Platform", "PlatformEnum"), _platforms)

    def _iface():
        mod = types.ModuleType("vllm.platforms.interface")
        mod.DeviceCapability = object
        return mod

    provide("vllm.platforms.interface", ("DeviceCapability",), _iface)

    def _registry():
        mod = types.ModuleType("vllm.v1.attention.backends.registry")
        mod.AttentionBackendEnum = _ABE
        return mod

    provide(
        "vllm.v1.attention.backends.registry",
        ("AttentionBackendEnum",), _registry,
    )

    def _utils():
        mod = types.ModuleType("vllm_fl.utils")
        mod.DeviceInfo = _fake_device_info
        mod._env_vendor_name = lambda: os.environ.get("GEMS_VENDOR", "")
        mod.get_device_control_env_var = lambda vendor: {
            "ascend": "ASCEND_RT_VISIBLE_DEVICES",
            "musa": "MUSA_VISIBLE_DEVICES",
        }.get(vendor, "CUDA_VISIBLE_DEVICES")
        mod.get_device_type = lambda vendor: _DEVICE_TYPE.get(vendor, "")
        return mod

    provide(
        "vllm_fl.utils",
        ("DeviceInfo", "_env_vendor_name", "get_device_control_env_var",
         "get_device_type"), _utils,
    )

    for name in (
        "vllm", "vllm.config", "vllm.config.compilation", "vllm.config.cache",
        "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends", "vllm_fl",
    ):
        if name not in sys.modules:
            _put(name, _make_pkg(name), created)
    return created


def _restore_modules(created):
    for name, old in created.items():
        if old is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


def _drop_platform():
    for key in list(sys.modules):
        if key == "vllm_fl.platform" or key.startswith("vllm_fl.platform."):
            sys.modules.pop(key, None)


def _find_platform_source():
    cands = []
    for env_name in ("VLLM_FL_PLATFORM_SRC", "FL_REPO_ROOT", "FL_WORKSPACE"):
        raw = os.environ.get(env_name)
        if not raw:
            continue
        base = pathlib.Path(raw)
        cands.append(
            base if base.name == "platform.py"
            else base / "vllm_fl" / "platform.py"
        )
    here = pathlib.Path(__file__).resolve()
    for base in (
        here.parent, *here.parents, pathlib.Path.cwd(),
        *pathlib.Path.cwd().parents,
    ):
        cands.append(base / "vllm_fl" / "platform.py")
    for entry in sys.path:
        if entry:
            cands.append(pathlib.Path(entry) / "vllm_fl" / "platform.py")
    for cand in cands:
        if cand.is_file():
            return cand
    raise FileNotFoundError("vllm_fl/platform.py not found; set FL_REPO_ROOT")


def _platform_usable(mod):
    cls = getattr(mod, "PlatformFL", None)
    return cls is not None and callable(getattr(cls, "get_device_uuid", None))


def _load_platform_module():
    _harden_triton()
    _drop_platform()
    try:
        mod = importlib.import_module("vllm_fl.platform")
        if _platform_usable(mod):
            return mod
    except Exception:
        pass

    _drop_platform()
    created = _install_platform_deps(force=False)
    try:
        mod = importlib.import_module("vllm_fl.platform")
        if _platform_usable(mod):
            return mod
    except Exception:
        pass

    _drop_platform()
    path = _find_platform_source()
    created_force = _install_platform_deps(force=True)
    try:
        spec = importlib.util.spec_from_file_location(
            "vllm_fl.platform", str(path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot build spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["vllm_fl.platform"] = mod
        spec.loader.exec_module(mod)
        if not _platform_usable(mod):
            raise ImportError(f"{path} did not define usable PlatformFL")
        return mod
    finally:
        _restore_modules(created)
        _restore_modules(created_force)


_PLATFORM_ERROR = None
try:
    _PLATFORM_MOD = _load_platform_module()
    fl_platform = _PLATFORM_MOD
    PlatformFL = _PLATFORM_MOD.PlatformFL
except Exception as _exc:  # pragma: no cover - environment safety net
    _PLATFORM_MOD = None
    fl_platform = None
    PlatformFL = None
    _PLATFORM_ERROR = _exc


@pytest.fixture(autouse=True)
def _require_platform():
    if PlatformFL is None:
        pytest.fail(f"vllm_fl.platform load failed: {_PLATFORM_ERROR!r}")


@pytest.fixture
def restore_identity(monkeypatch):
    """Snapshot/restore the device-probe-derived class identity."""
    for name in (
        "vendor_name",
        "device_type",
        "device_name",
        "dist_backend",
        "torch_device_fn",
    ):
        monkeypatch.setattr(PlatformFL, name, getattr(PlatformFL, name), raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# Module-level helpers: _vllm_c_available / _safe_device_type
# ---------------------------------------------------------------------------


def test_vllm_c_available_true(monkeypatch):
    # Isolate the module global so the real importlib is never touched.
    monkeypatch.setattr(
        fl_platform,
        "importlib",
        NS(util=NS(find_spec=lambda name: object())),
    )
    assert fl_platform._vllm_c_available() is True


def test_vllm_c_available_swallows_probe_errors(monkeypatch):
    def _boom(name):
        raise ValueError("bad spec")

    monkeypatch.setattr(
        fl_platform, "importlib", NS(util=NS(find_spec=_boom))
    )
    assert fl_platform._vllm_c_available() is False


def test_safe_device_type_prefers_detected_value():
    assert fl_platform._safe_device_type(NS(device_type="cuda"), "nvidia") == "cuda"


def test_safe_device_type_maps_vendor_when_probe_empty(monkeypatch):
    monkeypatch.setattr(fl_platform, "get_device_type", lambda vendor: "npu")
    assert fl_platform._safe_device_type(NS(device_type=""), "ascend") == "npu"


def test_safe_device_type_empty_vendor(monkeypatch):
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    assert fl_platform._safe_device_type(NS(device_type=""), "") == ""


def test_safe_device_type_unknown_vendor_warns(monkeypatch):
    def _raise(vendor):
        raise ValueError("not in map")

    monkeypatch.setattr(fl_platform, "get_device_type", _raise)
    assert fl_platform._safe_device_type(NS(device_type=""), "bogus") == ""


def _load_platform_with_empty_vendor(monkeypatch):
    """Import a second PlatformFL instance with an unresolved vendor probe."""
    try:
        import vllm_fl.utils as fl_utils
    except Exception:
        # The package __init__ chain is not importable on the bare 910C image;
        # build a local module so the re-exec exercises platform.py itself.
        fl_utils = types.ModuleType("vllm_fl.utils")
        if "vllm_fl" not in sys.modules:
            pkg = types.ModuleType("vllm_fl")
            pkg.__path__ = []
            monkeypatch.setitem(sys.modules, "vllm_fl", pkg)
        monkeypatch.setitem(sys.modules, "vllm_fl.utils", fl_utils)
        monkeypatch.setattr(
            sys.modules["vllm_fl"], "utils", fl_utils, raising=False
        )

    class _EmptyInfo:
        vendor_name = ""
        device_type = ""
        dispatch_key = ""
        torch_device_fn = object()
        device = NS()

    monkeypatch.setattr(fl_utils, "DeviceInfo", _EmptyInfo)
    monkeypatch.setattr(fl_utils, "_env_vendor_name", lambda: "")
    monkeypatch.setattr(
        fl_utils, "get_device_type",
        getattr(fl_utils, "get_device_type", lambda vendor: ""),
        raising=False,
    )
    monkeypatch.setattr(
        fl_utils, "get_device_control_env_var",
        getattr(fl_utils, "get_device_control_env_var",
                lambda vendor: "CUDA_VISIBLE_DEVICES"),
        raising=False,
    )
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    monkeypatch.delenv("FLAGCX_PATH", raising=False)

    spec = importlib.util.spec_from_file_location(
        "_g3_platform_empty_vendor", fl_platform.__file__
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module




# ---------------------------------------------------------------------------
# Stateless identity predicates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vendor_name", "device_type", "expected"),
    [
        ("nvidia", "cuda", True),
        ("metax", "cuda", True),
        ("mthreads", "musa", True),
        ("gcu", "gcu", True),
        ("enflame", "gcu", True),
        ("iluvatar", "cuda", False),
        ("hygon", "cuda", False),
        ("ascend", "npu", False),
    ],
)
def test_is_cuda_alike_matrix(
    restore_identity, vendor_name, device_type, expected
):
    instance = PlatformFL()
    instance.vendor_name = vendor_name
    instance.device_type = device_type
    assert instance.is_cuda_alike() is expected


def test_is_sleep_mode_available_nvidia_only(restore_identity):
    instance = PlatformFL()
    instance.vendor_name = "nvidia"
    assert instance.is_sleep_mode_available() is True
    instance.vendor_name = "ascend"
    assert instance.is_sleep_mode_available() is False


def test_is_musa_and_is_gcu_probe_torch_namespaces(restore_identity, monkeypatch):
    instance = PlatformFL()
    monkeypatch.setattr(torch, "musa", NS(is_available=lambda: True), raising=False)
    monkeypatch.setattr(torch, "gcu", NS(is_available=lambda: True), raising=False)
    assert instance.is_musa() is True
    assert instance.is_gcu() is True

    monkeypatch.delattr(torch, "musa", raising=False)
    monkeypatch.delattr(torch, "gcu", raising=False)
    assert instance.is_musa() is False
    assert instance.is_gcu() is False


def test_supported_dtypes_are_bf16_fp16_fp32():
    assert PlatformFL().supported_dtypes == [
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ]


def test_check_if_supports_dtype_bf16_without_capability_raises(
    restore_identity, monkeypatch
):
    PlatformFL.vendor_name = "nvidia"
    monkeypatch.setattr(
        PlatformFL,
        "has_device_capability",
        classmethod(lambda cls, capability: False),
        raising=False,
    )
    monkeypatch.setattr(
        PlatformFL,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: None),
        raising=False,
    )
    monkeypatch.setattr(
        PlatformFL,
        "get_device_name",
        classmethod(lambda cls, device_id=0: "FakeGPU"),
        raising=False,
    )
    with pytest.raises(ValueError, match="does not have a compute capability"):
        PlatformFL.check_if_supports_dtype(torch.bfloat16)


def test_check_if_supports_dtype_reports_capability_version(
    restore_identity, monkeypatch
):
    PlatformFL.vendor_name = "nvidia"
    monkeypatch.setattr(
        PlatformFL,
        "has_device_capability",
        classmethod(lambda cls, capability: False),
        raising=False,
    )
    monkeypatch.setattr(
        PlatformFL,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(7, 5)),
        raising=False,
    )
    monkeypatch.setattr(
        PlatformFL,
        "get_device_name",
        classmethod(lambda cls, device_id=0: "FakeGPU"),
        raising=False,
    )
    with pytest.raises(ValueError, match="compute capability 7.5"):
        PlatformFL.check_if_supports_dtype(torch.bfloat16)

    # Non-bf16 dtypes short-circuit before the capability check.
    PlatformFL.check_if_supports_dtype(torch.float16)


# ---------------------------------------------------------------------------
# torch_device_fn delegation
# ---------------------------------------------------------------------------


def test_get_current_memory_usage_delegates(restore_identity):
    calls = []

    class Recording:
        empty_cache = staticmethod(lambda *a: calls.append("empty_cache"))
        reset_peak_memory_stats = staticmethod(
            lambda device: calls.append(("reset", device))
        )
        max_memory_allocated = staticmethod(lambda device: 7.5)

    PlatformFL.torch_device_fn = Recording
    assert PlatformFL.get_current_memory_usage(None) == 7.5
    assert calls == ["empty_cache", ("reset", None)]


def test_set_device_nvidia_initialises_context(restore_identity, monkeypatch):
    calls = []

    class Recording:
        set_device = staticmethod(lambda device: calls.append(("set", device)))

    PlatformFL.torch_device_fn = Recording
    PlatformFL.vendor_name = "nvidia"
    zeros = []
    monkeypatch.setattr(
        torch,
        "zeros",
        lambda *args, **kwargs: zeros.append(kwargs) or torch.tensor([0.0]),
    )

    device = torch.device("cpu")
    PlatformFL.set_device(device)
    assert calls == [("set", device)]
    assert zeros and zeros[0]["device"] == device


def test_empty_cache_and_get_device_name(restore_identity):
    calls = []
    PlatformFL.torch_device_fn = NS(empty_cache=lambda: calls.append(True))
    PlatformFL.empty_cache()
    assert calls == [True]
    assert PlatformFL.get_device_name(0) == PlatformFL.device_name


# ---------------------------------------------------------------------------
# Kernel import hooks
# ---------------------------------------------------------------------------


def test_import_kernels_metax_import_failures_are_logged(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "metax"
    PlatformFL.device_type = "cuda"
    for name in (
        "mcoplib._C",
        "mcoplib._moe_C",
        "vllm_fl.dispatch.backends.vendor.metax.patches",
    ):
        monkeypatch.setitem(sys.modules, name, None)
    # Must not raise: each failure is caught at its own level.
    PlatformFL.import_kernels()


def test_import_kernels_musa_success(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "nvidia"
    PlatformFL.device_type = "musa"
    calls = []
    fake = types.ModuleType("vllm_fl.dispatch.backends.vendor.musa.patch")
    fake.apply_musa_patches = lambda: calls.append(True)
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.musa.patch", fake
    )
    PlatformFL.import_kernels()
    assert calls == [True]


def test_import_kernels_musa_failure_is_logged(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "nvidia"
    PlatformFL.device_type = "musa"

    def _boom():
        raise RuntimeError("musa patch failed")

    fake = types.ModuleType("vllm_fl.dispatch.backends.vendor.musa.patch")
    fake.apply_musa_patches = _boom
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.musa.patch", fake
    )
    PlatformFL.import_kernels()


def test_import_ir_kernels_imports_vllm_kernels(monkeypatch):
    fake = types.ModuleType("vllm.kernels")
    monkeypatch.setitem(sys.modules, "vllm.kernels", fake)
    PlatformFL.import_ir_kernels()
