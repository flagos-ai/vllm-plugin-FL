# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""G3 coverage: PlatformFL config hooks in the 814b platform.py snapshot.

Covers ``apply_config_platform_defaults`` (MetaX graph defaults) and the
block-size / multimodal-prefix branches of ``check_and_update_config``. Heavy
vendor patch modules are replaced through ``sys.modules`` fakes; the real vLLM
configuration enums are used so the comparisons exercise production values.
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

from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationMode

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
    PlatformFL = _PLATFORM_MOD.PlatformFL
except Exception as _exc:  # pragma: no cover - environment safety net
    _PLATFORM_MOD = None
    PlatformFL = None
    _PLATFORM_ERROR = _exc


@pytest.fixture(autouse=True)
def _require_platform():
    if PlatformFL is None:
        pytest.fail(f"vllm_fl.platform load failed: {_PLATFORM_ERROR!r}")


@pytest.fixture
def restore_identity(monkeypatch):
    for name in ("vendor_name", "device_type", "device_name", "dist_backend"):
        monkeypatch.setattr(PlatformFL, name, getattr(PlatformFL, name), raising=False)
    return monkeypatch


def _install_metax_patches(monkeypatch, failed):
    fake = types.ModuleType("vllm_fl.dispatch.backends.vendor.metax.patches")
    fake.ensure_patches = lambda: set(failed)
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.metax.patches", fake
    )
    return fake


# ---------------------------------------------------------------------------
# apply_config_platform_defaults
# ---------------------------------------------------------------------------


def test_apply_defaults_non_metax_is_noop(restore_identity):
    PlatformFL.vendor_name = "nvidia"
    config = NS(compilation_config=NS(mode="untouched"))
    PlatformFL.apply_config_platform_defaults(config)
    assert config.compilation_config.mode == "untouched"


def test_apply_defaults_missing_compilation_config_returns(
    restore_identity, monkeypatch
):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ())
    # No compilation_config attribute -> getattr(..., None) -> return.
    PlatformFL.apply_config_platform_defaults(NS())


def test_apply_defaults_warns_on_failed_patches(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ("leftover",))
    PlatformFL.apply_config_platform_defaults(NS())


def test_apply_defaults_explicit_eager_mode_returns(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ())
    compilation = NS(mode=CompilationMode.NONE, cudagraph_mode=None)
    PlatformFL.apply_config_platform_defaults(NS(compilation_config=compilation))
    assert compilation.cudagraph_mode is None


def test_apply_defaults_explicit_cudagraph_mode_wins(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ())
    compilation = NS(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
    )
    PlatformFL.apply_config_platform_defaults(NS(compilation_config=compilation))
    assert compilation.cudagraph_mode == CUDAGraphMode.PIECEWISE


def test_apply_defaults_installs_piecewise_defaults(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ())
    pass_config = NS(fuse_rope_kvcache_cat_mla=True)
    compilation = NS(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=None,
        cudagraph_capture_sizes=None,
        cudagraph_mm_encoder=True,
        pass_config=pass_config,
    )
    PlatformFL.apply_config_platform_defaults(NS(compilation_config=compilation))

    assert compilation.cudagraph_mode == CUDAGraphMode.PIECEWISE
    assert compilation.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert compilation.cudagraph_mm_encoder is False
    assert pass_config.fuse_rope_kvcache_cat_mla is False


# ---------------------------------------------------------------------------
# check_and_update_config branches
# ---------------------------------------------------------------------------


def _base_vllm_config():
    return NS(
        use_v2_model_runner=False,
        model_config=NS(use_mla=False, is_mm_prefix_lm=False, is_moe=False),
        parallel_config=NS(
            all2all_backend="allgather_reducescatter",
            data_parallel_size=1,
            tensor_parallel_size=1,
        ),
        scheduler_config=NS(is_multimodal_model=False, disable_chunked_mm_input=False),
        cache_config=NS(block_size=16),
        compilation_config=NS(
            compile_sizes=[],
            cudagraph_mode=CUDAGraphMode.NONE,
            mode=CompilationMode.NONE,
        ),
    )


def test_mm_prefix_lm_forces_chunked_mm_input_off(restore_identity):
    PlatformFL.vendor_name = "nvidia"
    PlatformFL.device_type = "cuda"
    config = _base_vllm_config()
    config.model_config.is_mm_prefix_lm = True
    config.scheduler_config.is_multimodal_model = True
    config.scheduler_config.disable_chunked_mm_input = False

    PlatformFL.check_and_update_config(config)

    assert config.scheduler_config.disable_chunked_mm_input is True
    assert config.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"


def _install_ascend_fakes(monkeypatch):
    refresh_calls = []
    compile_calls = []

    patch_mod = types.ModuleType("vllm_fl.dispatch.backends.vendor.ascend.patch")
    patch_mod.refresh_block_size = lambda cfg: refresh_calls.append(cfg)
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.ascend.patch", patch_mod
    )

    comp_mod = types.ModuleType(
        "vllm_fl.dispatch.backends.vendor.ascend.patches.compilation"
    )
    comp_mod._configure_ascend_compilation = lambda cfg: compile_calls.append(cfg)
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.ascend.patches.compilation",
        comp_mod,
    )
    return refresh_calls, compile_calls


def test_npu_default_block_size_is_128(restore_identity, monkeypatch):
    PlatformFL.vendor_name = "ascend"
    PlatformFL.device_type = "npu"
    refresh_calls, compile_calls = _install_ascend_fakes(monkeypatch)

    config = _base_vllm_config()
    config.model_config = None
    config.cache_config.block_size = None

    PlatformFL.check_and_update_config(config)

    assert config.cache_config.block_size == 128
    assert refresh_calls == [config]
    assert compile_calls == [config]


def test_kunlunxin_default_block_size_is_128(restore_identity):
    PlatformFL.vendor_name = "kunlunxin"
    PlatformFL.device_type = "cuda"
    config = _base_vllm_config()
    config.cache_config.block_size = None

    PlatformFL.check_and_update_config(config)

    assert config.cache_config.block_size == 128


def test_musa_default_block_size_is_64(restore_identity):
    PlatformFL.vendor_name = "mthreads"
    PlatformFL.device_type = "musa"
    config = _base_vllm_config()
    config.model_config = None
    config.cache_config.block_size = None
    config.compilation_config.inductor_compile_config = {}

    PlatformFL.check_and_update_config(config)

    assert config.cache_config.block_size == 64
    # MUSA also disables combo kernels at this hook.
    assert config.compilation_config.inductor_compile_config["combo_kernels"] is False
