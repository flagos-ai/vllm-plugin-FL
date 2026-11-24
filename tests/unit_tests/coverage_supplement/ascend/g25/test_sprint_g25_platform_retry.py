# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""G25 retry: make ``vllm_fl/platform.py`` collectable and executed on 910C.

G12 skipped at module level because ``vllm_fl.platform`` failed to import on the
bare torch_npu image.  This file never calls ``pytest.skip(allow_module_level)``:

1. A complete ``triton`` / ``triton.language`` stub is installed first.
2. ``vllm_fl.platform`` is imported for real.
3. Missing ``vllm`` symbols are injected into ``sys.modules`` and import retried.
4. If it still fails, ``platform.py`` is loaded straight from source with
   ``importlib.util.spec_from_file_location`` against fully stubbed deps.

Only the pure-Python branches of ``PlatformFL`` are exercised; no accelerator is
touched.  Everything is local filesystem / in-process.
"""

from __future__ import annotations

import enum
import importlib
import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


# ===========================================================================
# 1. Triton stub (must exist before touching vllm_fl)
# ===========================================================================
_LANG_ATTRS = (
    "float32", "float16", "bfloat16", "int32", "int64", "int8", "uint8",
    "uint16", "uint32", "uint64", "bool", "void", "constexpr", "static_range",
    "program_id", "arange", "load", "store", "where", "cdiv", "multiple_of",
    "max_contiguous", "next_power_of_2", "static_assert", "atomic_add",
    "atomic_max", "sum", "max", "min", "exp", "log", "sqrt", "zeros",
    "full", "reshape", "trans", "dot", "sort", "num_programs",
)


def _ensure_triton_stub() -> None:
    try:
        triton = importlib.import_module("triton")
    except Exception:
        triton = sys.modules.get("triton") or types.ModuleType("triton")
        sys.modules["triton"] = triton

    try:
        lang = importlib.import_module("triton.language")
    except Exception:
        lang = sys.modules.get("triton.language") or types.ModuleType(
            "triton.language"
        )
        sys.modules["triton.language"] = lang

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

    for name in _LANG_ATTRS:
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
    triton.language = lang
    if not hasattr(triton, "jit"):
        triton.jit = _jit
    if not hasattr(triton, "autotune"):
        triton.autotune = _autotune
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


# ===========================================================================
# 2. torch stub only when torch is genuinely absent (local dry-run support)
# ===========================================================================
def _ensure_torch_stub() -> None:
    try:
        import torch  # noqa: F401

        torch_mod = sys.modules["torch"]
        types_mod = getattr(torch_mod, "types", None)
        if types_mod is None or not hasattr(types_mod, "Device"):
            if types_mod is None:
                types_mod = types.ModuleType("torch.types")
                torch_mod.types = types_mod
                sys.modules.setdefault("torch.types", types_mod)
            types_mod.Device = object
        return
    except Exception:
        pass

    t = sys.modules.get("torch") or types.ModuleType("torch")
    t.dtype = getattr(t, "dtype", type("dtype", (), {}))
    t.Tensor = getattr(t, "Tensor", type("Tensor", (), {}))
    t.device = getattr(t, "device", type("device", (), {"__init__": lambda s, *a, **k: None}))
    for name in ("bfloat16", "float16", "float32", "float64", "int8",
                 "uint8", "int16", "int32", "int64", "bool"):
        setattr(t, name, name)
    t.types = sys.modules.get("torch.types") or types.ModuleType("torch.types")
    t.types.Device = getattr(t.types, "Device", object)
    sys.modules.setdefault("torch.types", t.types)
    t.cuda = getattr(t, "cuda", NS(is_available=lambda: False))
    t.zeros = getattr(t, "zeros", lambda *a, **k: None)
    t.manual_seed = getattr(t, "manual_seed", lambda *a, **k: None)
    sys.modules["torch"] = t


# ===========================================================================
# 3. Deterministic vllm-side fakes (also used by individual tests)
# ===========================================================================
class _CGMode(enum.Enum):
    NONE = "cg_none"
    PIECEWISE = "cg_piecewise"
    FULL = "cg_full"
    FULL_AND_PIECEWISE = "cg_full_and_piecewise"

    def has_full_cudagraphs(self):
        return self in (_CGMode.FULL, _CGMode.FULL_AND_PIECEWISE)

    def has_piecewise_cudagraphs(self):
        return self in (_CGMode.PIECEWISE, _CGMode.FULL_AND_PIECEWISE)


class _CompMode(enum.Enum):
    NONE = "cm_none"
    VLLM_COMPILE = "cm_vllm_compile"


class _PlatformEnum:
    OOT = "OOT"
    CUDA = "CUDA"


class _FakePlatform:
    device_type = ""
    vendor_name = ""
    dispatch_key = ""

    @classmethod
    def import_kernels(cls):  # noqa: D401 - base stub
        return None

    def get_compile_backend(self):
        return "inductor"

    def _align_hybrid_block_size(self, vllm_config, backend_cls):
        return None

    def insert_blocks_to_device(self, *a, **k):
        return None

    def swap_out_blocks_to_host(self, *a, **k):
        return None


class _BackendMember:
    def __init__(self, name, factory=None):
        self.name = name
        self._factory = factory or (lambda: object())

    def get_class(self):
        return self._factory()

    def __repr__(self):
        return f"<ABE {self.name}>"


class _FakeABE:
    TORCH_SDPA = _BackendMember("TORCH_SDPA")
    FLASH_ATTN = _BackendMember("FLASH_ATTN", lambda: _VitFlash)


class _VitFlash:
    @staticmethod
    def supports_head_size(head_size):
        return True

    @staticmethod
    def supports_dtype(dtype):
        return True


@dataclass
class _DeviceCapability:
    major: int
    minor: int

    def as_version_str(self):
        return f"{self.major}.{self.minor}"


def _quiet_logger():
    return NS(
        info=lambda *a, **k: None,
        info_once=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        warning_once=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )


def _fake_device_info():
    vendor = os.environ.get("GEMS_VENDOR", "")
    dev_type = ""
    if vendor:
        dev_type = {
            "ascend": "npu",
            "nvidia": "cuda",
            "metax": "cuda",
            "iluvatar": "cuda",
            "hygon": "cuda",
            "amd": "cuda",
            "kunlunxin": "cuda",
            "mthreads": "musa",
            "tsingmicro": "txda",
            "gcu": "gcu",
            "enflame": "gcu",
        }.get(vendor, "")
    try:
        import torch as _t
    except Exception:
        _t = NS(
            empty_cache=lambda: None,
            set_device=lambda d: None,
        )
    return NS(
        vendor_name=vendor,
        device_type=dev_type,
        dispatch_key="",
        torch_device_fn=_t,
        device=object(),
    )


def _vendor_device_type(vendor):
    return {
        "ascend": "npu",
        "nvidia": "cuda",
        "metax": "cuda",
        "iluvatar": "cuda",
        "hygon": "cuda",
        "amd": "cuda",
        "kunlunxin": "cuda",
        "mthreads": "musa",
        "tsingmicro": "txda",
        "gcu": "gcu",
        "enflame": "gcu",
    }.get(vendor, "")


# ===========================================================================
# 4. sys.modules stub installer
# ===========================================================================
_MISSING = object()


def _make_pkg(name):
    m = types.ModuleType(name)
    m.__path__ = []
    return m


def _ensure_parents(name, created):
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            m = _make_pkg(parent)
            created.setdefault(parent, _MISSING)
            sys.modules[parent] = m


def _put(name, mod, created):
    _ensure_parents(name, created)
    created.setdefault(name, sys.modules.get(name, _MISSING))
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and isinstance(sys.modules.get(parent), types.ModuleType):
        try:
            setattr(sys.modules[parent], child, mod)
        except Exception:
            pass


def _real_ok(name, attrs):
    try:
        mod = importlib.import_module(name)
    except Exception:
        return False
    return all(hasattr(mod, a) for a in attrs)


def _install_module_level_stubs(force=False):
    """Install module-level deps of platform.py; return restore map."""
    created = {}

    def provide(name, attrs, factory):
        if not force and _real_ok(name, attrs):
            _ensure_parents(name, created)
            return
        mod = factory()
        _put(name, mod, created)

    provide(
        "vllm.logger", ("init_logger",),
        lambda: types.SimpleNamespace(init_logger=lambda n: _quiet_logger()),
    )

    def _platforms():
        m = types.ModuleType("vllm.platforms")
        m.Platform = _FakePlatform
        m.PlatformEnum = _PlatformEnum
        return m

    provide("vllm.platforms", ("Platform", "PlatformEnum"), _platforms)

    def _platforms_iface():
        m = types.ModuleType("vllm.platforms.interface")
        m.DeviceCapability = _DeviceCapability
        return m

    provide(
        "vllm.platforms.interface", ("DeviceCapability",), _platforms_iface
    )

    def _abe_registry():
        m = types.ModuleType("vllm.v1.attention.backends.registry")
        m.AttentionBackendEnum = _FakeABE
        return m

    provide(
        "vllm.v1.attention.backends.registry",
        ("AttentionBackendEnum",),
        _abe_registry,
    )

    def _utils():
        m = types.ModuleType("vllm_fl.utils")
        m.DeviceInfo = lambda: _fake_device_info()
        m._env_vendor_name = lambda: os.environ.get("GEMS_VENDOR", "")
        m.get_device_control_env_var = lambda vendor: {
            "ascend": "ASCEND_RT_VISIBLE_DEVICES",
            "musa": "MUSA_VISIBLE_DEVICES",
        }.get(vendor, "CUDA_VISIBLE_DEVICES")
        m.get_device_type = _vendor_device_type
        return m

    provide(
        "vllm_fl.utils",
        ("DeviceInfo", "_env_vendor_name", "get_device_control_env_var",
         "get_device_type"),
        _utils,
    )

    # Make sure the parent packages exist for direct exec.
    for name in (
        "vllm", "vllm.config", "vllm.config.compilation", "vllm.config.cache",
        "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends",
        "vllm_fl",
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


def _drop_partial_platform():
    for key in list(sys.modules):
        if key == "vllm_fl.platform" or key.startswith("vllm_fl.platform."):
            sys.modules.pop(key, None)


def _find_platform_source():
    cands = []
    env = os.environ.get("VLLM_FL_PLATFORM_SRC")
    if env:
        cands.append(Path(env))
    here = Path(__file__).resolve()
    for base in (here.parent, *here.parents, Path.cwd(), *Path.cwd().parents):
        cands.append(base / "vllm_fl" / "platform.py")
    for entry in sys.path:
        if entry:
            cands.append(Path(entry) / "vllm_fl" / "platform.py")
    seen = set()
    for c in cands:
        try:
            key = str(c.resolve())
        except Exception:
            key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.is_file():
            return str(c)
    raise FileNotFoundError(
        "vllm_fl/platform.py not found; set VLLM_FL_PLATFORM_SRC"
    )


def _module_usable(mod):
    cls = getattr(mod, "PlatformFL", None)
    return cls is not None and callable(getattr(cls, "get_device_uuid", None))


def _load_platform_module():
    _ensure_triton_stub()
    _ensure_torch_stub()

    # Attempt 1: real import with triton stubbed.
    _drop_partial_platform()
    try:
        mod = importlib.import_module("vllm_fl.platform")
        if _module_usable(mod):
            return mod, "real"
    except Exception:
        pass

    # Attempt 2: inject missing vllm symbols, retry.
    _drop_partial_platform()
    created2 = _install_module_level_stubs(force=False)
    try:
        mod = importlib.import_module("vllm_fl.platform")
        if _module_usable(mod):
            return mod, "real-stubbed"
    except Exception:
        pass

    # Attempt 3: load platform.py straight from source with full stubs.
    _drop_partial_platform()
    path = _find_platform_source()
    created3 = _install_module_level_stubs(force=True)
    try:
        spec = importlib.util.spec_from_file_location("vllm_fl.platform", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot build spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["vllm_fl.platform"] = mod
        spec.loader.exec_module(mod)
        if not _module_usable(mod):
            raise ImportError(f"{path} did not define usable PlatformFL")
        return mod, "direct-source"
    finally:
        _restore_modules(created2)
        _restore_modules(created3)


_LOAD_ERROR = None
try:
    fl_platform, _LOAD_MODE = _load_platform_module()
    PlatformFL = fl_platform.PlatformFL
except Exception as exc:  # pragma: no cover - environment safety net
    fl_platform = None
    PlatformFL = None
    _LOAD_MODE = "failed"
    _LOAD_ERROR = exc


@pytest.fixture(autouse=True)
def _require_platform():
    if PlatformFL is None:
        pytest.fail(f"vllm_fl.platform load failed: {_LOAD_ERROR!r}")


# ===========================================================================
# 5. shared helpers
# ===========================================================================
_IDENTITY = ("vendor_name", "device_type", "device_name", "dist_backend")


@pytest.fixture
def restore(monkeypatch):
    for name in _IDENTITY:
        monkeypatch.setattr(
            PlatformFL, name, getattr(PlatformFL, name), raising=False
        )
    return monkeypatch


def _ensure_chain_mp(monkeypatch, full_name):
    parts = full_name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            mod = _make_pkg(parent)
            monkeypatch.setitem(sys.modules, parent, mod)


def _install_vllm_config_fakes(monkeypatch):
    _ensure_chain_mp(monkeypatch, "vllm.config.compilation")
    cfg = types.ModuleType("vllm.config")
    cfg.CUDAGraphMode = _CGMode
    cfg.get_current_vllm_config_or_none = lambda: None
    monkeypatch.setitem(sys.modules, "vllm.config", cfg)
    comp = types.ModuleType("vllm.config.compilation")
    comp.CompilationMode = _CompMode
    monkeypatch.setitem(sys.modules, "vllm.config.compilation", comp)
    cache = types.ModuleType("vllm.config.cache")
    cache.CacheDType = object
    monkeypatch.setitem(sys.modules, "vllm.config.cache", cache)
    _ensure_chain_mp(monkeypatch, "vllm.v1.attention.selector")
    sel = types.ModuleType("vllm.v1.attention.selector")
    sel.AttentionSelectorConfig = object
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.selector", sel)


@pytest.fixture(autouse=True)
def _vllm_cfg(monkeypatch):
    _install_vllm_config_fakes(monkeypatch)


def _quiet(monkeypatch):
    logger = _quiet_logger()
    monkeypatch.setattr(fl_platform, "logger", logger)
    return logger


def _install_metax_patches(monkeypatch, failed=()):
    name = "vllm_fl.dispatch.backends.vendor.metax.patches"
    _ensure_chain_mp(monkeypatch, name)
    fake = types.ModuleType(name)
    fake.ensure_patches = lambda: set(failed)
    monkeypatch.setitem(sys.modules, name, fake)
    return fake


def _install_attn_utils(monkeypatch):
    calls = []
    name = "vllm_fl.attention.utils"
    _ensure_chain_mp(monkeypatch, name)
    fake = types.ModuleType(name)
    fake.patch_mm_encoder_attention = lambda: calls.append(1)
    monkeypatch.setitem(sys.modules, name, fake)
    return calls


def _fake_pynvml(monkeypatch, uuid="GPU-UUID", p2p_status=0):
    mod = types.ModuleType("pynvml")
    mod.nvmlInit = lambda: None
    mod.nvmlShutdown = lambda: None
    mod.nvmlDeviceGetHandleByIndex = lambda i: ("handle", i)
    mod.nvmlDeviceGetUUID = lambda h: uuid
    mod.nvmlDeviceGetP2PStatus = lambda a, b, caps: p2p_status
    mod.NVML_P2P_CAPS_INDEX_NVLINK = 0
    mod.NVML_P2P_STATUS_OK = 0
    mod.NVMLError = type("NVMLError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "pynvml", mod)
    return mod


# ===========================================================================
# 6. apply_config_platform_defaults
# ===========================================================================
def test_apply_defaults_non_metax_noop(restore):
    PlatformFL.vendor_name = "nvidia"
    config = NS(compilation_config=NS(mode=_CompMode.NONE))
    PlatformFL.apply_config_platform_defaults(config)
    assert config.compilation_config.mode == _CompMode.NONE


def test_apply_defaults_missing_compilation_config(restore, monkeypatch):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ())
    PlatformFL.apply_config_platform_defaults(NS())


def test_apply_defaults_warns_failed_patches(restore, monkeypatch):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ("leftover",))
    _quiet(monkeypatch)
    PlatformFL.apply_config_platform_defaults(NS())


def test_apply_defaults_explicit_eager_returns(restore, monkeypatch):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ())
    compilation = NS(mode=_CompMode.NONE, cudagraph_mode=None)
    PlatformFL.apply_config_platform_defaults(NS(compilation_config=compilation))
    assert compilation.cudagraph_mode is None


def test_apply_defaults_explicit_cudagraph_wins(restore, monkeypatch):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ())
    compilation = NS(
        mode=_CompMode.VLLM_COMPILE, cudagraph_mode=_CGMode.PIECEWISE
    )
    PlatformFL.apply_config_platform_defaults(NS(compilation_config=compilation))
    assert compilation.cudagraph_mode == _CGMode.PIECEWISE


def test_apply_defaults_installs_piecewise(restore, monkeypatch):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ())
    pass_config = NS(fuse_rope_kvcache_cat_mla=True)
    compilation = NS(
        mode=_CompMode.VLLM_COMPILE,
        cudagraph_mode=None,
        cudagraph_capture_sizes=None,
        cudagraph_mm_encoder=True,
        pass_config=pass_config,
    )
    PlatformFL.apply_config_platform_defaults(NS(compilation_config=compilation))
    assert compilation.cudagraph_mode == _CGMode.PIECEWISE
    assert compilation.cudagraph_capture_sizes == [
        1, 2, 4, 8, 16, 32, 64, 128, 256
    ]
    assert compilation.cudagraph_mm_encoder is False
    assert pass_config.fuse_rope_kvcache_cat_mla is False


def test_apply_defaults_keeps_existing_capture_sizes(restore, monkeypatch):
    PlatformFL.vendor_name = "metax"
    _install_metax_patches(monkeypatch, ())
    compilation = NS(
        mode=_CompMode.VLLM_COMPILE,
        cudagraph_mode=None,
        cudagraph_capture_sizes=[1, 2],
    )
    PlatformFL.apply_config_platform_defaults(NS(compilation_config=compilation))
    assert compilation.cudagraph_capture_sizes == [1, 2]


# ===========================================================================
# 7. check_and_update_config
# ===========================================================================
def _base_vllm_config():
    return NS(
        use_v2_model_runner=False,
        speculative_config=None,
        attention_config=None,
        model_config=NS(
            use_mla=False,
            is_mm_prefix_lm=False,
            is_moe=False,
            is_diffusion=False,
            is_hybrid=False,
            model="m",
            verify_with_parallel_config=lambda pc: None,
        ),
        parallel_config=NS(
            all2all_backend="allgather_reducescatter",
            data_parallel_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            world_size_across_dp=None,
            disable_custom_all_reduce=False,
            worker_cls=None,
        ),
        scheduler_config=NS(
            is_multimodal_model=False,
            disable_chunked_mm_input=False,
            async_scheduling=False,
        ),
        cache_config=NS(block_size=16, mamba_cache_mode=None),
        compilation_config=NS(
            compile_sizes=[],
            cudagraph_mode=_CGMode.NONE,
            mode=_CompMode.NONE,
            backend="eager",
            inductor_compile_config={},
            use_inductor_graph_partition=False,
        ),
    )


def test_gcu_rejects_v2_runner(restore):
    PlatformFL.device_type = "gcu"
    PlatformFL.vendor_name = "gcu"
    config = _base_vllm_config()
    config.use_v2_model_runner = True
    with pytest.raises(ValueError, match="ModelRunnerFL"):
        PlatformFL.check_and_update_config(config)


def test_gcu_rejects_dspark(restore):
    PlatformFL.device_type = "gcu"
    PlatformFL.vendor_name = "gcu"
    config = _base_vllm_config()
    config.speculative_config = NS(method="dspark")
    with pytest.raises(ValueError, match="DSpark"):
        PlatformFL.check_and_update_config(config)


def test_gcu_rejects_dflash_mixed_kv(restore):
    PlatformFL.device_type = "gcu"
    PlatformFL.vendor_name = "gcu"
    config = _base_vllm_config()
    config.speculative_config = NS(method="dflash")
    config._dflash_needs_multi_kv_group = lambda: True
    config._is_dflash_candidate_draft = lambda: False
    with pytest.raises(ValueError, match="mixed-KV DFlash"):
        PlatformFL.check_and_update_config(config)


def test_gcu_rejects_diffusion(restore):
    PlatformFL.device_type = "gcu"
    PlatformFL.vendor_name = "gcu"
    config = _base_vllm_config()
    config.model_config.is_diffusion = True
    with pytest.raises(ValueError, match="diffusion"):
        PlatformFL.check_and_update_config(config)


def test_gcu_rejects_hybrid_align_spec(restore):
    PlatformFL.device_type = "gcu"
    PlatformFL.vendor_name = "gcu"
    config = _base_vllm_config()
    config.speculative_config = NS(method="mtp")
    config.model_config.is_hybrid = True
    config.cache_config.mamba_cache_mode = "align"
    with pytest.raises(ValueError, match="mamba_cache_mode='align'"):
        PlatformFL.check_and_update_config(config)


def test_gcu_inductor_combo_kernels_disabled(restore, monkeypatch):
    PlatformFL.device_type = "gcu"
    PlatformFL.vendor_name = "gcu"
    _install_attn_utils(monkeypatch)
    config = _base_vllm_config()
    config.compilation_config.backend = "inductor"
    config.compilation_config.mode = _CompMode.VLLM_COMPILE
    PlatformFL.check_and_update_config(config)
    icc = config.compilation_config.inductor_compile_config
    assert icc["combo_kernels"] is False
    assert icc["benchmark_combo_kernel"] is False
    assert config.parallel_config.disable_custom_all_reduce is True


def test_musa_rejects_v2_runner(restore):
    PlatformFL.device_type = "musa"
    PlatformFL.vendor_name = "mthreads"
    config = _base_vllm_config()
    config.use_v2_model_runner = True
    with pytest.raises(ValueError, match="MUSA requires ModelRunnerFL"):
        PlatformFL.check_and_update_config(config)


def test_txda_rejects_v2_runner(restore):
    PlatformFL.device_type = "txda"
    PlatformFL.vendor_name = "tsingmicro"
    config = _base_vllm_config()
    config.use_v2_model_runner = True
    with pytest.raises(ValueError, match="Tsingmicro TXDA requires ModelRunnerFL"):
        PlatformFL.check_and_update_config(config)


def test_txda_rejects_speculative(restore):
    PlatformFL.device_type = "txda"
    PlatformFL.vendor_name = "tsingmicro"
    config = _base_vllm_config()
    config.speculative_config = NS(method="mtp")
    with pytest.raises(ValueError, match="speculative"):
        PlatformFL.check_and_update_config(config)


def test_txda_rejects_context_parallel(restore):
    PlatformFL.device_type = "txda"
    PlatformFL.vendor_name = "tsingmicro"
    config = _base_vllm_config()
    config.parallel_config.prefill_context_parallel_size = 2
    with pytest.raises(ValueError, match="context parallel"):
        PlatformFL.check_and_update_config(config)


def test_txda_world_gt_one_requires_flagcx(restore):
    PlatformFL.device_type = "txda"
    PlatformFL.vendor_name = "tsingmicro"
    PlatformFL.dist_backend = "nccl"
    config = _base_vllm_config()
    config.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="FlagCX"):
        PlatformFL.check_and_update_config(config)


def test_txda_flagcx_multi_device_ok(restore):
    PlatformFL.device_type = "txda"
    PlatformFL.vendor_name = "tsingmicro"
    PlatformFL.dist_backend = "flagcx"
    config = _base_vllm_config()
    config.parallel_config.tensor_parallel_size = 2
    PlatformFL.check_and_update_config(config)
    assert config.parallel_config.worker_cls == "vllm_fl.worker.worker.WorkerFL"


def _install_ascend_fakes(monkeypatch, dsa_result=True):
    refresh_calls, compile_calls = [], []
    base = "vllm_fl.dispatch.backends.vendor.ascend"
    for full in (f"{base}.patch", f"{base}.patches.compilation",
                 f"{base}.patches.ascend_glm_dsa"):
        _ensure_chain_mp(monkeypatch, full)
    patch_mod = types.ModuleType(f"{base}.patch")
    patch_mod.refresh_block_size = lambda cfg: refresh_calls.append(cfg)
    monkeypatch.setitem(sys.modules, f"{base}.patch", patch_mod)
    comp_mod = types.ModuleType(f"{base}.patches.compilation")
    comp_mod._configure_ascend_compilation = lambda cfg: compile_calls.append(cfg)
    monkeypatch.setitem(sys.modules, f"{base}.patches.compilation", comp_mod)
    dsa_mod = types.ModuleType(f"{base}.patches.ascend_glm_dsa")
    dsa_mod.prepare_glm_dsa_dense_fallback = lambda mc: dsa_result
    monkeypatch.setitem(sys.modules, f"{base}.patches.ascend_glm_dsa", dsa_mod)
    return refresh_calls, compile_calls


def test_npu_glm_dsa_dense_fallback(restore, monkeypatch):
    PlatformFL.device_type = "npu"
    PlatformFL.vendor_name = "ascend"
    _install_ascend_fakes(monkeypatch, dsa_result=True)
    verified = []
    config = _base_vllm_config()
    config.model_config.verify_with_parallel_config = (
        lambda pc: verified.append(pc)
    )
    _quiet(monkeypatch)
    PlatformFL.check_and_update_config(config)
    assert verified == [config.parallel_config]


def test_npu_default_block_size_128(restore, monkeypatch):
    PlatformFL.device_type = "npu"
    PlatformFL.vendor_name = "ascend"
    refresh_calls, compile_calls = _install_ascend_fakes(monkeypatch)
    config = _base_vllm_config()
    config.model_config = None
    config.cache_config.block_size = None
    PlatformFL.check_and_update_config(config)
    assert config.cache_config.block_size == 128
    assert refresh_calls == [config]
    assert compile_calls == [config]


def test_kunlunxin_default_block_size_128(restore):
    PlatformFL.device_type = "cuda"
    PlatformFL.vendor_name = "kunlunxin"
    config = _base_vllm_config()
    config.cache_config.block_size = None
    PlatformFL.check_and_update_config(config)
    assert config.cache_config.block_size == 128


def test_musa_default_block_size_64(restore):
    PlatformFL.device_type = "musa"
    PlatformFL.vendor_name = "mthreads"
    config = _base_vllm_config()
    config.model_config = None
    config.cache_config.block_size = None
    PlatformFL.check_and_update_config(config)
    assert config.cache_config.block_size == 64
    assert (
        config.compilation_config.inductor_compile_config["combo_kernels"]
        is False
    )


def test_generic_default_block_size_16(restore):
    PlatformFL.device_type = "cuda"
    PlatformFL.vendor_name = "nvidia"
    config = _base_vllm_config()
    config.model_config = None
    config.cache_config.block_size = None
    PlatformFL.check_and_update_config(config)
    assert config.cache_config.block_size == 16


def test_mla_forces_block_size_64(restore):
    PlatformFL.device_type = "cuda"
    PlatformFL.vendor_name = "nvidia"
    config = _base_vllm_config()
    config.model_config.use_mla = True
    config.cache_config.block_size = 16
    PlatformFL.check_and_update_config(config)
    assert config.cache_config.block_size == 64


def test_compile_sizes_none_becomes_empty(restore):
    PlatformFL.device_type = "cuda"
    PlatformFL.vendor_name = "nvidia"
    config = _base_vllm_config()
    config.compilation_config.compile_sizes = None
    PlatformFL.check_and_update_config(config)
    assert config.compilation_config.compile_sizes == []


def test_txda_rejects_cudagraph_mode(restore):
    PlatformFL.device_type = "txda"
    PlatformFL.vendor_name = "tsingmicro"
    PlatformFL.dist_backend = "flagcx"
    config = _base_vllm_config()
    config.compilation_config.cudagraph_mode = _CGMode.PIECEWISE
    with pytest.raises(ValueError, match="graph mode"):
        PlatformFL.check_and_update_config(config)


def test_txda_rejects_compilation_mode(restore):
    PlatformFL.device_type = "txda"
    PlatformFL.vendor_name = "tsingmicro"
    PlatformFL.dist_backend = "flagcx"
    config = _base_vllm_config()
    config.compilation_config.mode = _CompMode.VLLM_COMPILE
    with pytest.raises(ValueError, match="eager execution only"):
        PlatformFL.check_and_update_config(config)


def test_iluvatar_configure_compilation(restore, monkeypatch):
    PlatformFL.device_type = "cuda"
    PlatformFL.vendor_name = "iluvatar"
    calls = []
    name = "vllm_fl.dispatch.backends.vendor.iluvatar.compilation"
    _ensure_chain_mp(monkeypatch, name)
    fake = types.ModuleType(name)
    fake.configure_compilation = lambda cc, mc: calls.append((cc, mc))
    monkeypatch.setitem(sys.modules, name, fake)
    config = _base_vllm_config()
    PlatformFL.check_and_update_config(config)
    assert calls and calls[0][0] is config.compilation_config
    assert (
        config.parallel_config.worker_cls
        == "vllm_fl.dispatch.backends.vendor.iluvatar.worker.worker."
        "IluvatarWorkerFL"
    )


def test_musa_full_cudagraph_downgraded(restore, monkeypatch):
    PlatformFL.device_type = "musa"
    PlatformFL.vendor_name = "mthreads"
    _quiet(monkeypatch)
    config = _base_vllm_config()
    config.model_config = None
    config.compilation_config.cudagraph_mode = _CGMode.FULL
    PlatformFL.check_and_update_config(config)
    assert config.compilation_config.cudagraph_mode == _CGMode.PIECEWISE


def test_deepep_disables_graphs(restore, monkeypatch):
    PlatformFL.device_type = "cuda"
    PlatformFL.vendor_name = "nvidia"
    _quiet(monkeypatch)
    config = _base_vllm_config()
    config.parallel_config.all2all_backend = "deepep_high_throughput"
    config.parallel_config.data_parallel_size = 2
    config.compilation_config.cudagraph_mode = _CGMode.PIECEWISE
    PlatformFL.check_and_update_config(config)
    assert config.compilation_config.cudagraph_mode == _CGMode.NONE


def test_kunlunxin_async_and_compile_disabled(restore, monkeypatch):
    PlatformFL.device_type = "cuda"
    PlatformFL.vendor_name = "kunlunxin"
    _quiet(monkeypatch)
    config = _base_vllm_config()
    config.scheduler_config.async_scheduling = True
    config.compilation_config.cudagraph_mode = _CGMode.PIECEWISE
    config.compilation_config.mode = _CompMode.VLLM_COMPILE
    PlatformFL.check_and_update_config(config)
    assert config.scheduler_config.async_scheduling is False
    assert config.compilation_config.cudagraph_mode == _CGMode.NONE
    assert config.compilation_config.mode == _CompMode.NONE


def test_metax_updates_attention_flags(restore):
    PlatformFL.device_type = "cuda"
    PlatformFL.vendor_name = "metax"
    config = _base_vllm_config()
    config.attention_config = NS(
        use_cudnn_prefill=True,
        use_trtllm_ragged_deepseek_prefill=True,
        use_trtllm_attention=True,
        disable_flashinfer_prefill=False,
    )
    PlatformFL.check_and_update_config(config)
    assert config.model_config.disable_cascade_attn is True
    ac = config.attention_config
    assert ac.use_cudnn_prefill is False
    assert ac.use_trtllm_ragged_deepseek_prefill is False
    assert ac.use_trtllm_attention is False
    assert ac.disable_flashinfer_prefill is True


class _SplittingMode:
    def __init__(self):
        self.splitting_ops = ["x"]

    def has_full_cudagraphs(self):
        return False

    def has_piecewise_cudagraphs(self):
        return True


def test_musa_mccl_tp_splitting(restore):
    PlatformFL.device_type = "musa"
    PlatformFL.vendor_name = "mthreads"
    PlatformFL.dist_backend = "mccl"
    config = _base_vllm_config()
    config.model_config.is_moe = True
    config.parallel_config.tensor_parallel_size = 2
    config.compilation_config.cudagraph_mode = _SplittingMode()
    config.compilation_config.splitting_ops = ["x"]
    config.compilation_config.mode = _CompMode.VLLM_COMPILE
    calls = []

    def _set_splitting_ops(all2all_backend=None, data_parallel_size=None):
        calls.append((all2all_backend, data_parallel_size))

    config.compilation_config.set_splitting_ops_for_v1 = _set_splitting_ops
    PlatformFL.check_and_update_config(config)
    assert calls == [("allgather_reducescatter", 1)]
    assert "vllm::all_reduce" in config.compilation_config.splitting_ops


# ===========================================================================
# 8. import_kernels
# ===========================================================================
def _fake_platform_super(monkeypatch):
    calls = []
    base = PlatformFL.__mro__[1]
    monkeypatch.setattr(
        base, "import_kernels", classmethod(lambda cls: calls.append("super")),
        raising=False,
    )
    return calls


def test_import_kernels_non_metax_calls_super(restore, monkeypatch):
    PlatformFL.vendor_name = "nvidia"
    PlatformFL.device_type = "cuda"
    calls = _fake_platform_super(monkeypatch)
    PlatformFL.import_kernels()
    assert calls == ["super"]


def test_import_kernels_metax_ok(restore, monkeypatch):
    PlatformFL.vendor_name = "metax"
    PlatformFL.device_type = "cuda"
    for name in ("mcoplib", "mcoplib._C", "mcoplib._moe_C"):
        mod = _make_pkg(name) if name == "mcoplib" else types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
    _install_metax_patches(monkeypatch, ())
    _quiet(monkeypatch)
    PlatformFL.import_kernels()


def test_import_kernels_metax_failures_warn(restore, monkeypatch):
    PlatformFL.vendor_name = "metax"
    PlatformFL.device_type = "cuda"
    mcoplib = _make_pkg("mcoplib")
    monkeypatch.setitem(sys.modules, "mcoplib", mcoplib)
    monkeypatch.setitem(sys.modules, "mcoplib._C", None)
    monkeypatch.setitem(sys.modules, "mcoplib._moe_C", None)
    name = "vllm_fl.dispatch.backends.vendor.metax.patches"
    _ensure_chain_mp(monkeypatch, name)
    monkeypatch.setitem(sys.modules, name, None)
    _quiet(monkeypatch)
    PlatformFL.import_kernels()


def test_import_kernels_musa_applies(restore, monkeypatch):
    PlatformFL.vendor_name = "mthreads"
    PlatformFL.device_type = "musa"
    _fake_platform_super(monkeypatch)
    applied = []
    name = "vllm_fl.dispatch.backends.vendor.musa.patch"
    _ensure_chain_mp(monkeypatch, name)
    fake = types.ModuleType(name)
    fake.apply_musa_patches = lambda: applied.append(1)
    monkeypatch.setitem(sys.modules, name, fake)
    PlatformFL.import_kernels()
    assert applied == [1]


def test_import_kernels_musa_failure_warns(restore, monkeypatch):
    PlatformFL.vendor_name = "mthreads"
    PlatformFL.device_type = "musa"
    _fake_platform_super(monkeypatch)
    name = "vllm_fl.dispatch.backends.vendor.musa.patch"
    _ensure_chain_mp(monkeypatch, name)
    monkeypatch.setitem(sys.modules, name, None)
    _quiet(monkeypatch)
    PlatformFL.import_kernels()


# ===========================================================================
# 9. get_device_uuid / get_device_capability
# ===========================================================================
def test_device_uuid_cuda_nvml(restore, monkeypatch):
    PlatformFL.device_type = "cuda"
    PlatformFL.vendor_name = "nvidia"
    monkeypatch.setattr(
        PlatformFL,
        "visible_device_id_to_physical_device_id",
        classmethod(lambda cls, i: i + 1),
        raising=False,
    )
    _fake_pynvml(monkeypatch, uuid="NVIDIA-UUID")
    assert PlatformFL.get_device_uuid(3) == "NVIDIA-UUID"


def test_device_uuid_iluvatar_synthetic(restore):
    PlatformFL.device_type = "cuda"
    PlatformFL.vendor_name = "iluvatar"
    assert PlatformFL.get_device_uuid(7) == "ILUVATAR-7"


def test_device_uuid_npu_env(restore, monkeypatch):
    PlatformFL.device_type = "npu"
    PlatformFL.vendor_name = "ascend"
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5")
    assert PlatformFL.get_device_uuid(1) == "NPU-5"


def test_device_uuid_npu_default(restore, monkeypatch):
    PlatformFL.device_type = "npu"
    PlatformFL.vendor_name = "ascend"
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    assert PlatformFL.get_device_uuid(2) == "NPU-2"


def test_device_uuid_generic(restore, monkeypatch):
    PlatformFL.device_type = "txda"
    PlatformFL.vendor_name = "tsingmicro"
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    assert PlatformFL.get_device_uuid(2) == "txda-2"


def test_device_capability_npu_none(restore):
    PlatformFL.device_type = "npu"
    assert PlatformFL.get_device_capability(0) is None


def test_device_capability_musa(restore, monkeypatch):
    PlatformFL.device_type = "musa"
    monkeypatch.setattr(
        fl_platform, "torch",
        NS(musa=NS(get_device_capability=lambda d: (9, 0))),
    )
    cap = PlatformFL.get_device_capability(0)
    assert (cap.major, cap.minor) == (9, 0)


def test_device_capability_gcu(restore, monkeypatch):
    PlatformFL.device_type = "gcu"
    monkeypatch.setattr(
        fl_platform, "torch",
        NS(gcu=NS(get_device_capability=lambda d: (8, 6))),
    )
    cap = PlatformFL.get_device_capability(0)
    assert (cap.major, cap.minor) == (8, 6)


def test_device_capability_gcu_absent(restore, monkeypatch):
    PlatformFL.device_type = "gcu"
    monkeypatch.setattr(fl_platform, "torch", NS())
    assert PlatformFL.get_device_capability(0) is None


def test_device_capability_txda_none(restore):
    PlatformFL.device_type = "txda"
    assert PlatformFL.get_device_capability(0) is None


def test_device_capability_default(restore, monkeypatch):
    PlatformFL.device_type = "cuda"
    monkeypatch.setattr(
        PlatformFL, "torch_device_fn",
        NS(get_device_capability=lambda d: (8, 0)), raising=False,
    )
    cap = PlatformFL.get_device_capability(0)
    assert (cap.major, cap.minor) == (8, 0)


# ===========================================================================
# 10. is_fully_connected
# ===========================================================================
def test_fully_connected_true(restore, monkeypatch):
    _fake_pynvml(monkeypatch, p2p_status=0)
    assert PlatformFL.is_fully_connected([0, 1]) is True


def test_fully_connected_bad_p2p(restore, monkeypatch):
    _fake_pynvml(monkeypatch, p2p_status=1)
    assert PlatformFL.is_fully_connected([0, 1]) is False


def test_fully_connected_nvml_error(restore, monkeypatch):
    mod = _fake_pynvml(monkeypatch)

    def _raise(a, b, caps):
        raise mod.NVMLError("boom")

    mod.nvmlDeviceGetP2PStatus = _raise
    _quiet(monkeypatch)
    assert PlatformFL.is_fully_connected([0, 1]) is False


def test_fully_connected_no_pynvml(restore, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    assert PlatformFL.is_fully_connected([0, 1]) is False


# ===========================================================================
# 11. get_vit_attn_backend
# ===========================================================================
def _abe(flash_factory=None, sdpa=None):
    sdpa = sdpa if sdpa is not None else _BackendMember("TORCH_SDPA")
    flash = _BackendMember(
        "FLASH_ATTN", flash_factory or (lambda: _VitFlash)
    )
    return NS(TORCH_SDPA=sdpa, FLASH_ATTN=flash)


class _FakeGcuAtten:
    supports = True

    @classmethod
    def supports_head_size(cls, head_size):
        return cls.supports

    @classmethod
    def supports_dtype(cls, dtype):
        return cls.supports


def _install_gcu_attention(monkeypatch, supports=True):
    name = "vllm_fl.dispatch.backends.vendor.gcu.impl.attention"
    _ensure_chain_mp(monkeypatch, name)
    fake = types.ModuleType(name)
    cls = type("AttentionGCUBackend", (), {
        "supports": supports,
        "supports_head_size": staticmethod(lambda hs: supports),
        "supports_dtype": staticmethod(lambda dt: supports),
    })
    fake.AttentionGCUBackend = cls
    monkeypatch.setitem(sys.modules, name, fake)
    return fake


def test_vit_explicit_supported(restore, monkeypatch):
    PlatformFL.vendor_name = "nvidia"
    _install_attn_utils(monkeypatch)
    abe = _abe()
    monkeypatch.setattr(fl_platform, "AttentionBackendEnum", abe)
    out = PlatformFL.get_vit_attn_backend(128, "fp16", backend=abe.TORCH_SDPA)
    assert out is abe.TORCH_SDPA


def test_vit_explicit_unsupported_asserts(restore, monkeypatch):
    PlatformFL.vendor_name = "nvidia"
    _install_attn_utils(monkeypatch)
    abe = _abe()
    monkeypatch.setattr(fl_platform, "AttentionBackendEnum", abe)
    with pytest.raises(AssertionError):
        PlatformFL.get_vit_attn_backend(128, "fp16", backend=object())


def test_vit_gcu_explicit_flash_ok(restore, monkeypatch):
    PlatformFL.vendor_name = "gcu"
    _install_attn_utils(monkeypatch)
    _install_gcu_attention(monkeypatch, supports=True)
    abe = _abe()
    monkeypatch.setattr(fl_platform, "AttentionBackendEnum", abe)
    _quiet(monkeypatch)
    out = PlatformFL.get_vit_attn_backend(
        128, "fp16", backend=abe.FLASH_ATTN
    )
    assert out is abe.FLASH_ATTN


def test_vit_gcu_explicit_flash_invalid(restore, monkeypatch):
    PlatformFL.vendor_name = "gcu"
    _install_attn_utils(monkeypatch)
    _install_gcu_attention(monkeypatch, supports=False)
    abe = _abe()
    monkeypatch.setattr(fl_platform, "AttentionBackendEnum", abe)
    with pytest.raises(ValueError):
        PlatformFL.get_vit_attn_backend(
            128, "fp16", backend=abe.FLASH_ATTN
        )


def test_vit_gcu_auto_flash_then_sdpa(restore, monkeypatch):
    PlatformFL.vendor_name = "gcu"
    _install_attn_utils(monkeypatch)
    abe = _abe()
    monkeypatch.setattr(fl_platform, "AttentionBackendEnum", abe)
    _install_gcu_attention(monkeypatch, supports=True)
    assert (
        PlatformFL.get_vit_attn_backend(128, "fp16") is abe.FLASH_ATTN
    )
    _install_gcu_attention(monkeypatch, supports=False)
    assert (
        PlatformFL.get_vit_attn_backend(128, "fp16") is abe.TORCH_SDPA
    )


def test_supported_vit_amd_rocm(restore, monkeypatch):
    PlatformFL.vendor_name = "amd"
    sentinel = object()
    fake = types.ModuleType("vllm.platforms.rocm")
    fake.RocmPlatform = NS(
        get_supported_vit_attn_backends=lambda: [sentinel]
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", fake)
    assert PlatformFL.get_supported_vit_attn_backends() == [sentinel]


def test_supported_vit_amd_no_rocm(restore, monkeypatch):
    PlatformFL.vendor_name = "amd"
    abe = _abe()
    monkeypatch.setattr(fl_platform, "AttentionBackendEnum", abe)
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", None)
    _quiet(monkeypatch)
    assert abe.TORCH_SDPA in PlatformFL.get_supported_vit_attn_backends()


def test_vit_amd_defers_to_rocm(restore, monkeypatch):
    PlatformFL.vendor_name = "amd"
    _install_attn_utils(monkeypatch)
    sentinel = object()
    fake = types.ModuleType("vllm.platforms.rocm")
    fake.RocmPlatform = NS(
        get_vit_attn_backend=lambda hs, dt, be: sentinel
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", fake)
    assert PlatformFL.get_vit_attn_backend(128, "fp16") is sentinel


def test_vit_amd_no_rocm_falls_back(restore, monkeypatch):
    PlatformFL.vendor_name = "amd"
    _install_attn_utils(monkeypatch)
    abe = _abe()
    monkeypatch.setattr(fl_platform, "AttentionBackendEnum", abe)
    monkeypatch.setitem(sys.modules, "vllm.platforms.rocm", None)
    _quiet(monkeypatch)
    assert PlatformFL.get_vit_attn_backend(128, "fp16") is abe.TORCH_SDPA


def test_vit_generic_flash_supported(restore, monkeypatch):
    PlatformFL.vendor_name = "nvidia"
    _install_attn_utils(monkeypatch)
    monkeypatch.setattr(
        PlatformFL, "get_device_capability",
        classmethod(lambda cls, d=0: NS(major=8)),
    )
    abe = _abe()
    monkeypatch.setattr(fl_platform, "AttentionBackendEnum", abe)
    assert PlatformFL.get_vit_attn_backend(128, "fp16") is abe.FLASH_ATTN


def test_vit_generic_flash_import_error(restore, monkeypatch):
    PlatformFL.vendor_name = "nvidia"
    _install_attn_utils(monkeypatch)
    monkeypatch.setattr(
        PlatformFL, "get_device_capability",
        classmethod(lambda cls, d=0: NS(major=8)),
    )

    def _raise():
        raise ImportError("no flash")

    abe = _abe(flash_factory=_raise)
    monkeypatch.setattr(fl_platform, "AttentionBackendEnum", abe)
    assert PlatformFL.get_vit_attn_backend(128, "fp16") is abe.TORCH_SDPA


def test_vit_generic_low_capability(restore, monkeypatch):
    PlatformFL.vendor_name = "nvidia"
    _install_attn_utils(monkeypatch)
    monkeypatch.setattr(
        PlatformFL, "get_device_capability",
        classmethod(lambda cls, d=0: NS(major=7)),
    )
    abe = _abe()
    monkeypatch.setattr(fl_platform, "AttentionBackendEnum", abe)
    assert PlatformFL.get_vit_attn_backend(128, "fp16") is abe.TORCH_SDPA


# ===========================================================================
# 12. _vllm_c_is_registered_provider
# ===========================================================================
def _install_ir_modules(monkeypatch, cfg_cls):
    _ensure_chain_mp(monkeypatch, "vllm.config.kernel")
    config_kernel = types.ModuleType("vllm.config.kernel")
    config_kernel.IrOpPriorityConfig = cfg_cls
    monkeypatch.setitem(sys.modules, "vllm.config.kernel", config_kernel)

    class IrOp:
        def __init__(self, impls=None):
            self.impls = impls if impls is not None else {}

    _ensure_chain_mp(monkeypatch, "vllm.ir.op")
    op_mod = types.ModuleType("vllm.ir.op")
    op_mod.IrOp = IrOp
    ops_mod = types.ModuleType("vllm.ir.ops")
    ir_pkg = types.ModuleType("vllm.ir")
    ir_pkg.__path__ = []
    ir_pkg.op = op_mod
    ir_pkg.ops = ops_mod
    monkeypatch.setitem(sys.modules, "vllm.ir", ir_pkg)
    monkeypatch.setitem(sys.modules, "vllm.ir.op", op_mod)
    monkeypatch.setitem(sys.modules, "vllm.ir.ops", ops_mod)
    return IrOp, ops_mod


@dataclass
class _IrCfg:
    rms_norm: object = None
    fused_add_rms_norm: object = None


def test_vllm_c_all_registered(monkeypatch):
    IrOp, ops = _install_ir_modules(monkeypatch, _IrCfg)
    good = IrOp({"vllm_c": 1, "native": 2})
    ops.rms_norm = good
    ops.fused_add_rms_norm = good
    assert PlatformFL._vllm_c_is_registered_provider() is True


def test_vllm_c_missing_impl(monkeypatch):
    IrOp, ops = _install_ir_modules(monkeypatch, _IrCfg)
    bad = IrOp({"native": 1})
    ops.rms_norm = bad
    ops.fused_add_rms_norm = bad
    assert PlatformFL._vllm_c_is_registered_provider() is False


def test_vllm_c_candidate_not_ir_op(monkeypatch):
    _, ops = _install_ir_modules(monkeypatch, _IrCfg)
    ops.rms_norm = object()
    ops.fused_add_rms_norm = object()
    assert PlatformFL._vllm_c_is_registered_provider() is False


def test_vllm_c_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.config.kernel", None)
    assert PlatformFL._vllm_c_is_registered_provider() is False


def test_vllm_c_fields_type_error(monkeypatch):
    class _NotDataclass:
        pass

    _install_ir_modules(monkeypatch, _NotDataclass)
    assert PlatformFL._vllm_c_is_registered_provider() is False


@dataclass
class _EmptyIrCfg:
    pass


def test_vllm_c_empty_config(monkeypatch):
    _install_ir_modules(monkeypatch, _EmptyIrCfg)
    assert PlatformFL._vllm_c_is_registered_provider() is False


@dataclass
class _SingleIrCfg:
    only: object = None


def test_vllm_c_impls_raises(monkeypatch):
    IrOp, ops = _install_ir_modules(monkeypatch, _SingleIrCfg)

    class _Bad(IrOp):
        def __init__(self):
            pass

        @property
        def impls(self):
            raise RuntimeError("boom")

    ops.only = _Bad()
    assert PlatformFL._vllm_c_is_registered_provider() is False


def test_vllm_c_candidate_missing(monkeypatch):
    _install_ir_modules(monkeypatch, _SingleIrCfg)
    assert PlatformFL._vllm_c_is_registered_provider() is False
