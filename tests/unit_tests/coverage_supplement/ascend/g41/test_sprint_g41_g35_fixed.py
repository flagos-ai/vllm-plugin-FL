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

"""G41 fixed port of the G35 residual coverage file.

G35 collected 35 errors on the 910C image because it first tried
``importlib.import_module("vllm_fl.platform")``.  That triggers the package
``vllm_fl/__init__.py`` chain and, worse, ``vllm_fl.utils.DeviceInfo()`` whose
FlagGems ``DeviceDetector`` has no ``vendor_name`` attribute on that image:

    AttributeError: 'DeviceDetector' object has no attribute 'vendor_name'

``DeviceInfo()`` runs in the ``PlatformFL`` class body, so the real import can
never succeed there.  This file uses the G25-proven direct-source strategy for
*both* modules:

1. Install a complete ``torch`` / ``triton`` stub.
2. Force-replace every dependency module in ``sys.modules`` with a fresh stub
   (no "keep the real module if importable" branch, so the broken FlagGems
   detector is never consulted).
3. Load ``vllm_fl/platform.py`` and ``vllm_fl/worker/worker.py`` straight from
   source via ``importlib.util.spec_from_file_location``, never importing the
   ``vllm_fl`` package chain.

All 35 original G35 cases are preserved verbatim.  No accelerator, no network,
no subprocess, no ``hc``/SSH access.
"""

from __future__ import annotations

import enum
import importlib
import importlib.util
import os
import sys
import types
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


# ===========================================================================
# 1. torch / triton stubs
# ===========================================================================
def _ensure_torch_stub() -> None:
    try:
        import torch  # noqa: F401

        t = sys.modules["torch"]
    except Exception:
        t = sys.modules.get("torch") or types.ModuleType("torch")
        sys.modules["torch"] = t

    if not hasattr(t, "dtype"):
        t.dtype = type("dtype", (), {})
    if not hasattr(t, "Tensor"):
        t.Tensor = type("Tensor", (), {})
    if not hasattr(t, "device"):

        class _Device:
            def __init__(self, spec="cpu"):
                self.spec = str(spec)
                self.type = self.spec.split(":")[0]

            def __str__(self):
                return self.spec

            def __repr__(self):
                return f"device({self.spec})"

        t.device = _Device
    if not hasattr(t, "types"):
        t.types = types.ModuleType("torch.types")
        sys.modules.setdefault("torch.types", t.types)
    if not hasattr(t.types, "Device"):
        t.types.Device = object
    if not hasattr(t, "nn"):
        nn = types.ModuleType("torch.nn")
        nn.Module = type("Module", (), {})
        t.nn = nn
        sys.modules["torch.nn"] = nn
    for name in (
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "int32",
        "int64",
        "bool",
    ):
        if not hasattr(t, name):
            setattr(t, name, type("_dt", (), {"name": name})())
    if not hasattr(t, "accelerator"):
        t.accelerator = NS(
            device_count=lambda: 0,
            empty_cache=lambda: None,
            synchronize=lambda: None,
            get_memory_info=lambda: (0, 0),
        )
    if not hasattr(t, "zeros"):
        t.zeros = lambda *a, **k: type("_T", (), {})()
    if not hasattr(t, "ones"):
        t.ones = lambda *a, **k: type("_T", (), {})()
    if not hasattr(t, "manual_seed"):
        t.manual_seed = lambda *a, **k: None
    if not hasattr(t, "set_float32_matmul_precision"):
        t.set_float32_matmul_precision = lambda *a, **k: None

    # Keep real torch's accelerator when present, only patch missing pieces.
    acc = getattr(t, "accelerator", None)
    if acc is not None:
        for name, value in (
            ("device_count", lambda: 0),
            ("empty_cache", lambda: None),
            ("synchronize", lambda: None),
            ("get_memory_info", lambda: (0, 0)),
        ):
            if not hasattr(acc, name):
                try:
                    setattr(acc, name, value)
                except Exception:
                    pass


_LANG_ATTRS = (
    "float32",
    "float16",
    "bfloat16",
    "int32",
    "int64",
    "constexpr",
    "static_range",
    "program_id",
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
        ("program_id", lambda axis=0: 0),
    ):
        if not hasattr(lang, name):
            setattr(lang, name, value)
    triton.language = lang
    if not hasattr(triton, "jit"):
        triton.jit = _jit
    if not hasattr(triton, "autotune"):
        triton.autotune = _autotune
for sub in ("triton.runtime", "triton.compiler", "triton.backends",
            "triton.runtime.driver", "triton.runtime.jit", "triton.runtime.autotuner",
            "triton.compiler.compiler", "triton.backends.ascend", "triton.backends.nvidia",
            "triton.backends.compiler", "triton.tools", "triton.tools.disasm"):
    _m = types.ModuleType(sub)
    _m.__path__ = []

    class _Any:
        def __getattr__(self, _n):
            return self

        def __call__(self, *a, **k):
            return self

        def __iter__(self):
            return iter(())

        def __mro_entries__(self, bases):
            return (object,)

    _m.__getattr__ = lambda _n, _mod=_m: _Any()
    sys.modules.setdefault(sub, _m)


# ===========================================================================
# 2. deterministic fakes shared by both modules
# ===========================================================================
def _logger():
    return NS(
        info=lambda *a, **k: None,
        info_once=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        warning_once=lambda *a, **k: None,
        error=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )


class _CUDAGraphMode(enum.Enum):
    NONE = "NONE"
    PIECEWISE = "PIECEWISE"
    FULL = "FULL"
    FULL_AND_PIECEWISE = "FULL_AND_PIECEWISE"

    def has_full_cudagraphs(self):
        return self in (type(self).FULL, type(self).FULL_AND_PIECEWISE)

    def has_piecewise_cudagraphs(self):
        return self in (type(self).PIECEWISE, type(self).FULL_AND_PIECEWISE)


class _CompilationMode(enum.Enum):
    NONE = "NONE"
    VLLM_COMPILE = "VLLM_COMPILE"


class _PlatformEnum:
    OOT = "OOT"
    CUDA = "CUDA"


class _AttentionBackendEnum(enum.Enum):
    TORCH_SDPA = "TORCH_SDPA"
    FLASH_ATTN = "FLASH_ATTN"

    def get_class(self):
        return NS(
            supports_head_size=lambda hs: True,
            supports_dtype=lambda dt: True,
        )


@dataclass
class _DeviceCapability:
    major: int
    minor: int

    def as_version_str(self):
        return f"{self.major}.{self.minor}"


class _Platform:
    @classmethod
    def import_kernels(cls):
        return None

    @classmethod
    def get_compile_backend(cls):
        return "inductor"

    @classmethod
    def _align_hybrid_block_size(cls, vllm_config, backend_cls):
        return None


class _IntermediateTensors:
    def __init__(self, tensors):
        self.tensors = tensors


class _Handle:
    def wait(self):
        pass


class _WorkerBase:
    def __init__(self, *a, **k):
        pass

    def get_supported_kv_cache_layouts(self):
        return []


class _CompilationTimes:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _EmptyOutput:
    pass


class _MemorySnapshot:
    def __init__(self, device=None, **kw):
        self.device = device
        self.free_memory = kw.get("free_memory", 1 << 30)
        self.total_memory = kw.get("total_memory", 2 << 30)


def _with_ctx_factory():
    def managed_inference_mode(*a, **k):
        def deco(fn):
            return fn

        return deco

    return managed_inference_mode


def _decorator_factory(*a, **k):
    def deco(fn):
        return fn

    return deco


def _device_info():
    vendor = os.environ.get("GEMS_VENDOR", "")
    return NS(
        vendor_name=vendor,
        device_type=_VENDOR_DEVICE.get(vendor, ""),
        dispatch_key="",
        torch_device_fn=sys.modules["torch"],
        device=object(),
    )


_VENDOR_DEVICE = {
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
}


def _get_device_type(vendor):
    return _VENDOR_DEVICE.get(vendor or "", "")


def _get_device_name(vendor):
    # cuda-alike vendors reuse the vendor name; others use their device type.
    device = _VENDOR_DEVICE.get(vendor or "", "")
    if device == "cuda" and vendor not in ("iluvatar", "hygon", "amd"):
        return vendor
    return device


def _get_device_control_env_var(vendor):
    return {
        "ascend": "ASCEND_RT_VISIBLE_DEVICES",
        "mthreads": "MUSA_VISIBLE_DEVICES",
    }.get(vendor, "CUDA_VISIBLE_DEVICES")


_CURRENT_PLATFORM = NS(
    device_type="",
    vendor_name="",
    dist_backend="nccl",
    is_cuda_alike=lambda: False,
    is_cuda=lambda: False,
    is_rocm=lambda: False,
    is_xpu=lambda: False,
    is_cpu=lambda: True,
    is_sleep_mode_available=lambda: False,
    logical_device_id_to_visible_device_id=lambda i: i,
    set_device=lambda d: None,
    check_if_supports_dtype=lambda dt: None,
)


def _build_stub_table():
    return {
        "vllm.logger": {"init_logger": lambda name: _logger()},
        "vllm.envs": {
            "VLLM_FLOAT32_MATMUL_PRECISION": "high",
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": False,
            "VLLM_USE_BREAKABLE_CUDAGRAPH": False,
        },
        "vllm.config": {
            "CUDAGraphMode": _CUDAGraphMode,
            "VllmConfig": type("VllmConfig", (), {}),
            "set_current_vllm_config": lambda cfg: nullcontext(),
            "get_current_vllm_config_or_none": lambda: None,
        },
        "vllm.config.compilation": {"CompilationMode": _CompilationMode},
        "vllm.config.cache": {"CacheDType": object},
        "vllm.config.kernel": {},
        "vllm.platforms": {
            "Platform": _Platform,
            "PlatformEnum": _PlatformEnum,
            "current_platform": _CURRENT_PLATFORM,
        },
        "vllm.platforms.interface": {
            "DeviceCapability": _DeviceCapability,
            "set_assigned_physical_gpu_ids": lambda ids: None,
        },
        "vllm.v1.attention.backends.registry": {
            "AttentionBackendEnum": _AttentionBackendEnum
        },
        "vllm.v1.attention.selector": {"AttentionSelectorConfig": object},
        "vllm.v1.core.sched.output": {
            "GrammarOutput": _EmptyOutput,
            "SchedulerOutput": _EmptyOutput,
        },
        "vllm.v1.kv_cache_interface": {
            "KVCacheConfig": _EmptyOutput,
            "KVCacheSpec": _EmptyOutput,
        },
        "vllm.v1.outputs": {
            "AsyncModelRunnerOutput": type("AsyncModelRunnerOutput", (), {}),
            "DraftTokenIds": _EmptyOutput,
            "ModelRunnerOutput": type("ModelRunnerOutput", (), {}),
        },
        "vllm.v1.utils": {
            "compute_iteration_details": lambda so: NS(
                num_ctx_requests=0,
                num_ctx_tokens=0,
                num_generation_requests=0,
                num_generation_tokens=0,
            ),
            "report_usage_stats": lambda cfg: None,
        },
        "vllm.v1.worker.gpu.warmup": {"warmup_kernels": lambda *a, **k: None},
        "vllm.v1.worker.gpu.model_runner": {
            "GPUModelRunner": type("GPUModelRunner", (), {})
        },
        "vllm.v1.worker.sentinel.gpu_worker_sentinel": {
            "WorkerSentinel": type("WorkerSentinel", (), {})
        },
        "vllm.v1.worker.startup_plan": {
            "maybe_apply_startup_plan": lambda w: None,
            "maybe_save_startup_plan": lambda w, v: None,
        },
        "vllm.v1.worker.utils": {
            "request_memory": lambda s, c: 0,
            "is_residual_scattered_for_sp": lambda *a, **k: False,
        },
        "vllm.v1.engine": {
            "ReconfigureDistributedRequest": _EmptyOutput,
            "ReconfigureRankType": _EmptyOutput,
        },
        "vllm.v1.worker.worker_base": {
            "CompilationTimes": _CompilationTimes,
            "WorkerBase": _WorkerBase,
        },
        "vllm.v1.worker.workspace": {
            "init_workspace_manager": lambda *a: None
        },
        "vllm.v1.worker.elastic_ep.elastic_execute": {
            "ElasticEPScalingExecutor": type(
                "ElasticEPScalingExecutor",
                (),
                {"__init__": lambda self, worker=None: None},
            )
        },
        "vllm.device_allocator": {"get_mem_allocator_instance": lambda: None},
        "vllm.device_allocator.cumem": {
            "CuMemAllocator": type("CuMemAllocator", (), {"instance": None})
        },
        "vllm.device_allocator.sleep_mode_backend": {
            "SleepModeBackend": type("SleepModeBackend", (), {})
        },
        "vllm.distributed": {
            "ensure_model_parallel_initialized": lambda *a: None,
            "init_distributed_environment": lambda *a: None,
            "set_custom_all_reduce": lambda flag: None,
        },
        "vllm.distributed.ec_transfer": {
            "ensure_ec_transfer_initialized": lambda cfg: None,
            "ensure_ec_transfer_shutdown": lambda: None,
        },
        "vllm.distributed.eplb.eplb_utils": {
            "override_envs_for_eplb": lambda *a, **k: None
        },
        "vllm.distributed.kv_transfer": {
            "ensure_kv_transfer_initialized": lambda cfg, kv: None,
            "ensure_kv_transfer_shutdown": lambda: None,
            "get_kv_transfer_group": lambda: NS(
                get_handshake_metadata=lambda: None
            ),
            "has_kv_transfer_group": lambda: False,
        },
        "vllm.distributed.kv_transfer.kv_connector.v1.base": {
            "KVConnectorHandshakeMetadata": _EmptyOutput
        },
        "vllm.distributed.parallel_state": {
            "Handle": _Handle,
            "checkpoint_prepare_distributed_state": lambda: None,
            "checkpoint_restore_distributed_state": lambda: None,
            "get_pp_group": lambda: NS(
                is_first_rank=True, is_last_rank=True, rank_in_group=0
            ),
            "get_tp_group": lambda: NS(rank_in_group=0),
            "get_pcp_group": lambda: NS(rank_in_group=0),
            "prepare_communication_buffer_for_model": lambda *a, **k: None,
        },
        "vllm.distributed.weight_transfer": {
            "WeightTransferEngine": type("WeightTransferEngine", (), {}),
            "WeightTransferEngineFactory": NS(
                create_engine=lambda *a, **k: None
            ),
        },
        "vllm.distributed.utils": {
            "get_worker_rank_suffix": lambda global_rank=None: f"r{global_rank}"
        },
        "vllm.lora.request": {"LoRARequest": _EmptyOutput},
        "vllm.model_executor.model_loader": {
            "TensorizerLoader": NS(save_model=lambda *a, **k: None)
        },
        "vllm.model_executor.models.interfaces": {
            "is_mixture_of_experts": lambda model: False
        },
        "vllm.model_executor.warmup.kernel_warmup": {
            "kernel_warmup": lambda w: None
        },
        "vllm.model_executor.determinism.batch_invariant": {
            "init_batch_invariance": lambda: None
        },
        "vllm.multimodal.gpu_ipc_memory": {
            "reserve_mm_ipc_gpu_memory": lambda *a, **k: 0
        },
        "vllm.profiler.wrapper": {
            "CudaProfilerWrapper": type("CudaProfilerWrapper", (), {}),
            "ProtonProfilerWrapper": type("ProtonProfilerWrapper", (), {}),
            "TorchProfilerWrapper": type("TorchProfilerWrapper", (), {}),
        },
        "vllm.sequence": {"IntermediateTensors": _IntermediateTensors},
        "vllm.tasks": {"SupportedTask": _EmptyOutput},
        "vllm.tracing": {"instrument": _decorator_factory},
        "vllm.utils.gc_utils": {
            "freeze_gc_heap": lambda: None,
            "maybe_attach_gc_debug_callback": lambda: None,
        },
        "vllm.utils.gpu_sync_debug": {
            "enable_gpu_sync_check": lambda: None,
            "with_gpu_sync_check": lambda fn: fn,
        },
        "vllm.utils.mem_constants": {"GiB_bytes": 1 << 30},
        "vllm.utils.mem_utils": {
            "MemorySnapshot": _MemorySnapshot,
            "format_gib": lambda b: round(b, 2),
            "memory_profiling": lambda *a, **k: nullcontext(
                NS(
                    total_consumed=0,
                    transient_peak_headroom=0,
                    after_profile=NS(free_memory=0),
                    non_kv_cache_memory=0,
                )
            ),
        },
        "vllm.utils.torch_utils": {
            "set_random_seed": lambda seed: None,
            "set_torch_threads_for_runtime": lambda: None,
        },
        "vllm.utils.jit_monitor": {"activate": lambda *a, **k: None},
        "vllm.kernels": {},
        "vllm_fl.utils": {
            "DeviceInfo": _device_info,
            "_env_vendor_name": lambda: os.environ.get("GEMS_VENDOR", ""),
            "get_device_control_env_var": _get_device_control_env_var,
            "get_device_type": _get_device_type,
            "get_device_name": _get_device_name,
            "get_flag_gems_whitelist_blacklist": lambda **k: ([], []),
            "SPLITTING_OPS": {},
        },
        "vllm_fl.envs": {},
        "vllm_fl.dispatch": {
            "call_op": lambda *a, **k: "backend",
            "warmup_cached_ops": lambda *a, **k: None,
        },
        "vllm_fl.dispatch.io_common": {
            "managed_inference_mode": _with_ctx_factory()
        },
        "vllm_fl.worker.runtime": {"_initialize_fl_runtime": lambda r: None},
        "vllm_fl.worker.model_runner": {
            "ModelRunnerFL": type(
                "ModelRunnerFL",
                (),
                {"__init__": lambda self, cfg, dev: None},
            )
        },
        "vllm_fl.worker.scheduler_fl": {},
        "vllm_fl.attention.utils": {
            "patch_mm_encoder_attention": lambda: None
        },
    }


def _ensure_pkg(name):
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    mod = types.ModuleType(name)
    mod.__path__ = []
    sys.modules[name] = mod
    return mod


def _force_module(name, attrs):
    """Replace ``name`` with a fresh stub module.

    The G35 loader kept any importable real module (``_get_or_make``).  On the
    910C image that kept FlagGems' real ``vllm_fl.utils`` whose
    ``DeviceInfo()`` raises ``'DeviceDetector' object has no attribute
    'vendor_name'`` while ``PlatformFL``'s class body is evaluated.  Replacing
    unconditionally removes that failure mode.
    """
    mod = types.ModuleType(name)
    mod.__path__ = []
    for key, value in attrs.items():
        setattr(mod, key, value)
    parts = name.split(".")
    for i in range(1, len(parts)):
        _ensure_pkg(".".join(parts[:i]))
    sys.modules[name] = mod
    if len(parts) > 1:
        parent = sys.modules[".".join(parts[:-1])]
        setattr(parent, parts[-1], mod)
    return mod


def _install_stubs() -> None:
    for name in (
        "vllm",
        "vllm.config",
        "vllm.config.compilation",
        "vllm.config.cache",
        "vllm.platforms",
        "vllm.platforms.interface",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
        "vllm.v1.worker",
        "vllm.distributed",
        "vllm.utils",
        "vllm.model_executor",
        "vllm.lora",
        "vllm.multimodal",
        "vllm.profiler",
        "vllm_fl",
        "vllm_fl.worker",
        "vllm_fl.dispatch",
        "vllm_fl.attention",
    ):
        _ensure_pkg(name)
    for name, attrs in _build_stub_table().items():
        _force_module(name, attrs)


def _inject(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        if parent_name not in sys.modules:
            pkg = types.ModuleType(parent_name)
            pkg.__path__ = []
            sys.modules[parent_name] = pkg
    sys.modules[name] = mod
    parent = sys.modules.get(".".join(parts[:-1]))
    if parent is not None:
        setattr(parent, parts[-1], mod)
    return mod


# ===========================================================================
# 3. source discovery + forced-stub direct-source loader
# ===========================================================================
_SRC_ENV = {
    "vllm_fl/platform.py": ("VLLM_FL_PLATFORM_SRC",),
    "vllm_fl/worker/worker.py": ("VLLM_FL_WORKER_SRC",),
}


def _find_source(rel: str) -> str:
    cands = []
    for var in (
        *_SRC_ENV.get(rel, ()),
        "VLLM_FL_SRC_ROOT",
        "VLLM_FL_WORKTREE",
        "FL_SRC_ROOT",
        "FL_REPO_ROOT",
        "FL_WORKSPACE",
    ):
        root = os.environ.get(var)
        if not root:
            continue
        p = Path(root)
        cands.append(p if p.suffix == ".py" else p / rel)
    temp = os.environ.get("TEMP") or os.environ.get("TMP")
    if temp:
        cands.append(Path(temp) / "opencode" / "cov910c" / "wt-814b" / rel)
    here = Path(__file__).resolve()
    for base in (here.parent, *here.parents, Path.cwd(), *Path.cwd().parents):
        cands.append(base / rel)
    for entry in sys.path:
        if entry:
            cands.append(Path(entry) / rel)
    seen = set()
    for cand in cands:
        try:
            key = str(cand.resolve())
        except Exception:
            key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        f"{rel} not found; set VLLM_FL_SRC_ROOT or VLLM_FL_PLATFORM_SRC"
    )


def _drop(module_name):
    for key in list(sys.modules):
        if key == module_name or key.startswith(module_name + "."):
            sys.modules.pop(key, None)


def _load_from_source(module_name, rel, checker):
    """Load ``rel`` straight from disk; never import the ``vllm_fl`` chain."""
    _drop(module_name)
    path = _find_source(rel)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    parent_name, _, child = module_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child, mod)
    spec.loader.exec_module(mod)
    if not checker(mod):
        raise ImportError(f"{path} missing expected symbols")
    return mod, f"direct-source:{path}"


def _has_platform(mod):
    return hasattr(mod, "PlatformFL")


def _has_worker(mod):
    return hasattr(mod, "WorkerFL")


_LOAD_ERROR = None
_FL_MODE = _WK_MODE = None
try:
    _ensure_torch_stub()
    _ensure_triton_stub()
    _install_stubs()
    fl, _FL_MODE = _load_from_source(
        "vllm_fl.platform", "vllm_fl/platform.py", _has_platform
    )
    wk, _WK_MODE = _load_from_source(
        "vllm_fl.worker.worker",
        "vllm_fl/worker/worker.py",
        _has_worker,
    )
    PlatformFL = fl.PlatformFL
    WorkerFL = wk.WorkerFL
except Exception as exc:  # pragma: no cover - environment guard
    fl = None
    wk = None
    PlatformFL = None
    WorkerFL = None
    _LOAD_ERROR = exc


@pytest.fixture(autouse=True)
def _require_modules():
    if PlatformFL is None or WorkerFL is None:
        pytest.fail(f"vllm_fl modules unavailable: {_LOAD_ERROR!r}")


@contextmanager
def _set_fl(monkeypatch, vendor="nvidia", device="cuda", dist="nccl", name=None):
    for attr, value in (
        ("vendor_name", vendor),
        ("device_type", device),
        ("dist_backend", dist),
        ("device_name", name if name is not None else vendor),
    ):
        monkeypatch.setattr(PlatformFL, attr, value, raising=False)
    yield


def _quiet(monkeypatch, module=None):
    log = _logger()
    monkeypatch.setattr(module or fl, "logger", log)
    return log


# ===========================================================================
# 4. PlatformFL predicates / dtype / memory helpers
# ===========================================================================
def test_platform_cuda_alike_and_identity_predicates(monkeypatch):
    cases = [
        ("iluvatar", "cuda", False),
        ("mthreads", "musa", True),
        ("hygon", "cuda", False),
        ("gcu", "gcu", True),
        ("enflame", "gcu", True),
        ("nvidia", "cuda", True),
        ("tsingmicro", "txda", False),
    ]
    for vendor, device, expected in cases:
        with _set_fl(monkeypatch, vendor, device):
            assert PlatformFL.is_cuda_alike(PlatformFL) is expected
    with _set_fl(monkeypatch, "nvidia", "cuda"):
        assert PlatformFL.is_cuda(PlatformFL) is True
        assert PlatformFL.is_rocm(PlatformFL) is False
        assert PlatformFL.is_sleep_mode_available(PlatformFL) is True
    with _set_fl(monkeypatch, "iluvatar", "cuda"):
        assert PlatformFL.is_cuda(PlatformFL) is False
    with _set_fl(monkeypatch, "amd", "cuda"):
        assert PlatformFL.is_rocm(PlatformFL) is True


def test_platform_musa_gcu_probes(monkeypatch):
    monkeypatch.setattr(
        fl, "torch", NS(musa=NS(is_available=lambda: True))
    )
    assert PlatformFL.is_musa(PlatformFL) is True
    monkeypatch.setattr(fl, "torch", NS())
    assert PlatformFL.is_musa(PlatformFL) is False
    monkeypatch.setattr(fl, "torch", NS(gcu=NS(is_available=lambda: True)))
    assert PlatformFL.is_gcu(PlatformFL) is True
    monkeypatch.setattr(fl, "torch", NS(gcu=NS(is_available=lambda: False)))
    assert PlatformFL.is_gcu(PlatformFL) is False


def test_platform_supported_dtypes_and_pin_memory(monkeypatch):
    assert PlatformFL.supported_dtypes.fget(PlatformFL) == [
        fl.torch.bfloat16,
        fl.torch.float16,
        fl.torch.float32,
    ]
    with _set_fl(monkeypatch, "nvidia", "cuda"):
        assert PlatformFL.is_pin_memory_available() is True
    with _set_fl(monkeypatch, "tsingmicro", "txda"):
        assert PlatformFL.is_pin_memory_available() is False


def test_platform_check_if_supports_dtype(monkeypatch):
    monkeypatch.setattr(fl, "torch", NS(bfloat16="bf16"))
    with _set_fl(monkeypatch, "nvidia", "cuda"):
        monkeypatch.setattr(
            PlatformFL,
            "has_device_capability",
            classmethod(lambda cls, n: False),
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
            classmethod(lambda cls, device_id=0: "OldGPU"),
            raising=False,
        )
        with pytest.raises(ValueError, match="does not have a compute capability"):
            PlatformFL.check_if_supports_dtype("bf16")
        monkeypatch.setattr(
            PlatformFL,
            "get_device_capability",
            classmethod(
                lambda cls, device_id=0: NS(as_version_str=lambda: "7.0")
            ),
            raising=False,
        )
        with pytest.raises(ValueError, match="compute capability 7.0"):
            PlatformFL.check_if_supports_dtype("bf16")
    with _set_fl(monkeypatch, "amd", "cuda"):
        assert PlatformFL.check_if_supports_dtype("bf16") is None


def test_platform_memory_helpers(monkeypatch):
    calls = []

    class _DeviceFn:
        def empty_cache(self):
            calls.append("empty")

        def set_device(self, device):
            calls.append(("set", device))

        def reset_peak_memory_stats(self, device):
            calls.append(("reset", device))

        def max_memory_allocated(self, device):
            calls.append(("max", device))
            return 1234

    monkeypatch.setattr(PlatformFL, "torch_device_fn", _DeviceFn(), raising=False)
    assert PlatformFL.get_current_memory_usage(0) == 1234
    assert calls[:3] == ["empty", ("reset", 0), ("max", 0)]

    zeros = []
    monkeypatch.setattr(
        fl, "torch", NS(zeros=lambda *a, **k: zeros.append((a, k)))
    )
    with _set_fl(monkeypatch, "nvidia", "cuda"):
        PlatformFL.set_device("dev0")
        assert zeros
    zeros.clear()
    with _set_fl(monkeypatch, "amd", "cuda"):
        PlatformFL.set_device("dev1")
        assert zeros == []
    PlatformFL.empty_cache()
    assert calls[-1] == "empty"
    with _set_fl(monkeypatch, "nvidia", "cuda", name="NVIDIA-A100"):
        assert PlatformFL.get_device_name(0) == "NVIDIA-A100"


def test_platform_import_ir_kernels(monkeypatch):
    mod = types.ModuleType("vllm.kernels")
    monkeypatch.setitem(sys.modules, "vllm.kernels", mod)
    PlatformFL.import_ir_kernels()
    assert sys.modules["vllm.kernels"] is mod


# ===========================================================================
# 5. PlatformFL backend selectors
# ===========================================================================
def test_platform_get_attn_backend_cls(monkeypatch):
    calls = []
    dispatch = sys.modules["vllm_fl.dispatch"]
    monkeypatch.setattr(
        dispatch, "call_op", lambda *a, **k: calls.append((a, k)) or "attn.Cls"
    )
    _quiet(monkeypatch)
    with _set_fl(monkeypatch, "nvidia", "cuda"):
        out = PlatformFL.get_attn_backend_cls(
            None, NS(use_mla=True, use_sparse=False)
        )
    assert out == "attn.Cls"
    assert calls[0][1] == {"use_mla": True, "use_sparse": False}
    with _set_fl(monkeypatch, "amd", "cuda"):
        with pytest.raises(NotImplementedError, match="Sparse MLA"):
            PlatformFL.get_attn_backend_cls(
                None, NS(use_mla=False, use_sparse=True)
            )


def test_platform_punica_and_communicator_cls(monkeypatch):
    _quiet(monkeypatch)
    assert PlatformFL.get_punica_wrapper().endswith("PunicaWrapperGPU")
    with _set_fl(monkeypatch, "iluvatar", "cuda", "flagcx"):
        assert "IluvatarCommunicatorFL" in PlatformFL.get_device_communicator_cls()
    with _set_fl(monkeypatch, "amd", "cuda", "flagcx"):
        assert "CudaFlagcxCommunicatorFL" in (
            PlatformFL.get_device_communicator_cls()
        )
    with _set_fl(monkeypatch, "nvidia", "cuda", "flagcx"):
        assert PlatformFL.get_device_communicator_cls().endswith(
            "CommunicatorFL"
        )
    with _set_fl(monkeypatch, "nvidia", "cuda", "nccl"):
        assert PlatformFL.get_device_communicator_cls().endswith(
            "CudaCommunicator"
        )


def test_platform_static_graph_wrapper_cls(monkeypatch):
    with _set_fl(monkeypatch, "mthreads", "musa"):
        assert "MUSAGraphWrapper" in PlatformFL.get_static_graph_wrapper_cls()
    with _set_fl(monkeypatch, "amd", "cuda"):
        monkeypatch.setattr(
            fl.importlib.import_module("vllm.config"),
            "get_current_vllm_config_or_none",
            lambda: None,
            raising=False,
        )
        assert PlatformFL.get_static_graph_wrapper_cls().endswith(
            "graph.GraphWrapper"
        )
        monkeypatch.setattr(
            fl.importlib.import_module("vllm.config"),
            "get_current_vllm_config_or_none",
            lambda: NS(use_v2_model_runner=True),
            raising=False,
        )
        assert PlatformFL.get_static_graph_wrapper_cls().endswith(
            "cuda_graph.CUDAGraphWrapper"
        )
    with _set_fl(monkeypatch, "nvidia", "cuda"):
        assert PlatformFL.get_static_graph_wrapper_cls().endswith(
            "graph.GraphWrapper"
        )


def test_platform_compile_backend_and_static_graph_mode(monkeypatch):
    base = PlatformFL.__mro__[1]
    monkeypatch.setattr(
        base,
        "get_compile_backend",
        classmethod(lambda cls: "super-backend"),
        raising=False,
    )
    with _set_fl(monkeypatch, "ascend", "npu"):
        assert PlatformFL.get_compile_backend() == "eager"
    with _set_fl(monkeypatch, "mthreads", "musa"):
        assert PlatformFL.get_compile_backend() == "eager"
    with _set_fl(monkeypatch, "nvidia", "cuda"):
        assert PlatformFL.get_compile_backend() == "super-backend"
        assert PlatformFL.support_static_graph_mode() is True
    with _set_fl(monkeypatch, "tsingmicro", "txda"):
        assert PlatformFL.support_static_graph_mode() is False


# ===========================================================================
# 6. PlatformFL block copies / hybrid / flag helpers
# ===========================================================================
class _FakeSlice:
    def __init__(self):
        self.to_called = None
        self.cpu_called = False

    def to(self, device):
        self.to_called = device
        return self

    def cpu(self):
        self.cpu_called = True
        return self


class _FakeCache:
    def __init__(self, device="raw"):
        self.device = device
        self.assigned = None
        self.slice = _FakeSlice()

    def __getitem__(self, idx):
        return self.slice

    def __setitem__(self, idx, value):
        self.assigned = value


def test_platform_block_copies():
    src = _FakeCache("cpu")
    dst = _FakeCache("cuda:0")
    PlatformFL.insert_blocks_to_device(src, dst, [0], [1])
    assert dst.assigned is src.slice
    assert src.slice.to_called == "cuda:0"

    src2 = _FakeCache("cuda:0")
    dst2 = _FakeCache("cpu")
    PlatformFL.swap_out_blocks_to_host(src2, dst2, [0], [1])
    assert dst2.assigned is src2.slice
    assert src2.slice.cpu_called is True


def test_platform_align_hybrid_block_size(monkeypatch):
    calls = []
    base = PlatformFL.__mro__[1]
    monkeypatch.setattr(
        base,
        "_align_hybrid_block_size",
        classmethod(lambda cls, cfg, backend: calls.append(("super", cfg))),
        raising=False,
    )
    _inject(
        "vllm_fl.kv_cache.ascend.hybrid",
        uses_dense_hybrid_cache=lambda mc: True,
        align_hybrid_block_size=lambda cfg: calls.append(("align", cfg)),
    )
    cfg = NS(model_config="mc")
    with _set_fl(monkeypatch, "ascend", "npu"):
        PlatformFL._align_hybrid_block_size(cfg, object())
    assert calls == [("align", cfg)]

    calls.clear()
    _inject(
        "vllm_fl.kv_cache.ascend.hybrid",
        uses_dense_hybrid_cache=lambda mc: False,
        align_hybrid_block_size=lambda cfg: calls.append(("align", cfg)),
    )
    with _set_fl(monkeypatch, "ascend", "npu"):
        PlatformFL._align_hybrid_block_size(cfg, object())
    assert calls == [("super", cfg)]
    calls.clear()
    with _set_fl(monkeypatch, "nvidia", "cuda"):
        PlatformFL._align_hybrid_block_size(cfg, object())
    assert calls == [("super", cfg)]


def test_platform_feature_flags(monkeypatch):
    assert PlatformFL.support_hybrid_kv_cache() is True
    assert PlatformFL.opaque_attention_op() is True
    with _set_fl(monkeypatch, "nvidia", "cuda", "nccl"):
        assert PlatformFL.use_custom_allreduce() is True
    with _set_fl(monkeypatch, "nvidia", "cuda", "flagcx"):
        assert PlatformFL.use_custom_allreduce() is False
    with _set_fl(monkeypatch, "hygon", "cuda", "nccl"):
        assert PlatformFL.use_custom_allreduce() is False
    with _set_fl(monkeypatch, "amd", "cuda", "nccl"):
        assert PlatformFL.use_custom_allreduce() is False


# ===========================================================================
# 7. PlatformFL pre-registration / ROCm / fp8 / IR priority
# ===========================================================================
def test_platform_pre_register_npu(monkeypatch):
    imported = []
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    monkeypatch.setattr(
        fl.importlib,
        "import_module",
        lambda name: imported.append(name) or types.ModuleType(name),
    )
    _quiet(monkeypatch)
    with _set_fl(monkeypatch, "ascend", "npu", name="npu"):
        PlatformFL.pre_register_and_update()
    assert os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"
    assert imported == ["vllm_fl.dispatch.backends.vendor.ascend"]


def test_platform_pre_register_iluvatar_and_amd(monkeypatch):
    calls = []
    _inject(
        "vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar",
        patch_triton_chained_or_for_iluvatar=lambda: calls.append("chained"),
    )
    _inject(
        "vllm_fl.dispatch.backends.vendor.iluvatar.moe_router",
        patch_moe_router=lambda: calls.append("router"),
    )
    with _set_fl(monkeypatch, "iluvatar", "cuda"):
        PlatformFL.pre_register_and_update()
    assert calls == ["chained", "router"]

    calls.clear()
    monkeypatch.setattr(
        PlatformFL,
        "_disable_rocm_native_kernels",
        classmethod(lambda cls: calls.append("rocm")),
        raising=False,
    )
    with _set_fl(monkeypatch, "amd", "cuda"):
        PlatformFL.pre_register_and_update()
    assert calls == ["rocm"]


def test_platform_disable_rocm_native_kernels(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_USE_SKINNY_GEMM", raising=False)
    monkeypatch.setattr(fl.importlib.util, "find_spec", lambda name: object())
    PlatformFL._disable_rocm_native_kernels()
    assert os.environ.get("VLLM_ROCM_USE_SKINNY_GEMM") is None

    _quiet(monkeypatch)
    monkeypatch.setattr(fl.importlib.util, "find_spec", lambda name: None)
    PlatformFL._disable_rocm_native_kernels()
    assert os.environ["VLLM_ROCM_USE_SKINNY_GEMM"] == "0"


def test_platform_supports_fp8(monkeypatch):
    monkeypatch.setattr(
        PlatformFL,
        "has_device_capability",
        classmethod(lambda cls, value: value == 89),
        raising=False,
    )
    with _set_fl(monkeypatch, "nvidia", "cuda"):
        assert PlatformFL.supports_fp8() is True
    with _set_fl(monkeypatch, "amd", "cuda"):
        assert PlatformFL.supports_fp8() is False


def test_platform_device_properties(monkeypatch):
    props = NS(total_memory=4096, multi_processor_count=32)
    monkeypatch.setattr(
        PlatformFL,
        "torch_device_fn",
        NS(get_device_properties=lambda i: props),
        raising=False,
    )
    assert PlatformFL.get_device_total_memory(0) == 4096
    assert PlatformFL.num_compute_units(0) == 32


def test_platform_custom_op_collectives(monkeypatch):
    cases = [
        ("tsingmicro", "txda", "nccl", True),
        ("nvidia", "cuda", "nccl", True),
        ("thead", "cuda", "nccl", True),
        ("iluvatar", "cuda", "nccl", True),
        ("mthreads", "musa", "mccl", True),
        ("mthreads", "musa", "nccl", False),
        ("amd", "cuda", "flagcx", True),
        ("amd", "cuda", "nccl", False),
        ("metax", "cuda", "nccl", False),
    ]
    for vendor, device, dist, expected in cases:
        with _set_fl(monkeypatch, vendor, device, dist):
            assert PlatformFL.use_custom_op_collectives() is expected


def test_platform_deep_gemm_and_arch_pdl(monkeypatch):
    with _set_fl(monkeypatch, "nvidia", "cuda"):
        monkeypatch.setattr(
            PlatformFL,
            "is_device_capability",
            classmethod(lambda cls, n: n == 90),
            raising=False,
        )
        monkeypatch.setattr(
            PlatformFL,
            "is_device_capability_family",
            classmethod(lambda cls, n: False),
            raising=False,
        )
        assert PlatformFL.support_deep_gemm() is True
        monkeypatch.setattr(
            PlatformFL,
            "is_device_capability",
            classmethod(lambda cls, n: False),
            raising=False,
        )
        assert PlatformFL.support_deep_gemm() is False
        monkeypatch.setattr(
            PlatformFL,
            "torch_device_fn",
            NS(
                current_device=lambda: 0,
                get_device_capability=lambda d: (9, 0),
            ),
            raising=False,
        )
        assert PlatformFL.is_arch_support_pdl() is True
        monkeypatch.setattr(
            PlatformFL,
            "torch_device_fn",
            NS(
                current_device=lambda: (_ for _ in ()).throw(RuntimeError("x")),
                get_device_capability=lambda d: (0, 0),
            ),
            raising=False,
        )
        assert PlatformFL.is_arch_support_pdl() is False
    with _set_fl(monkeypatch, "amd", "cuda"):
        assert PlatformFL.support_deep_gemm() is False
        assert PlatformFL.is_arch_support_pdl() is False


def test_platform_default_ir_op_priority(monkeypatch):
    rec = []

    class _IrOpPriorityConfig:
        @classmethod
        def with_default(cls, default, **kw):
            rec.append((default, kw))
            return ("cfg", tuple(default))

    monkeypatch.setitem(
        sys.modules,
        "vllm.config.kernel",
        types.SimpleNamespace(IrOpPriorityConfig=_IrOpPriorityConfig),
    )
    cc = NS(
        backend="inductor", mode=_CompilationMode.VLLM_COMPILE
    )
    cfg = NS(compilation_config=cc)
    with _set_fl(monkeypatch, "nvidia", "cuda"):
        assert PlatformFL.get_default_ir_op_priority(cfg) == ("cfg", ("native",))
    assert rec[-1][0] == ["native"]

    cc2 = NS(backend="eager", mode=_CompilationMode.NONE)
    monkeypatch.setattr(fl, "_vllm_c_available", lambda: True)
    with _set_fl(monkeypatch, "nvidia", "cuda"):
        assert PlatformFL.get_default_ir_op_priority(
            NS(compilation_config=cc2)
        ) == ("cfg", ("vllm_c", "native"))
    monkeypatch.setattr(fl, "_vllm_c_available", lambda: False)
    with _set_fl(monkeypatch, "amd", "cuda"):
        monkeypatch.setattr(
            PlatformFL,
            "_vllm_c_is_registered_provider",
            staticmethod(lambda: True),
            raising=False,
        )
        assert PlatformFL.get_default_ir_op_priority(
            NS(compilation_config=cc2)
        ) == ("cfg", ("vllm_c", "native"))
    with _set_fl(monkeypatch, "ascend", "npu"):
        assert PlatformFL.get_default_ir_op_priority(
            NS(compilation_config=cc2)
        ) == ("cfg", ("native",))
    assert rec[-1][1]["rms_norm"] == ["native"]


# ===========================================================================
# 8. PlatformFL seed / misc
# ===========================================================================
def test_platform_manual_seed_all(monkeypatch):
    calls = []
    with_native = NS(manual_seed_all=lambda seed: calls.append(("native", seed)))
    monkeypatch.setattr(PlatformFL, "torch_device_fn", with_native, raising=False)
    PlatformFL.manual_seed_all(7)
    assert calls == [("native", 7)]

    calls.clear()
    gens = {}

    class _Gen:
        def __init__(self, idx):
            self.idx = idx

        def manual_seed(self, seed):
            calls.append(("gen", self.idx, seed))

    device_fn = NS(
        device_count=lambda: 2,
        _get_or_create_default_generator=lambda idx: gens.setdefault(
            idx, _Gen(idx)
        ),
    )
    monkeypatch.setattr(PlatformFL, "torch_device_fn", device_fn, raising=False)
    monkeypatch.setattr(
        fl, "torch", NS(manual_seed=lambda seed: calls.append(("torch", seed)))
    )
    PlatformFL.manual_seed_all(11)
    assert calls == [("torch", 11), ("gen", 0, 11), ("gen", 1, 11)]


def test_platform_misc_flags():
    assert PlatformFL.is_integrated_gpu(0) is False
    assert PlatformFL.check_runner_kv_caches_multi_layer() is None


# ===========================================================================
# 9. WorkerFL.init_device
# ===========================================================================
def _worker_self(**over):
    parallel = NS(
        assigned_physical_gpu_ids=None,
        local_world_size=1,
        distributed_executor_backend="mp",
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        data_parallel_rank_local=None,
        data_parallel_index=0,
        enable_dbo=False,
    )
    worker = NS(
        device_config=NS(device_type="cuda", device="cuda:0"),
        parallel_config=parallel,
        device=NS(spec="cuda:0", type="cuda"),
        vllm_config=NS(
            parallel_config=parallel,
            speculative_config=None,
            model_config=NS(),
        ),
        local_rank=0,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:1",
        use_v2_model_runner=False,
        model_config=NS(dtype="float16", seed=7),
        cache_config=NS(kv_cache_memory_bytes=None),
    )
    worker.__dict__.update(over)
    return worker


def _platform_for_init(device="cuda", vendor="nvidia", calls=None):
    calls = calls if calls is not None else []

    def _device_id(idx):
        calls.append(("visible", idx))
        return idx

    return NS(
        device_type=device,
        vendor_name=vendor,
        dist_backend="nccl",
        logical_device_id_to_visible_device_id=_device_id,
        set_device=lambda dev: calls.append(("set", dev)),
        check_if_supports_dtype=lambda dt: calls.append(("dtype", dt)),
    )


def _patch_init(monkeypatch, platform, calls, device_count=8):
    monkeypatch.setattr(
        wk, "_apply_worker_vendor_patches", lambda: calls.append("patches")
    )
    monkeypatch.setattr(wk, "current_platform", platform)
    monkeypatch.setattr(
        wk.torch,
        "accelerator",
        NS(
            device_count=lambda: device_count,
            empty_cache=lambda: calls.append("empty"),
            synchronize=lambda: None,
            get_memory_info=lambda: (0, 0),
        ),
    )

    class _Dev:
        def __init__(self, spec):
            self.spec = spec
            self.type = str(spec).split(":")[0]

        def __str__(self):
            return self.spec

    monkeypatch.setattr(wk.torch, "device", _Dev)
    monkeypatch.setattr(
        wk,
        "init_worker_distributed_environment",
        lambda *a: calls.append(("dist", a)),
    )
    monkeypatch.setattr(
        wk, "set_random_seed", lambda seed: calls.append(("seed", seed))
    )
    monkeypatch.setattr(wk, "gc", NS(collect=lambda: calls.append("gc")))
    monkeypatch.setattr(wk, "MemorySnapshot", lambda device: NS(device=device))
    monkeypatch.setattr(wk, "request_memory", lambda snap, cache: 1234)
    monkeypatch.setattr(
        wk, "init_workspace_manager", lambda *a: calls.append(("ws", a))
    )
    monkeypatch.setattr(
        wk, "report_usage_stats", lambda cfg: calls.append("stats")
    )


def test_worker_init_device_rejects_mismatch(monkeypatch):
    worker = _worker_self()
    platform = _platform_for_init(device="npu", vendor="ascend")
    calls = []
    _patch_init(monkeypatch, platform, calls)
    with pytest.raises(RuntimeError, match="Unsupported device type"):
        WorkerFL.init_device(worker)


def test_worker_init_device_v1_assigned_ids(monkeypatch):
    calls = []
    worker = _worker_self()
    worker.parallel_config.assigned_physical_gpu_ids = [0, 1]
    worker.parallel_config.local_world_size = 2
    platform = _platform_for_init(calls=calls)
    _patch_init(monkeypatch, platform, calls)
    monkeypatch.setenv("NCCL_ASYNC_ERROR_HANDLING", "1")

    runner_calls = []
    _inject(
        "vllm.worker_runner_fl_stub",
        ModelRunnerFL=lambda cfg, dev: runner_calls.append((cfg, dev))
        or NS(dev=dev),
    )
    _inject(
        "vllm_fl.worker.model_runner",
        ModelRunnerFL=lambda cfg, dev: runner_calls.append((cfg, dev))
        or NS(dev=dev),
    )
    iface = sys.modules["vllm.platforms.interface"]
    monkeypatch.setattr(
        iface,
        "set_assigned_physical_gpu_ids",
        lambda ids: calls.append(("assigned", ids)),
        raising=False,
    )

    WorkerFL.init_device(worker)
    assert "NCCL_ASYNC_ERROR_HANDLING" not in os.environ
    assert worker.local_rank == 0
    assert ("assigned", [0, 1]) in calls
    assert worker.device.spec == "cuda:0"
    assert ("dtype", "float16") in calls
    assert worker.init_snapshot.device is worker.device
    assert worker.requested_memory == 1234
    assert runner_calls and runner_calls[0][0] is worker.vllm_config
    assert ("ws", (worker.device, 1, 1)) in calls
    assert "stats" in calls


def test_worker_init_device_adjusts_dp_local_rank(monkeypatch):
    calls = []
    worker = _worker_self()
    worker.parallel_config.local_world_size = 4
    worker.parallel_config.data_parallel_rank_local = 1
    worker.parallel_config.assigned_physical_gpu_ids = None
    platform = _platform_for_init(calls=calls)
    _patch_init(monkeypatch, platform, calls, device_count=8)
    _inject(
        "vllm_fl.worker.model_runner",
        ModelRunnerFL=lambda cfg, dev: NS(dev=dev),
    )
    WorkerFL.init_device(worker)
    assert worker.local_rank == 4
    assert worker.device.spec == "cuda:4"
    assert ("visible", 4) in calls


def test_worker_init_device_v2_amd_runner(monkeypatch):
    calls = []
    worker = _worker_self(use_v2_model_runner=True)
    platform = _platform_for_init(vendor="amd")
    _patch_init(monkeypatch, platform, calls)
    worker.parallel_config.assigned_physical_gpu_ids = None
    worker.parallel_config.local_world_size = 8
    worker.parallel_config.data_parallel_index = 0
    installed = []
    _inject(
        "vllm_fl.dispatch.backends.vendor.amd.v2_runner_compat",
        install_v2_runner_compat=lambda: installed.append("amd"),
    )
    made = []
    gpu_mod = types.ModuleType("vllm.v1.worker.gpu.model_runner")
    gpu_mod.GPUModelRunner = lambda cfg, dev: made.append((cfg, dev)) or NS(
        dev=dev
    )
    monkeypatch.setitem(
        sys.modules, "vllm.v1.worker.gpu.model_runner", gpu_mod
    )
    WorkerFL.init_device(worker)
    assert installed == ["amd"]
    assert made and made[0][1] is worker.device


def test_worker_init_device_v2_npu_runner(monkeypatch):
    calls = []
    worker = _worker_self(use_v2_model_runner=True)
    platform = _platform_for_init(device="npu", vendor="ascend")
    _patch_init(monkeypatch, platform, calls)
    worker.device_config.device_type = "npu"
    worker.parallel_config.assigned_physical_gpu_ids = None
    worker.parallel_config.local_world_size = 8
    worker.parallel_config.data_parallel_index = 0
    shims = []
    _inject(
        "vllm_fl.dispatch.backends.vendor.ascend.patches",
        ascend_v2_device=NS(
            install_ascend_v2_device_shims=lambda: shims.append("shim")
        ),
    )
    _inject(
        "vllm_fl.dispatch.backends.vendor.ascend.impl.triton_utils",
        init_device_properties_triton=lambda: calls.append("triton-props"),
    )
    made = []
    _inject(
        "vllm_fl.dispatch.backends.vendor.ascend.worker.model_runner_v2",
        GPUModelRunnerFLV2=lambda cfg, dev: made.append((cfg, dev))
        or NS(dev=dev),
    )
    WorkerFL.init_device(worker)
    assert shims == ["shim"]
    assert "triton-props" in calls
    assert made and made[0][1] is worker.device


# ===========================================================================
# 10. WorkerFL.determine_available_memory residual branches
# ===========================================================================
def _determine_worker(**over):
    worker = NS(
        cache_config=NS(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.9
        ),
        init_snapshot=NS(free_memory=100, total_memory=200),
        vllm_config=NS(
            compilation_config=NS(cudagraph_mode=_CUDAGraphMode.PIECEWISE)
        ),
        model_runner=NS(
            profile_run=lambda: None,
            profile_cudagraph_memory=lambda: 5,
            model_memory_usage=10,
        ),
        model_config=NS(multimodal_config=None),
        parallel_config=NS(_api_process_count=1),
        requested_memory=10,
        use_v2_model_runner=False,
        device=NS(spec="musa:0", type="musa"),
    )
    worker.__dict__.update(over)
    return worker


def _profile_result(free=80):
    return NS(
        total_consumed=10,
        transient_peak_headroom=5,
        after_profile=NS(free_memory=free),
        non_kv_cache_memory=20,
    )


def _run_determine(monkeypatch, worker, result, platform, env_flag):
    logs = _logger()
    info, warning, error = [], [], []
    logs.info = lambda *a, **k: info.append(a)
    logs.warning = lambda *a, **k: warning.append(a)
    logs.error = lambda *a, **k: error.append(a)
    init_logs = _logger()
    init_logs.info = lambda *a, **k: info.append(a)
    monkeypatch.setattr(wk, "logger", logs)
    monkeypatch.setattr(wk, "init_logger", lambda name: init_logs)
    monkeypatch.setattr(wk, "current_platform", platform)
    monkeypatch.setattr(wk, "maybe_apply_startup_plan", lambda w: None)
    monkeypatch.setattr(
        wk, "memory_profiling", lambda *a, **k: _NullCtx(result)
    )
    monkeypatch.setattr(
        wk,
        "envs",
        NS(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=env_flag),
    )
    reserved = []
    monkeypatch.setattr(
        wk,
        "reserve_mm_ipc_gpu_memory",
        lambda *a: reserved.append(a) or 777,
    )
    out = WorkerFL.determine_available_memory(worker)
    return out, reserved, info, warning, error


class _NullCtx:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return False


def test_worker_determine_musa_logs_and_error(monkeypatch):
    calls = []
    platform = NS(
        device_type="musa",
        vendor_name="mthreads",
        is_cuda_alike=lambda: True,
        torch_device_fn=NS(
            mem_get_info=lambda dev: calls.append(dev) or (256, 1024)
        ),
    )
    worker = _determine_worker()
    worker.requested_memory = 10
    out, reserved, info, warning, error = _run_determine(
        monkeypatch, worker, _profile_result(80), platform, True
    )
    assert out == 777
    assert calls == [worker.device if hasattr(worker, "device") else None]
    assert error, "negative MUSA budget must be logged as an error"
    assert worker.available_kv_cache_memory_bytes == -15
    assert reserved[0][0] == -15
    assert info


def test_worker_determine_musa_positive_and_disabled_warning(monkeypatch):
    platform = NS(
        device_type="musa",
        vendor_name="mthreads",
        is_cuda_alike=lambda: True,
        torch_device_fn=NS(mem_get_info=lambda dev: (512, 1024)),
    )
    worker = _determine_worker()
    worker.requested_memory = 100
    out, reserved, info, warning, error = _run_determine(
        monkeypatch, worker, _profile_result(80), platform, False
    )
    assert out == 777
    assert not error
    assert warning, "disabled graph estimate should warn"
    assert worker.available_kv_cache_memory_bytes == 80
    assert reserved[0][0] == 80


def test_worker_determine_profiling_assertion(monkeypatch):
    platform = NS(
        device_type="cuda",
        vendor_name="nvidia",
        is_cuda_alike=lambda: True,
    )
    worker = _determine_worker()
    with pytest.raises(AssertionError, match="Error in memory profiling"):
        _run_determine(
            monkeypatch,
            worker,
            NS(
                total_consumed=0,
                transient_peak_headroom=0,
                after_profile=NS(free_memory=120),
                non_kv_cache_memory=0,
            ),
            platform,
            False,
        )


def test_worker_determine_amd_warmup_and_profile_config(monkeypatch):
    platform = NS(
        device_type="cuda",
        vendor_name="amd",
        is_cuda_alike=lambda: False,
    )
    worker = _determine_worker()
    worker.vllm_config.compilation_config.cudagraph_mode = (
        _CUDAGraphMode.NONE
    )
    worker.use_v2_model_runner = True
    worker.requested_memory = 100
    warmups = []
    configs = []
    dispatch = sys.modules["vllm_fl.dispatch"]
    monkeypatch.setattr(
        dispatch,
        "warmup_cached_ops",
        lambda **k: warmups.append(k),
        raising=False,
    )
    monkeypatch.setattr(
        wk,
        "set_current_vllm_config",
        lambda cfg: configs.append(cfg) or _NullCtx("cfg"),
    )
    out, reserved, info, warning, error = _run_determine(
        monkeypatch, worker, _profile_result(80), platform, False
    )
    assert out == 777
    assert warmups == [{"amd_compile": True}]
    assert configs == [worker.vllm_config]
    assert worker.available_kv_cache_memory_bytes == 80
    assert reserved[0][0] == 80


# ===========================================================================
# 11. WorkerFL allocator restore + profiler reuse
# ===========================================================================
def test_worker_scoped_allocator_restores_size_max(monkeypatch):
    calls = []
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(
        wk,
        "current_platform",
        NS(is_cuda=lambda: True, device_type="cuda", vendor_name="nvidia"),
    )
    monkeypatch.setattr(
        wk,
        "torch",
        NS(
            _C=NS(
                _accelerator_setAllocatorSettings=lambda value: calls.append(
                    value
                )
            )
        ),
    )
    with WorkerFL._scoped_allocator_max_split(None, 20):
        pass
    size_max = (2**64 - 1) // (1024 * 1024)
    assert calls == [
        "max_split_size_mb:20",
        f"max_split_size_mb:{size_max}",
    ]


def test_worker_profile_reuses_existing_profiler(monkeypatch):
    calls = []
    monkeypatch.setattr(wk, "logger", _logger())
    _inject(
        "vllm.distributed.utils",
        get_worker_rank_suffix=lambda global_rank=None: f"rank{global_rank}",
    )
    worker = NS(
        profiler_config=NS(profiler="torch"),
        profiler=NS(start=lambda: calls.append("start")),
        rank=3,
        local_rank=0,
    )
    WorkerFL.profile(worker, is_start=True, profile_prefix=None)
    assert calls == ["start"]
    assert worker.profiler is not None


def test_worker_get_device_memory_pool_xpu_sleep_enabled(monkeypatch):
    used = []

    class _Alloc:
        def get_current_usage(self):
            return 0

        def use_memory_pool(self, tag):
            used.append(tag)
            return _NullCtx(tag)

    worker = NS(
        vllm_config=NS(
            model_config=NS(enable_sleep_mode=True)
        )
    )
    platform = NS(
        device_type="xpu",
        vendor_name="intel",
        is_cuda_alike=lambda: False,
        is_xpu=lambda: True,
        is_cpu=lambda: False,
        is_sleep_mode_available=lambda: True,
    )
    monkeypatch.setattr(wk, "current_platform", platform)
    monkeypatch.setattr(
        wk, "get_mem_allocator_instance", lambda: _Alloc(), raising=False
    )
    with WorkerFL._maybe_get_memory_pool_context(worker, tag="kv_cache") as val:
        assert val == "kv_cache"
    assert used == ["kv_cache"]
