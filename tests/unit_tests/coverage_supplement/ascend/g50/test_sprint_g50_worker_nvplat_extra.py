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

"""G50 residual coverage for ``worker.py`` + ``nvidia_platform.py``.

Local-only, no accelerator, no network, no ``hc``/SSH.  Both modules are
loaded straight from source (never through the ``vllm_fl`` package chain,
whose ``DeviceInfo`` breaks on the 910C image) using the G41-proven scaffold.
"""

from __future__ import annotations

import enum
import importlib
import importlib.util
import os
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


# ===========================================================================
# 1. torch stub hardening (also fixes the pure-python torchstub locally)
# ===========================================================================
def _ensure_torch_stub() -> None:
    try:
        import torch  # noqa: F401

        t = sys.modules["torch"]
    except Exception:
        t = sys.modules.get("torch") or types.ModuleType("torch")
        sys.modules["torch"] = t

    nn_attr = getattr(t, "nn", None)
    if "torch.nn" not in sys.modules:
        nn = types.ModuleType("torch.nn")
        nn.Module = getattr(nn_attr, "Module", type("Module", (), {}))
        sys.modules["torch.nn"] = nn
        parent = sys.modules.get("torch")
        if parent is not None:
            try:
                parent.nn = nn
            except Exception:
                pass
    if not hasattr(t, "nn"):
        t.nn = sys.modules["torch.nn"]
    if not hasattr(t, "Tensor"):
        t.Tensor = type("Tensor", (), {})
    if not hasattr(t, "dtype"):
        t.dtype = type("dtype", (), {})
    if not hasattr(t, "device"):

        class _Device:
            def __init__(self, spec="cpu"):
                self.spec = str(spec)
                self.type = self.spec.split(":")[0]

        t.device = _Device
    if not hasattr(t, "accelerator"):
        t.accelerator = NS(
            device_count=lambda: 0,
            empty_cache=lambda: None,
            synchronize=lambda: None,
            get_memory_info=lambda: (0, 0),
        )
    if not hasattr(t, "set_float32_matmul_precision"):
        t.set_float32_matmul_precision = lambda *a, **k: None
    for name in ("bfloat16", "float16", "float32", "float64", "int8", "int64"):
        if not hasattr(t, name):
            setattr(t, name, type("_dt", (), {"name": name})())


# ===========================================================================
# 2. deterministic fakes / vLLM stub table
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


class _CompilationMode(enum.Enum):
    NONE = "NONE"
    VLLM_COMPILE = "VLLM_COMPILE"


class _PlatformEnum:
    OOT = "OOT"
    CUDA = "CUDA"


class _Platform:
    @classmethod
    def import_kernels(cls):
        return None


class _CudaPlatform:
    dist_backend = "nccl"

    @classmethod
    def check_and_update_config(cls, vllm_config):
        return None

    @classmethod
    def get_device_communicator_cls(cls):
        return "vllm.distributed.cuda.CudaCommunicator"

    @classmethod
    def use_custom_allreduce(cls):
        return True

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, selector, num_heads=None):
        return "native.attn"


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


class _MemorySnapshot:
    def __init__(self, device=None, **kw):
        self.device = device
        self.free_memory = kw.get("free_memory", 1 << 30)
        self.total_memory = kw.get("total_memory", 2 << 30)


class _ModelRunnerOutput:
    pass


class _AsyncModelRunnerOutput:
    pass


def _deco(*a, **k):
    def wrap(fn):
        return fn

    return wrap


def _device_info():
    return NS(
        vendor_name="",
        device_type="",
        dispatch_key="",
        torch_device_fn=sys.modules["torch"],
        device=object(),
    )


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
)


def _stub_table():
    return {
        "vllm.logger": {"init_logger": lambda name: _logger()},
        "vllm.envs": {"VLLM_FLOAT32_MATMUL_PRECISION": "high"},
        "vllm.config": {
            "CUDAGraphMode": _CUDAGraphMode,
            "VllmConfig": type("VllmConfig", (), {}),
            "set_current_vllm_config": lambda cfg: nullcontext(),
            "get_current_vllm_config_or_none": lambda: None,
        },
        "vllm.config.compilation": {"CompilationMode": _CompilationMode},
        "vllm.config.cache": {"CacheDType": object},
        "vllm.platforms": {
            "Platform": _Platform,
            "PlatformEnum": _PlatformEnum,
            "current_platform": _CURRENT_PLATFORM,
        },
        "vllm.platforms.interface": {
            "DeviceCapability": object,
            "set_assigned_physical_gpu_ids": lambda ids: None,
        },
        "vllm.platforms.cuda": {"CudaPlatform": _CudaPlatform},
        "vllm.v1.core.sched.output": {
            "GrammarOutput": object,
            "SchedulerOutput": object,
        },
        "vllm.v1.kv_cache_interface": {
            "KVCacheConfig": object,
            "KVCacheSpec": object,
        },
        "vllm.v1.outputs": {
            "AsyncModelRunnerOutput": _AsyncModelRunnerOutput,
            "DraftTokenIds": object,
            "ModelRunnerOutput": _ModelRunnerOutput,
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
        "vllm.v1.worker.gpu.model_runner": {"GPUModelRunner": object},
        "vllm.v1.worker.sentinel.gpu_worker_sentinel": {"WorkerSentinel": object},
        "vllm.v1.worker.startup_plan": {
            "maybe_apply_startup_plan": lambda w: None,
            "maybe_save_startup_plan": lambda w, v: None,
        },
        "vllm.v1.worker.utils": {
            "request_memory": lambda s, c: 0,
            "is_residual_scattered_for_sp": lambda *a, **k: False,
        },
        "vllm.v1.engine": {
            "ReconfigureDistributedRequest": object,
            "ReconfigureRankType": object,
        },
        "vllm.v1.worker.worker_base": {
            "CompilationTimes": _CompilationTimes,
            "WorkerBase": _WorkerBase,
        },
        "vllm.v1.worker.workspace": {"init_workspace_manager": lambda *a: None},
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
            "SleepModeBackend": object,
            "SleepModeBackendFactory": NS(create_backend=lambda cfg: None),
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
            "KVConnectorHandshakeMetadata": object
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
            "WeightTransferEngine": object,
            "WeightTransferEngineFactory": NS(create_engine=lambda *a, **k: None),
        },
        "vllm.distributed.utils": {
            "get_worker_rank_suffix": lambda global_rank=None: f"r{global_rank}"
        },
        "vllm.lora.request": {"LoRARequest": object},
        "vllm.model_executor.model_loader": {
            "TensorizerLoader": NS(save_model=lambda *a, **k: None),
            "ShardedStateLoader": NS(save_model=lambda *a, **k: None),
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
        "vllm.tasks": {"SupportedTask": object},
        "vllm.tracing": {"instrument": _deco},
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
            "memory_profiling": lambda *a, **k: nullcontext(),
        },
        "vllm.utils.torch_utils": {
            "set_random_seed": lambda seed: None,
            "set_torch_threads_for_runtime": lambda: None,
        },
        "vllm.utils.jit_monitor": {"activate": lambda *a, **k: None},
        "vllm.compilation.compiler_interface": {
            "trigger_inductor_lazy_init": lambda dev: None
        },
        "vllm.compilation.passes.vllm_inductor_pass": {
            "get_match_table": lambda: {}
        },
        "vllm_fl.utils": {"DeviceInfo": _device_info},
        "vllm_fl.envs": {},
        "vllm_fl.dispatch": {
            "call_op": lambda *a, **k: "backend",
            "warmup_cached_ops": lambda *a, **k: None,
        },
        "vllm_fl.dispatch.io_common": {"managed_inference_mode": _deco},
        "vllm_fl.worker.runtime": {"_initialize_fl_runtime": lambda r: None},
        "vllm_fl.worker.model_runner": {
            "ModelRunnerFL": type(
                "ModelRunnerFL",
                (),
                {"__init__": lambda self, cfg, dev: None},
            )
        },
        "vllm_fl.worker.scheduler_fl": {},
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
    mod = types.ModuleType(name)
    mod.__path__ = []
    for key, value in attrs.items():
        setattr(mod, key, value)
    parts = name.split(".")
    for i in range(1, len(parts)):
        _ensure_pkg(".".join(parts[:i]))
    sys.modules[name] = mod
    if len(parts) > 1:
        setattr(sys.modules[".".join(parts[:-1])], parts[-1], mod)
    return mod


def _install_stubs():
    for name in (
        "vllm", "vllm.config", "vllm.platforms", "vllm.v1", "vllm.v1.core",
        "vllm.v1.attention", "vllm.v1.attention.backends", "vllm.v1.worker",
        "vllm.distributed", "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1", "vllm.utils",
        "vllm.model_executor", "vllm.lora", "vllm.multimodal",
        "vllm.profiler", "vllm.compilation", "vllm.compilation.passes",
        "vllm_fl", "vllm_fl.worker", "vllm_fl.dispatch", "vllm_fl.attention",
    ):
        _ensure_pkg(name)
    for name, attrs in _stub_table().items():
        _force_module(name, attrs)


def _inject(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    parts = name.split(".")
    for i in range(1, len(parts)):
        _ensure_pkg(".".join(parts[:i]))
    sys.modules[name] = mod
    if len(parts) > 1:
        setattr(sys.modules[".".join(parts[:-1])], parts[-1], mod)
    return mod


# ===========================================================================
# 3. direct-source loader
# ===========================================================================
_SRC = {
    "vllm_fl/nvidia_platform.py": ("VLLM_FL_NVIDIA_PLATFORM_SRC",),
    "vllm_fl/worker/worker.py": ("VLLM_FL_WORKER_SRC",),
}


def _find_source(rel):
    cands = []
    for var in (
        *_SRC.get(rel, ()),
        "VLLM_FL_SRC_ROOT",
        "VLLM_FL_WORKTREE",
        "FL_SRC_ROOT",
    ):
        root = os.environ.get(var)
        if root:
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
    raise FileNotFoundError(f"{rel} not found")


def _load(module_name, rel, checker):
    for key in list(sys.modules):
        if key == module_name or key.startswith(module_name + "."):
            sys.modules.pop(key, None)
    path = _find_source(rel)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    parent_name, _, child = module_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child, mod)
    spec.loader.exec_module(mod)
    if not checker(mod):
        raise ImportError(f"{path} missing expected symbols")
    return mod


_LOAD_ERROR = None
nvplat = wk = None
try:
    _ensure_torch_stub()
    _install_stubs()
    nvplat = _load(
        "vllm_fl.nvidia_platform",
        "vllm_fl/nvidia_platform.py",
        lambda m: hasattr(m, "NvidiaPlatformFL"),
    )
    wk = _load(
        "vllm_fl.worker.worker",
        "vllm_fl/worker/worker.py",
        lambda m: hasattr(m, "WorkerFL"),
    )
except Exception as exc:  # pragma: no cover - environment guard
    _LOAD_ERROR = exc


@pytest.fixture(autouse=True)
def _require_modules():
    if nvplat is None or wk is None:
        pytest.fail(f"vllm_fl modules unavailable: {_LOAD_ERROR!r}")


def _make_worker(**attrs):
    """Real WorkerFL instance (bypassing __init__) so bound methods resolve."""
    w = object.__new__(wk.WorkerFL)
    w.__dict__.update(attrs)
    return w


def _quiet(monkeypatch, module):
    log = _logger()
    seen = {"info": [], "warning": [], "warning_once": [], "debug": []}
    for name in seen:
        setattr(log, name, lambda *a, _n=name, **k: seen[_n].append(a))
    monkeypatch.setattr(module, "logger", log)
    return seen


# ===========================================================================
# 4. nvidia_platform.py
# ===========================================================================
def test_nvplat_refresh_dist_backend(monkeypatch):
    P = nvplat.NvidiaPlatformFL
    monkeypatch.setenv("FLAGCX_PATH", "/opt/flagcx")
    assert P._refresh_dist_backend() is True
    assert P.dist_backend == "flagcx"
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    assert P._refresh_dist_backend() is False
    assert P.dist_backend == "flagcx" or P.dist_backend == "nccl"
    assert P.dist_backend == "nccl"


def test_nvplat_check_and_update_config_flagcx(monkeypatch):
    P = nvplat.NvidiaPlatformFL
    monkeypatch.setenv("FLAGCX_PATH", "/opt/flagcx")
    seen = _quiet(monkeypatch, nvplat)
    base_calls = []
    monkeypatch.setattr(
        nvplat.CudaPlatform,
        "check_and_update_config",
        classmethod(lambda cls, cfg: base_calls.append(cfg)),
    )
    pc = NS(disable_custom_all_reduce=False)
    passes = NS(fuse_allreduce_rms=True)
    cfg = NS(parallel_config=pc, compilation_config=NS(pass_config=passes))
    P.check_and_update_config(cfg)
    assert base_calls == [cfg]
    assert pc.disable_custom_all_reduce is True
    assert passes.fuse_allreduce_rms is False
    assert seen["warning_once"], "fuse_allreduce_rms disable must warn"
    assert pc.worker_cls == "vllm_fl.worker.nvidia_worker.NvidiaWorkerFL"

    # native (no FLAGCX_PATH) keeps upstream custom all-reduce / fused pass
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    seen["warning_once"].clear()
    pc2 = NS(disable_custom_all_reduce=False)
    passes2 = NS(fuse_allreduce_rms=True)
    cfg2 = NS(parallel_config=pc2, compilation_config=NS(pass_config=passes2))
    P.check_and_update_config(cfg2)
    assert pc2.disable_custom_all_reduce is False
    assert passes2.fuse_allreduce_rms is True
    assert not seen["warning_once"]
    assert pc2.worker_cls.endswith("NvidiaWorkerFL")


def test_nvplat_communicator_and_allreduce(monkeypatch):
    P = nvplat.NvidiaPlatformFL
    monkeypatch.setenv("FLAGCX_PATH", "/opt/flagcx")
    assert P.get_device_communicator_cls() == (
        "vllm_fl.distributed.nvidia_communicator.NvidiaCommunicatorFL"
    )
    assert P.use_custom_allreduce() is False
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    assert P.get_device_communicator_cls() == (
        "vllm.distributed.cuda.CudaCommunicator"
    )
    assert P.use_custom_allreduce() is True


def test_nvplat_attn_backend_flagcx_and_native(monkeypatch):
    P = nvplat.NvidiaPlatformFL
    calls = []
    _inject(
        "vllm_fl.dispatch.backends.flaggems.flaggems",
        FlagGemsBackend=lambda: NS(
            attention_backend=lambda use_mla, use_sparse: calls.append(
                (use_mla, use_sparse)
            )
            or "flaggems.Attn"
        ),
    )
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", "yes")
    assert (
        P.get_attn_backend_cls(None, NS(use_mla=True, use_sparse=False), 8)
        == "flaggems.Attn"
    )
    assert calls == [(True, False)]

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", "0")
    base = []
    monkeypatch.setattr(
        nvplat.CudaPlatform,
        "get_attn_backend_cls",
        classmethod(
            lambda cls, sel, cfg, num_heads=None: base.append(
                (sel, cfg, num_heads)
            )
            or "native.Attn"
        ),
    )
    sel_cfg = NS(use_mla=False, use_sparse=False)
    assert P.get_attn_backend_cls("sel", sel_cfg, None) == "native.Attn"
    assert base == [("sel", sel_cfg, None)]


# ===========================================================================
# 5. worker.py module helpers
# ===========================================================================
def test_worker_module_helpers(monkeypatch):
    # _num_workspace_lanes
    cfg = NS(speculative_config=NS(use_dspark=lambda: True))
    assert wk._num_workspace_lanes(cfg, True) == 2
    assert wk._num_workspace_lanes(cfg, False) == 1
    assert wk._num_workspace_lanes(NS(speculative_config=None), True) == 1

    # _normalize_process_group_backend
    cases = [
        (NS(device_type="ptpu", vendor_name="x"), "flagcx", "pccl"),
        (NS(device_type="ptpu", vendor_name="x"), "nccl", "pccl"),
        (NS(device_type="ptpu", vendor_name="x"), "gloo", "gloo"),
        (NS(device_type="npu", vendor_name="ascend"), "flagcx", "hccl"),
        (NS(device_type="cuda", vendor_name="amd"), "flagcx", "nccl"),
        (NS(device_type="cuda", vendor_name="nvidia"), "nccl", "nccl"),
    ]
    for platform, backend, expected in cases:
        monkeypatch.setattr(wk, "current_platform", platform)
        assert wk._normalize_process_group_backend(backend) == expected

    # _is_hybrid_kv_model
    assert wk._is_hybrid_kv_model(
        NS(cache_config=NS(mamba_cache_mode="align"))
    ) is True
    assert wk._is_hybrid_kv_model(
        NS(cache_config=NS(mamba_cache_mode="all"))
    ) is True
    assert wk._is_hybrid_kv_model(
        NS(cache_config=NS(mamba_cache_mode="none"),
           model_config=NS(is_hybrid=True))
    ) is True

    class _Bad:
        @property
        def is_hybrid(self):
            raise RuntimeError("unresolved")

    assert wk._is_hybrid_kv_model(
        NS(cache_config=None, model_config=_Bad())
    ) is False
    assert wk._is_hybrid_kv_model(
        NS(
            cache_config=None,
            model_config=NS(
                hf_text_config=NS(layer_types=["linear_attention"])
            ),
        )
    ) is True
    assert wk._is_hybrid_kv_model(
        NS(
            cache_config=None,
            model_config=NS(hf_text_config=NS(layer_types=["full_attention"])),
        )
    ) is False


def test_worker_async_intermediate_tensors():
    waited = []
    w = wk.AsyncIntermediateTensors(
        {"a": 1},
        comm_handles=[NS(wait=lambda: waited.append("h"))],
        comm_postprocess=[lambda: waited.append("p")],
    )
    assert w.tensors == {"a": 1}
    assert waited == ["h", "p"]
    assert w.tensors == {"a": 1}
    assert waited == ["h", "p"], "wait_for_comm must be idempotent"
    assert w._comm_waited is True

    plain = wk.AsyncIntermediateTensors({"b": 2})
    plain.wait_for_comm()
    plain.wait_for_comm()
    assert plain.tensors == {"b": 2}


# ===========================================================================
# 6. worker sleep / wake
# ===========================================================================
class _Buf:
    def __init__(self, name):
        self.name = name
        self.data = NS(copy_=lambda v: None)

    def cpu(self):
        return self

    def clone(self):
        return self


class _Model:
    def __init__(self):
        self.buffers = [_Buf("w")]

    def named_buffers(self):
        return [(buf.name, buf) for buf in self.buffers]


def _sleep_worker(backend, model, draft=None):
    return _make_worker(
        model_runner=NS(
            model=model,
            get_draft_model=lambda: draft,
            post_kv_cache_wake_up=lambda: setattr(backend, "kv", True),
        ),
        _sleep_mode_backend=backend,
        _sleep_saved_buffers={},
        _sleep_saved_draft_buffers={},
    )


def test_worker_sleep_level2_saves_buffers(monkeypatch):
    calls = []
    infos = iter([(100, 1000), (250, 1000)])
    monkeypatch.setattr(
        wk,
        "torch",
        NS(
            accelerator=NS(
                synchronize=lambda: calls.append("sync"),
                get_memory_info=lambda: next(infos),
            )
        ),
    )
    monkeypatch.setattr(wk, "current_platform", NS(is_rocm=lambda: False))
    seen = _quiet(monkeypatch, wk)
    backend = NS(suspend=lambda level: calls.append(("suspend", level)))
    worker = _sleep_worker(backend, _Model(), draft=_Model())
    wk.WorkerFL.sleep(worker, level=2)
    assert ("suspend", 2) in calls
    assert worker._sleep_saved_buffers and "w" in worker._sleep_saved_buffers
    assert worker._sleep_saved_draft_buffers and "w" in (
        worker._sleep_saved_draft_buffers
    )
    assert seen["info"]

    # _get_sleep_mode_backend lazily builds and caches the backend
    made = []
    sleeps_mod = sys.modules["vllm.device_allocator.sleep_mode_backend"]
    monkeypatch.setattr(
        sleeps_mod.SleepModeBackendFactory,
        "create_backend",
        lambda cfg: made.append(cfg) or backend,
    )
    lazy = _make_worker(vllm_config=NS(model_config="mc"),
                        _sleep_mode_backend=None)
    assert wk.WorkerFL._get_sleep_mode_backend(lazy) is backend
    assert made == ["mc"]
    assert lazy._sleep_mode_backend is backend


def test_worker_sleep_rocm_deadline_and_assert(monkeypatch):
    sleeps = []
    backend = NS(suspend=lambda level: None)
    monkeypatch.setattr(wk, "current_platform", NS(is_rocm=lambda: True))
    monkeypatch.setattr(
        wk, "time", NS(monotonic=lambda: 0.0, sleep=lambda s: sleeps.append(s))
    )
    _quiet(monkeypatch, wk)

    infos = iter([(100, 1000), (90, 1000), (150, 1000)])
    monkeypatch.setattr(
        wk,
        "torch",
        NS(accelerator=NS(synchronize=lambda: None,
                          get_memory_info=lambda: next(infos))),
    )
    worker = _sleep_worker(backend, _Model())
    wk.WorkerFL.sleep(worker, level=1)
    assert sleeps == [0.1]

    infos = iter([(100, 1000), (50, 1000)])
    monos = iter([0.0])
    monkeypatch.setattr(
        wk,
        "time",
        NS(monotonic=lambda: next(monos, 10.0), sleep=lambda s: sleeps.append(s)),
    )
    monkeypatch.setattr(
        wk,
        "torch",
        NS(accelerator=NS(synchronize=lambda: None,
                          get_memory_info=lambda: next(infos))),
    )
    with pytest.raises(AssertionError, match="Memory usage increased"):
        wk.WorkerFL.sleep(worker, level=1)


def test_worker_wake_up_restores_buffers(monkeypatch):
    resumed = []
    copied = []
    draft_copied = []

    class _B:
        def __init__(self, tag):
            self.data = tag

    model = NS(
        named_buffers=lambda: [
            ("w", NS(data=NS(copy_=copied.append)))
        ]
    )
    draft = NS(
        named_buffers=lambda: [
            ("d", NS(data=NS(copy_=draft_copied.append)))
        ]
    )
    backend = NS(resume=lambda tags: resumed.append(tags))
    worker = _make_worker(
        model_runner=NS(
            model=model, get_draft_model=lambda: draft,
            post_kv_cache_wake_up=lambda: resumed.append("kv"),
        ),
        _sleep_mode_backend=backend,
        _sleep_saved_buffers={"w": NS(data="saved")},
        _sleep_saved_draft_buffers={"d": NS(data="saved-d")},
    )
    wk.WorkerFL.wake_up(worker, tags=None)
    assert copied == ["saved"]
    assert draft_copied == ["saved-d"]
    assert worker._sleep_saved_buffers == {}
    assert worker._sleep_saved_draft_buffers == {}
    assert resumed[0] is None and "kv" in resumed


# ===========================================================================
# 7. worker delegation / lifecycle
# ===========================================================================
def test_worker_checkpoint_prepare_restore(monkeypatch):
    seen = []
    monkeypatch.setattr(
        wk, "checkpoint_prepare_distributed_state", lambda: seen.append("prep")
    )
    monkeypatch.setattr(
        wk, "checkpoint_restore_distributed_state", lambda: seen.append("restore")
    )
    wk.WorkerFL.checkpoint_prepare(NS())
    wk.WorkerFL.checkpoint_restore(NS())
    assert seen == ["prep", "restore"]


def test_worker_load_model_weight_transfer(monkeypatch):
    loaded = []
    engines = []
    threads = []
    monkeypatch.setattr(
        wk.WeightTransferEngineFactory,
        "create_engine",
        lambda *a, **k: engines.append(a) or "ENG",
    )
    monkeypatch.setattr(
        wk, "set_torch_threads_for_runtime", lambda: threads.append(1)
    )
    worker = NS(
        vllm_config=NS(weight_transfer_config=NS()),
        device=NS(type="npu"),
        model_runner=NS(
            load_model=lambda **k: loaded.append(k), get_model=lambda: "MODEL"
        ),
    )
    worker._maybe_get_memory_pool_context = lambda tag: nullcontext()
    worker._scoped_allocator_max_split = lambda max_split_size_mb: nullcontext()
    wk.WorkerFL.load_model(worker, load_dummy_weights=True)
    assert loaded == [{"load_dummy_weights": True}]
    assert worker.weight_transfer_engine == "ENG"
    assert engines and engines[0][-1] == "MODEL"
    assert threads == [1]

    worker2 = NS(
        vllm_config=NS(weight_transfer_config=None),
        device=NS(type="cuda"),
        model_runner=NS(load_model=lambda **k: None),
    )
    worker2._maybe_get_memory_pool_context = lambda tag: nullcontext()
    worker2._scoped_allocator_max_split = lambda max_split_size_mb: nullcontext()
    wk.WorkerFL.load_model(worker2)
    assert not hasattr(worker2, "weight_transfer_engine")


def test_worker_update_config_and_reload(monkeypatch):
    seen = []
    worker = NS(
        vllm_config=NS(),
        model_runner=NS(
            update_config=lambda o: seen.append(("update", o)),
            reload_weights=lambda *a, **k: seen.append(("reload", a, k)),
        ),
    )
    wk.WorkerFL.update_config(worker, {"x": 1})
    wk.WorkerFL.reload_weights(worker, 1, b=2)
    assert seen == [("update", {"x": 1}), ("reload", (1,), {"b": 2})]


def test_worker_kv_handshake_metadata(monkeypatch):
    monkeypatch.setattr(wk, "has_kv_transfer_group", lambda: False)
    assert wk.WorkerFL.get_kv_connector_handshake_metadata(NS()) is None

    monkeypatch.setattr(wk, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(
        wk,
        "get_kv_transfer_group",
        lambda: NS(get_handshake_metadata=lambda: None),
    )
    assert wk.WorkerFL.get_kv_connector_handshake_metadata(NS()) is None

    monkeypatch.setattr(
        wk,
        "get_kv_transfer_group",
        lambda: NS(get_handshake_metadata=lambda: "meta"),
    )
    monkeypatch.setattr(wk, "get_pp_group", lambda: NS(rank_in_group=1))
    monkeypatch.setattr(wk, "get_tp_group", lambda: NS(rank_in_group=2))
    assert wk.WorkerFL.get_kv_connector_handshake_metadata(NS()) == {
        (1, 2): "meta"
    }


def test_worker_kv_spec_layouts_and_max_len(monkeypatch):
    seen = []
    worker = _make_worker(
        vllm_config=NS(),
        model_config=NS(max_model_len=16),
        model_runner=NS(
            get_kv_cache_spec=lambda: "SPEC",
            update_max_model_len=lambda n: seen.append(n),
        ),
    )
    assert wk.WorkerFL.get_kv_cache_spec(worker) == "SPEC"
    assert wk.WorkerFL.get_supported_kv_cache_layouts(worker) == []
    _quiet(monkeypatch, wk)
    wk.WorkerFL.update_max_model_len(worker, 512)
    assert worker.model_config.max_model_len == 512
    assert seen == [512]


def test_worker_initialize_from_config(monkeypatch):
    seen = []
    monkeypatch.setattr(
        wk, "ensure_kv_transfer_initialized", lambda cfg, kv: seen.append("kv")
    )
    worker = NS(
        cache_config=NS(),
        vllm_config=NS(),
        model_config=NS(enable_return_routed_experts=True),
        model_runner=NS(
            initialize_kv_cache=lambda cfg: seen.append("init"),
            init_routed_experts_capturer=lambda: seen.append("routed"),
            _init_kv_zero_meta=lambda: seen.append("zero"),
        ),
    )
    worker._maybe_get_memory_pool_context = lambda tag: nullcontext()
    kv = NS(num_blocks=4, needs_kv_cache_zeroing=True)
    wk.WorkerFL.initialize_from_config(worker, kv)
    assert worker.cache_config.num_gpu_blocks == 4
    assert seen == ["kv", "init", "routed", "zero"]


def test_worker_reset_caches_and_get_model():
    seen = []
    worker = NS(
        model_runner=NS(
            reset_mm_cache=lambda: seen.append("mm"),
            reset_encoder_cache=lambda: seen.append("enc"),
            get_model=lambda: "MODEL",
            get_draft_model=lambda: "DRAFT",
        )
    )
    wk.WorkerFL.reset_mm_cache(worker)
    wk.WorkerFL.reset_encoder_cache(worker)
    assert wk.WorkerFL.get_model(worker) == "MODEL"
    assert wk.WorkerFL.get_draft_model(worker) == "DRAFT"
    assert seen == ["mm", "enc"]


def test_worker_draft_weight_update_gates_and_target():
    engine = NS(supports_draft_weight_update=True)
    worker = NS(
        weight_transfer_engine=engine,
        speculative_config=NS(draft_model_config=NS()),
        model_runner=NS(get_draft_model=lambda: "d"),
    )
    assert wk.WorkerFL.supports_draft_weight_updates(worker) is True
    worker.weight_transfer_engine = None
    assert wk.WorkerFL.supports_draft_weight_updates(worker) is False
    worker.weight_transfer_engine = engine
    worker.model_runner = NS(get_draft_model=lambda: None)
    assert wk.WorkerFL.supports_draft_weight_updates(worker) is False
    worker.model_runner = NS(get_draft_model=lambda: "d")
    worker.speculative_config = None
    assert wk.WorkerFL.supports_draft_weight_updates(worker) is False

    seen = []
    target = NS(
        weight_transfer_engine=NS(
            set_weight_update_target=lambda m, c: seen.append((m, c))
        ),
        get_draft_model=lambda: None,
        speculative_config=NS(draft_model_config=NS()),
    )
    with pytest.raises(RuntimeError, match="no draft model"):
        wk.WorkerFL._set_draft_weight_update_target(target)

    target.get_draft_model = lambda: "DRAFT"
    target.speculative_config = NS(draft_model_config=None)
    with pytest.raises(RuntimeError, match="no draft model"):
        wk.WorkerFL._set_draft_weight_update_target(target)

    draft_cfg = NS()
    target.speculative_config = NS(draft_model_config=draft_cfg)
    wk.WorkerFL._set_draft_weight_update_target(target)
    assert seen == [("DRAFT", draft_cfg)]


def test_worker_tasks_table_encoder_stats(monkeypatch):
    worker = NS(
        model_runner=NS(
            get_supported_tasks=lambda: ("generate",),
            get_encoder_timing_stats=lambda: {"enc": {"t": 1}},
        )
    )
    assert wk.WorkerFL.get_supported_tasks(worker) == ("generate",)
    assert wk.WorkerFL.get_encoder_timing_stats(worker) == {"enc": {"t": 1}}
    monkeypatch.setitem(
        sys.modules,
        "vllm.compilation.passes.vllm_inductor_pass",
        NS(get_match_table=lambda: {"pass": 2}),
    )
    assert wk.WorkerFL.get_compilation_match_table(worker) == {"pass": 2}


# ===========================================================================
# 8. annotate_profile / execute / sample
# ===========================================================================
def _sched_output():
    return NS(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=NS(
            req_ids=[], num_computed_tokens=[], is_context_phase=lambda r: False
        ),
        num_scheduled_tokens={"a": 3},
    )


def test_worker_annotate_profile_paths(monkeypatch):
    monkeypatch.setattr(
        wk,
        "compute_iteration_details",
        lambda so: NS(
            num_ctx_requests=1,
            num_ctx_tokens=2,
            num_generation_requests=3,
            num_generation_tokens=4,
        ),
    )
    assert isinstance(
        wk.WorkerFL.annotate_profile(NS(profiler=None, vllm_config=NS()),
                                     _sched_output()),
        nullcontext,
    )

    steps = []
    profiler = NS(
        step=lambda: steps.append(1),
        is_running=False,
    )
    worker = NS(profiler=profiler, vllm_config=NS())
    assert isinstance(
        wk.WorkerFL.annotate_profile(worker, _sched_output()), nullcontext
    )
    assert steps == [1]

    captured = {}
    profiler = NS(
        step=lambda: None,
        is_running=True,
        annotate_context_manager=lambda a: captured.setdefault("a", a),
    )
    worker = NS(
        profiler=profiler,
        vllm_config=NS(profiler_config=NS(detailed_trace_annotation=False)),
    )
    wk.WorkerFL.annotate_profile(worker, _sched_output())
    assert captured["a"] == "execute_context_1(2)_generation_3(4)"

    captured.clear()
    profiler.annotate_context_manager = lambda a: captured.setdefault("a", a)
    sched = NS(
        scheduled_new_reqs=[NS(req_id="a", num_computed_tokens=5)],
        scheduled_cached_reqs=NS(
            req_ids=["b"],
            num_computed_tokens=[4],
            is_context_phase=lambda r: False,
        ),
        num_scheduled_tokens={"a": 2, "b": 1},
    )
    worker.vllm_config.profiler_config.detailed_trace_annotation = True
    wk.WorkerFL.annotate_profile(worker, sched)
    assert captured["a"].startswith("execute_3_context_1(sq2sk7")
    assert "_generation_" in captured["a"]
    assert "sk5sqsq1sqsk5" in captured["a"]


def test_worker_sample_tokens_and_take_draft(monkeypatch):
    worker = NS(
        model_runner=NS(
            sample_tokens=lambda g: ("sample", g),
            take_draft_token_ids=lambda: "draft",
            uniform_decode_query_len=4,
            _dummy_run=lambda n, **k: ("dummy", n, k),
        )
    )
    assert wk.WorkerFL.sample_tokens(worker, "grammar") == ("sample", "grammar")
    assert wk.WorkerFL.take_draft_token_ids(worker) == "draft"
    wk.WorkerFL.execute_dummy_batch(worker)
    # _dummy_run signature check is implicit; delegation must not raise


def test_worker_execute_model_fast_and_pp_paths(monkeypatch):
    outputs = sys.modules["vllm.v1.outputs"]
    monkeypatch.setattr(
        wk,
        "get_pp_group",
        lambda: NS(is_first_rank=True, is_last_rank=True, rank_in_group=0),
    )
    monkeypatch.setattr(wk, "get_tp_group", lambda: NS(rank_in_group=0))
    out = outputs.ModelRunnerOutput()
    waited = []
    worker = NS(
        _pp_send_work=[NS(wait=lambda: waited.append(1))],
        model_runner=NS(execute_model=lambda s, i: out),
        use_v2_model_runner=False,
    )
    worker.annotate_profile = lambda s: nullcontext()
    result = wk.WorkerFL.execute_model(
        worker, NS(total_num_scheduled_tokens=1)
    )
    assert result is out
    assert waited == [1]
    assert worker._pp_send_work == []

    pooled = []
    worker2 = NS(
        _pp_send_work=[],
        model_runner=NS(
            execute_model=lambda s, i: None,
            is_pooling_model=True,
            pool=lambda: pooled.append(1),
        ),
        use_v2_model_runner=True,
    )
    worker2.annotate_profile = lambda s: nullcontext()
    assert wk.WorkerFL.execute_model(
        worker2, NS(total_num_scheduled_tokens=0)
    ) is None
    assert pooled == [1]


def test_worker_execute_model_pp_intermediate(monkeypatch):
    handles = [NS(wait=lambda: None)]
    sent = []

    class _PP:
        is_first_rank = False
        is_last_rank = False
        rank_in_group = 1

        def irecv_tensor_dict(self, **k):
            return {"t": 1}, handles, [lambda: None]

        def isend_tensor_dict(self, tensors, **k):
            sent.append(tensors)
            return ["work"]

    monkeypatch.setattr(wk, "get_pp_group", lambda: _PP())
    monkeypatch.setattr(wk, "get_tp_group", lambda: NS(rank_in_group=0))
    worker = NS(
        _pp_send_work=[],
        model_runner=NS(execute_model=lambda s, i: wk_module_intermediate()),
        use_v2_model_runner=False,
        vllm_config=NS(parallel_config=NS(distributed_executor_backend="mp")),
    )
    worker.annotate_profile = lambda s: nullcontext()
    assert wk.WorkerFL.execute_model(
        worker, NS(total_num_scheduled_tokens=2)
    ) is None
    assert sent == [{"t": 2}]
    assert worker._pp_send_work == ["work"]

    pooled = []
    worker2 = NS(
        _pp_send_work=[],
        model_runner=NS(
            execute_model=lambda s, i: wk_module_intermediate(),
            is_pooling_model=True,
            pool=lambda: pooled.append(1),
        ),
        use_v2_model_runner=True,
        vllm_config=NS(parallel_config=NS(distributed_executor_backend="mp")),
    )
    worker2.annotate_profile = lambda s: nullcontext()
    assert wk.WorkerFL.execute_model(
        worker2, NS(total_num_scheduled_tokens=2)
    ) is None
    assert pooled == [1]


def wk_module_intermediate():
    return sys.modules["vllm.sequence"].IntermediateTensors({"t": 2})


# ===========================================================================
# 9. profile / lora / save / weight transfer / shutdown / misc
# ===========================================================================
def test_worker_profile_dispatch_and_errors(monkeypatch):
    monkeypatch.setattr(wk, "logger", _logger())
    worker = NS(profiler_config=None, profiler=None)
    with pytest.raises(RuntimeError, match="Profiling is not enabled"):
        wk.WorkerFL.profile(worker)

    created = []
    for name in ("TorchProfilerWrapper", "CudaProfilerWrapper",
                 "ProtonProfilerWrapper"):
        monkeypatch.setattr(
            wk,
            name,
            lambda *a, _n=name, **k: created.append(_n)
            or NS(start=lambda: created.append("start"),
                  stop=lambda: created.append("stop"),
                  shutdown=lambda: None),
        )
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.utils",
        NS(get_worker_rank_suffix=lambda global_rank=None: f"r{global_rank}"),
    )
    for kind, expected in (
        ("torch", "TorchProfilerWrapper"),
        ("cuda", "CudaProfilerWrapper"),
        ("proton", "ProtonProfilerWrapper"),
    ):
        created.clear()
        worker = NS(
            profiler_config=NS(profiler=kind),
            profiler=None,
            rank=2,
            local_rank=1,
        )
        wk.WorkerFL.profile(worker, is_start=True, profile_prefix="p")
        assert expected in created
        wk.WorkerFL.profile(worker, is_start=False)
        if kind == "proton":
            assert worker.profiler is None
        else:
            assert worker.profiler is not None

    worker = NS(profiler_config=NS(profiler="torch"), profiler=None)
    wk.WorkerFL.profile(worker, is_start=False)  # warns, no-op

    worker = NS(
        profiler_config=NS(profiler="bogus"), profiler=None, rank=0,
        local_rank=0,
    )
    with pytest.raises(ValueError, match="Invalid profiler"):
        wk.WorkerFL.profile(worker, is_start=True)


def test_worker_lora_and_health_and_tensorizer(monkeypatch):
    seen = []
    worker = NS(
        model_runner=NS(
            add_lora=lambda r: seen.append(("add", r)) or True,
            remove_lora=lambda i: seen.append(("remove", i)) or True,
            list_loras=lambda: {1},
            pin_lora=lambda i: seen.append(("pin", i)) or True,
            model="MODEL",
        ),
        model_config=NS(),
    )
    worker.get_model = lambda: "MODEL"
    assert wk.WorkerFL.add_lora(worker, "req") is True
    assert wk.WorkerFL.remove_lora(worker, 3) is True
    assert wk.WorkerFL.list_loras(worker) == {1}
    assert wk.WorkerFL.pin_lora(worker, 4) is True
    assert wk.WorkerFL.check_health(worker) is None

    loader = sys.modules["vllm.model_executor.model_loader"]
    monkeypatch.setattr(
        loader.ShardedStateLoader,
        "save_model",
        lambda *a, **k: seen.append(("shard", a, k)),
    )
    wk.WorkerFL.save_sharded_state(worker, "/tmp/ckpt", pattern="p", max_size=8)
    assert ("shard", ("MODEL", "/tmp/ckpt"),
            {"pattern": "p", "max_size": 8}) in seen

    monkeypatch.setattr(
        loader.TensorizerLoader,
        "save_model",
        lambda *a, **k: seen.append(("tensor", a, k)),
    )
    wk.WorkerFL.save_tensorized_model(worker, "cfg")
    assert ("tensor", ("MODEL",),
            {"tensorizer_config": "cfg", "model_config": worker.model_config}
            ) in seen
    assert seen[:4] == [
        ("add", "req"), ("remove", 3), ("pin", 4), ("shard", ("MODEL",
        "/tmp/ckpt"), {"pattern": "p", "max_size": 8})
    ][:4]


def test_worker_weight_transfer_lifecycle(monkeypatch):
    seen = []
    worker = _make_worker(
        weight_transfer_engine=None, vllm_config=NS(),
        _weight_update_active=False, _weight_update_is_draft=False,
    )
    with pytest.raises(RuntimeError, match="Weight transfer not configured"):
        wk.WorkerFL.init_weight_transfer_engine(worker, {})

    engine = NS(
        supports_draft_weight_update=False,
        parse_init_info=lambda d: ("typed", d),
        init_transfer_engine=lambda v: seen.append(("init", v)),
        start_weight_update=lambda: seen.append("start"),
        update_weights=lambda d: seen.append(("update", d)),
        finish_weight_update=lambda: seen.append("finish"),
        reset_weight_update_target=lambda: seen.append("reset"),
        set_weight_update_target=lambda m, c: seen.append(("target", m)),
    )
    worker.weight_transfer_engine = engine
    worker._weight_update_active = False
    worker._weight_update_is_draft = False
    worker.model_runner = NS(
        reset_lora_state=lambda: seen.append("lora"),
        get_draft_model=lambda: "DRAFT",
    )
    worker.speculative_config = NS(draft_model_config=NS())

    wk.WorkerFL.init_weight_transfer_engine(worker, {"k": 1})
    assert ("init", ("typed", {"k": 1})) in seen

    with pytest.raises(RuntimeError, match="start_weight_update must be called"):
        wk.WorkerFL.update_weights(worker, {})
    with pytest.raises(RuntimeError, match="without a matching"):
        wk.WorkerFL.finish_weight_update(worker)

    wk.WorkerFL.start_weight_update(worker)
    assert "start" in seen
    with pytest.raises(RuntimeError, match="already active"):
        wk.WorkerFL.start_weight_update(worker)
    wk.WorkerFL.update_weights(worker, {"chunk": 1})
    assert ("update", {"chunk": 1}) in seen
    wk.WorkerFL.finish_weight_update(worker)
    assert "finish" in seen and "lora" in seen
    assert worker._weight_update_active is False

    with pytest.raises(RuntimeError, match="does not support draft"):
        wk.WorkerFL.start_draft_weight_update(worker)

    engine.supports_draft_weight_update = True
    seen.clear()
    wk.WorkerFL.start_draft_weight_update(worker)
    assert ("target", "DRAFT") in seen
    assert worker._weight_update_is_draft is True
    wk.WorkerFL.finish_weight_update(worker)
    assert "lora" not in seen

    def _boom_update(_info):
        raise ValueError("bad chunk")

    engine.update_weights = _boom_update
    wk.WorkerFL.start_weight_update(worker)
    with pytest.raises(ValueError, match="bad chunk"):
        wk.WorkerFL.update_weights(worker, {})
    assert worker._weight_update_active is False
    assert "reset" in seen

    def _boom_start():
        raise RuntimeError("start failed")

    engine.start_weight_update = _boom_start
    seen.clear()
    with pytest.raises(RuntimeError, match="start failed"):
        wk.WorkerFL.start_weight_update(worker)
    assert "reset" in seen
    assert worker._weight_update_active is False


def test_worker_shutdown_releases(monkeypatch):
    seen = []
    monkeypatch.setattr(wk, "gc", NS(unfreeze=lambda: seen.append("gc")))
    monkeypatch.setattr(wk, "ensure_kv_transfer_shutdown",
                        lambda: seen.append("kv"))
    monkeypatch.setattr(wk, "ensure_ec_transfer_shutdown",
                        lambda: seen.append("ec"))
    cumem = sys.modules["vllm.device_allocator.cumem"]
    instance = NS(release_pools=lambda: seen.append("pools"))
    monkeypatch.setattr(cumem.CuMemAllocator, "instance", instance)
    monkeypatch.setattr(wk, "current_platform",
                        NS(is_cuda_alike=lambda: True))
    worker = NS(
        profiler=NS(shutdown=lambda: seen.append("profiler")),
        weight_transfer_engine=NS(shutdown=lambda: seen.append("engine")),
        model_runner=NS(shutdown=lambda: seen.append("runner")),
    )
    wk.WorkerFL.shutdown(worker)
    assert seen == ["gc", "kv", "ec", "profiler", "engine", "runner", "pools"]

    seen.clear()
    monkeypatch.setattr(cumem.CuMemAllocator, "instance", None)
    monkeypatch.setattr(wk, "current_platform",
                        NS(is_cuda_alike=lambda: False))
    wk.WorkerFL.shutdown(NS(profiler=None, weight_transfer_engine=None,
                            model_runner=None))
    assert seen == ["gc", "kv", "ec"]


def test_worker_elastic_ep_execute():
    seen = []
    worker = NS(
        elastic_ep_executor=NS(
            execute=lambda m, *a, **k: seen.append((m, a, k)) or "done"
        )
    )
    assert wk.WorkerFL.elastic_ep_execute(worker, "step", 1, mode="x") == "done"
    assert seen == [("step", (1,), {"mode": "x"})]


def test_worker_compile_or_warm_up_v1_txda(monkeypatch):
    seen = []

    class _Range:
        end = 16

        def __contains__(self, value):
            return False

    cc = NS(
        mode=_CompilationMode.VLLM_COMPILE,
        backend="inductor",
        compile_sizes=[8],
        cudagraph_mode=_CUDAGraphMode.PIECEWISE,
        cudagraph_capture_sizes=[8],
        get_compile_ranges=lambda: [_Range()],
        compilation_time=1.5,
        encoder_compilation_time=0.25,
    )
    runner = NS(
        lora_config="lora",
        is_pooling_model=False,
        maybe_remove_all_loras=lambda cfg: seen.append(("rm_lora", cfg)),
        _dummy_run=lambda *a, **k: seen.append(("dummy", a, k)) or ("h", "last"),
        _dummy_sampler_run=lambda hidden_states: seen.append(
            ("sampler", hidden_states)
        ),
        _dummy_pooler_run=lambda hidden: seen.append(("pooler", hidden)),
    )
    worker = NS(
        vllm_config=NS(
            compilation_config=cc,
            kernel_config=NS(enable_jit_warmup=False),
        ),
        compilation_config=cc,
        model_config=NS(enforce_eager=True, seed=5),
        cache_config=NS(kv_cache_memory_bytes=123),
        use_v2_model_runner=False,
        device=NS(type="npu"),
        observability_config=NS(
            jit_monitor_mode="once", jit_monitor_verbose=False
        ),
        scheduler_config=NS(max_num_seqs=2, max_num_batched_tokens=8),
        model_runner=runner,
    )
    monkeypatch.setattr(wk, "current_platform",
                        NS(device_type="txda", vendor_name="tsingmicro"))
    monkeypatch.setattr(wk, "kernel_warmup",
                        lambda w: seen.append("kernel-warmup"))
    monkeypatch.setattr(wk, "set_random_seed",
                        lambda s: seen.append(("seed", s)))
    monkeypatch.setattr(wk, "freeze_gc_heap",
                        lambda: seen.append("freeze"))
    monkeypatch.setattr(wk, "maybe_attach_gc_debug_callback",
                        lambda: seen.append("gc-cb"))
    monkeypatch.setattr(wk, "enable_gpu_sync_check",
                        lambda: seen.append("sync"))
    monkeypatch.setattr(wk, "set_torch_threads_for_runtime",
                        lambda: seen.append("threads"))
    trigger = sys.modules["vllm.compilation.compiler_interface"]
    monkeypatch.setattr(
        trigger, "trigger_inductor_lazy_init",
        lambda dev: seen.append(("trigger", dev)),
    )
    monkeypatch.setattr(
        wk,
        "get_pp_group",
        lambda: NS(is_last_rank=True, is_first_rank=True, rank_in_group=0),
    )
    _quiet(monkeypatch, wk)
    times = wk.WorkerFL.compile_or_warm_up_model(worker)
    assert times.language_model == 1.5
    assert times.encoder == 0.25
    assert ("rm_lora", "lora") in seen
    assert ("sampler", "last") in seen
    assert ("seed", 5) in seen
    assert ("dummy", (16,), {"skip_eplb": True, "remove_lora": False}) in seen
    assert ("trigger", worker.device) in seen
    assert "kernel-warmup" not in seen, "txda must skip generic kernel_warmup"
    assert "freeze" in seen and "gc-cb" in seen and "sync" in seen
    assert "threads" not in seen, "npu must not reset torch threads"

    # v1 pooling branch warms the pooler instead of the sampler
    runner.is_pooling_model = True
    wk.WorkerFL.compile_or_warm_up_model(worker)
    assert ("pooler", "h") in seen


def test_worker_compile_or_warm_up_v2_startup_plan(monkeypatch):
    seen = []
    cc = NS(
        mode=_CompilationMode.NONE,
        backend="eager",
        compile_sizes=None,
        cudagraph_mode=_CUDAGraphMode.NONE,
        cudagraph_capture_sizes=None,
        get_compile_ranges=lambda: [],
        compilation_time=0.0,
        encoder_compilation_time=0.0,
    )
    runner = NS(
        lora_config=None,
        maybe_remove_all_loras=lambda cfg: None,
        capture_model=lambda: 100,
    )
    worker = NS(
        vllm_config=NS(
            compilation_config=cc,
            kernel_config=NS(enable_jit_warmup=False),
        ),
        compilation_config=cc,
        model_config=NS(enforce_eager=False, seed=1),
        cache_config=NS(kv_cache_memory_bytes=None,
                        gpu_memory_utilization=0.9),
        use_v2_model_runner=True,
        device=NS(type="cuda"),
        observability_config=NS(
            jit_monitor_mode="once", jit_monitor_verbose=True
        ),
        scheduler_config=NS(max_num_seqs=1, max_num_batched_tokens=1),
        model_runner=runner,
        cudagraph_memory_estimate=80,
        peak_activation_memory=5,
        total_consumed=1,
        init_snapshot=NS(free_memory=1 << 30, total_memory=2 << 30),
        requested_memory=1 << 29,
        available_kv_cache_memory_bytes=10,
        execute_model=lambda: None,
        sample_tokens=lambda g: None,
    )
    monkeypatch.setattr(wk, "current_platform",
                        NS(device_type="cuda", vendor_name="nvidia"))

    def _raise_import(_w):
        raise ImportError("torchvision missing")

    monkeypatch.setattr(wk, "kernel_warmup", _raise_import)
    monkeypatch.setattr(
        wk, "warmup_kernels", lambda *a: seen.append(("warmup", len(a)))
    )
    monkeypatch.setattr(
        wk,
        "maybe_save_startup_plan",
        lambda w, limit: seen.append(("plan", w is worker, limit)),
    )
    monkeypatch.setattr(
        wk, "activate", lambda **k: seen.append(("jit", k)), raising=False
    )
    jit_mod = types.ModuleType("vllm.utils.jit_monitor")
    jit_mod.activate = lambda **k: seen.append(("jit", k))
    monkeypatch.setitem(sys.modules, "vllm.utils.jit_monitor", jit_mod)
    monkeypatch.setattr(wk, "freeze_gc_heap", lambda: None)
    monkeypatch.setattr(wk, "maybe_attach_gc_debug_callback", lambda: None)
    monkeypatch.setattr(wk, "enable_gpu_sync_check", lambda: None)
    monkeypatch.setattr(
        wk, "set_torch_threads_for_runtime", lambda: seen.append("threads")
    )
    seen_log = _quiet(monkeypatch, wk)
    wk.WorkerFL.compile_or_warm_up_model(worker)
    assert ("warmup", 3) in seen
    plans = [s for s in seen if isinstance(s, tuple) and s[0] == "plan"]
    assert plans and plans[0][1] is True
    assert any(s[0] == "jit" for s in seen if isinstance(s, tuple))
    assert "threads" in seen
    assert seen_log["warning"], "ImportError in kernel_warmup must be logged"


def test_init_worker_distributed_environment(monkeypatch):
    seen = []
    pc = NS(
        disable_custom_all_reduce=True,
        distributed_timeout_seconds=30,
        world_size=2,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    cfg = NS(parallel_config=pc, kernel_config=NS(moe_backend="auto"))
    bi = sys.modules["vllm.model_executor.determinism.batch_invariant"]
    monkeypatch.setattr(bi, "init_batch_invariance",
                        lambda: seen.append("bi"))
    monkeypatch.setattr(wk, "override_envs_for_eplb",
                        lambda c, moe_backend=None: seen.append(("eplb",
                                                                 moe_backend)))
    monkeypatch.setattr(wk, "set_custom_all_reduce",
                        lambda flag: seen.append(("car", flag)))
    monkeypatch.setattr(
        wk, "init_distributed_environment",
        lambda *a: seen.append(("init", a)),
    )
    monkeypatch.setattr(
        wk, "ensure_model_parallel_initialized",
        lambda *a: seen.append(("mp", a)),
    )
    monkeypatch.setattr(wk, "ensure_ec_transfer_initialized",
                        lambda c: seen.append("ec"))
    monkeypatch.setattr(
        wk, "current_platform", NS(device_type="npu", vendor_name="ascend")
    )
    wk.init_worker_distributed_environment(
        cfg, 1, distributed_init_method="tcp://x", local_rank=0,
        backend="flagcx",
    )
    assert "bi" in seen
    assert ("eplb", "auto") in seen
    assert ("car", False) in seen
    init_args = [s[1] for s in seen if isinstance(s, tuple) and s[0] == "init"]
    assert init_args and init_args[0][4] == "hccl"
    assert init_args[0][5] == wk.timedelta(seconds=30)
    assert "ec" in seen

    seen.clear()
    pc.distributed_timeout_seconds = None
    monkeypatch.setattr(
        wk, "current_platform", NS(device_type="cuda", vendor_name="nvidia")
    )
    wk.init_worker_distributed_environment(cfg, 0, backend="nccl")
    init_args = [s[1] for s in seen if isinstance(s, tuple) and s[0] == "init"]
    assert init_args[0][2] == "env://"
    assert init_args[0][4] == "nccl"
    assert init_args[0][5] is None
