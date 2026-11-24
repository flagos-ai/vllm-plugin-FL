"""Coverage tests for vllm_fl worker / scheduler / runtime (Enflame S60).

Self-contained: vLLM configs/executors/distributed + vllm_fl siblings +
xgrammar/flag_gems are replaced by in-memory stubs, so no GPU, distributed
process, model weights or network are touched. Set ENFLAME_SRC to point at a
different checkout of the plugin.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.util
import logging
import os
import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest


class _FakeDevice:
    def __init__(self, spec):
        self._spec = str(spec)
        self.type = self._spec.split(":")[0]

    def __repr__(self):
        return f"device(type='{self.type}')"

    def __eq__(self, other):
        return isinstance(other, _FakeDevice) and other._spec == self._spec

    def __hash__(self):
        return hash(self._spec)


def _build_fake_torch():
    module = types.ModuleType("torch")
    nn_mod = types.ModuleType("torch.nn")

    class Module:
        pass

    nn_mod.Module = Module
    module.nn = nn_mod
    module.Tensor = type("Tensor", (), {})
    module.device = _FakeDevice
    module.dtype = type("dtype", (), {})
    module._C = types.SimpleNamespace()
    module.set_float32_matmul_precision = lambda *a, **k: None
    module.accelerator = types.SimpleNamespace()
    sys.modules["torch"] = module
    sys.modules["torch.nn"] = nn_mod
    return module


try:
    import torch
except ImportError:
    torch = _build_fake_torch()

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SOURCE_ROOT = os.environ.get("ENFLAME_SRC") or os.path.abspath(
    os.path.join(_THIS_DIR, "..", "enflame-src-030b")
)
if not os.path.isdir(os.path.join(_SOURCE_ROOT, "vllm_fl")):
    pytest.skip(
        f"enflame source checkout not found at {_SOURCE_ROOT}",
        allow_module_level=True,
    )


# --------------------------------------------------------------------------
# Stub classes used by the fake vLLM tree
# --------------------------------------------------------------------------
class CUDAGraphMode:
    NONE = "none"
    PIECEWISE = "piecewise"
    FULL_AND_PIECEWISE = "full_and_piecewise"


class CompilationMode:
    NONE = "none"
    VLLM_COMPILE = "vllm_compile"
    DYNAMO_TRACE_ONCE = "dynamo_trace_once"


class IntermediateTensors:
    def __init__(self, tensors=None):
        self.tensors = dict(tensors or {})


class ModelRunnerOutput:
    pass


class AsyncModelRunnerOutput:
    pass


class DraftTokenIds:
    pass


class GrammarOutput:
    pass


class SchedulerOutput:
    pass


class KVCacheConfig:
    def __init__(self, num_blocks=1, needs_kv_cache_zeroing=False):
        self.num_blocks = num_blocks
        self.needs_kv_cache_zeroing = needs_kv_cache_zeroing


class KVCacheSpec:
    pass


class KVConnectorHandshakeMetadata:
    pass


class Handle:
    def __init__(self, tag="h"):
        self.tag = tag
        self.waited = False

    def wait(self):
        self.waited = True


class CompilationTimes:
    def __init__(self, language_model=0.0, encoder=0.0):
        self.language_model = language_model
        self.encoder = encoder


class WorkerBase:
    def __init__(self, vllm_config=None, local_rank=0, rank=0,
                 distributed_init_method="env://", is_driver_worker=False,
                 **kwargs):
        self.vllm_config = vllm_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        self.device = None
        self.model_runner = None

    @property
    def model_config(self):
        return self.vllm_config.model_config

    @property
    def parallel_config(self):
        return self.vllm_config.parallel_config

    @property
    def cache_config(self):
        return self.vllm_config.cache_config

    @property
    def scheduler_config(self):
        return self.vllm_config.scheduler_config

    @property
    def speculative_config(self):
        return self.vllm_config.speculative_config

    @property
    def device_config(self):
        return self.vllm_config.device_config

    @property
    def compilation_config(self):
        return self.vllm_config.compilation_config

    @property
    def observability_config(self):
        return self.vllm_config.observability_config

    @property
    def load_config(self):
        return self.vllm_config.load_config

    def get_supported_kv_cache_layouts(self):
        return []


def _native_sample(self, grammar_output):
    return getattr(self, "_native_sample_result", None)


def _sample_tokens(self, grammar_output):
    return _native_sample(self, grammar_output)


_sample_tokens.__wrapped__ = _native_sample


def _native_exec(self, scheduler_output):
    return getattr(self, "_native_exec_result", None)


def _execute_model(self, scheduler_output):
    return _native_exec(self, scheduler_output)


_execute_model.__wrapped__ = _native_exec


class GPUWorker(WorkerBase):
    sample_tokens = _sample_tokens
    execute_model = _execute_model

    def init_device(self):
        self._base_init_device_calls = getattr(self, "_base_init_device_calls", 0) + 1

    def determine_available_memory(self):
        return getattr(self, "_base_available_memory", 4096)

    def _init_workspace_and_model_runner(self):
        self._base_workspace_init_calls = (
            getattr(self, "_base_workspace_init_calls", 0) + 1
        )


class AsyncScheduler:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class MemorySnapshot:
    def __init__(self, device=None):
        self.device = device
        self.free_memory = 0
        self.total_memory = 0


class ElasticEPScalingExecutor:
    def __init__(self, worker):
        self.worker = worker
        self.calls = []

    def execute(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return ("elastic", method)


class WorkerSentinel:
    def __init__(self, worker=None):
        self.worker = worker
        self.commands = []

    def handle_command(self, cmd):
        self.commands.append(cmd)
        return ("handled", cmd)


class WeightTransferEngine:
    supports_draft_weight_update = False

    def parse_init_info(self, info):
        return ("parsed", info)

    def init_transfer_engine(self, info):
        self.engine_info = info

    def start_weight_update(self):
        self.started = True

    def update_weights(self, info):
        self.updated = info

    def finish_weight_update(self):
        self.finished = True

    def reset_weight_update_target(self):
        self.reset_calls = getattr(self, "reset_calls", 0) + 1

    def set_weight_update_target(self, model, cfg):
        self.target = (model, cfg)

    def shutdown(self):
        self.shut = True


class WeightTransferEngineFactory:
    @staticmethod
    def create_engine(*args, **kwargs):
        return WeightTransferEngine()


class _ProfilerWrapperBase:
    is_running = False

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def shutdown(self):
        self.shutdown_called = True

    def step(self):
        pass


class TorchProfilerWrapper(_ProfilerWrapperBase):
    pass


class CudaProfilerWrapper(_ProfilerWrapperBase):
    pass


class ProtonProfilerWrapper(_ProfilerWrapperBase):
    pass


class TensorizerLoader:
    @staticmethod
    def save_model(*args, **kwargs):
        return None


class ShardedStateLoader:
    @staticmethod
    def save_model(*args, **kwargs):
        return None


class FakePlatform:
    device_type = "gcu"
    device_name = "gcu"
    vendor_name = "enflame"
    dist_backend = "flagcx"
    torch_device_fn = None

    def is_cuda_alike(self):
        return False

    def is_cuda(self):
        return False

    def is_xpu(self):
        return False

    def is_cpu(self):
        return False

    def is_rocm(self):
        return False

    def is_sleep_mode_available(self):
        return False

    def logical_device_id_to_visible_device_id(self, device_id):
        return device_id

    def set_device(self, device):
        self._set_device = device

    def check_if_supports_dtype(self, dtype):
        self._checked_dtype = dtype

    def device_count(self):
        return 8


_PLATFORM = FakePlatform()


def _identity_instrument(*args, **kwargs):
    def decorator(fn):
        return fn

    return decorator


def _with_gpu_sync_check(fn):
    return fn


def _managed_inference_mode():
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorator


@contextlib.contextmanager
def _set_current_vllm_config(cfg):
    yield


def _format_gib(value):
    return round(float(value) / (1 << 30), 2)


def _request_memory(snapshot, cache_config, *args, **kwargs):
    return 0


def _reserve_mm_ipc_gpu_memory(bytes_, *args, **kwargs):
    return bytes_


@contextlib.contextmanager
def _memory_profiling(snapshot, weights_memory=0):
    yield SimpleNamespace(
        total_consumed=0,
        transient_peak_headroom=0,
        non_kv_cache_memory=0,
        after_profile=SimpleNamespace(
            free_memory=getattr(snapshot, "free_memory", 0)
        ),
    )


def _compute_iteration_details(scheduler_output):
    return SimpleNamespace(
        num_ctx_requests=1,
        num_ctx_tokens=1,
        num_generation_requests=0,
        num_generation_tokens=0,
    )


def _use_flaggems_default():
    return False


def _default_whitelist_blacklist():
    return (None, None)


def _patch_mm_encoder_attention_default():
    return None


def _register_oot_ops_default(*args, **kwargs):
    return None


def _patch_qsa_cache_hint_default(*args, **kwargs):
    return False


# --------------------------------------------------------------------------
# Fake module tree installation
# --------------------------------------------------------------------------
def _ensure_pkg(name):
    if name in sys.modules:
        return sys.modules[name]
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            mod = types.ModuleType(sub)
            mod.__path__ = []
            sys.modules[sub] = mod
    return sys.modules[name]


def _m(name, **attrs):
    _ensure_pkg(name)
    mod = sys.modules[name]
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _real_pkg(name, directory):
    mod = types.ModuleType(name)
    mod.__path__ = [directory]
    mod.__package__ = name
    sys.modules[name] = mod
    return mod


class _FlagGemsStub(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.only_enable = mock.Mock()
        self.enable = mock.Mock()
        self._FULL_CONFIG = [("add",), ("mul",), ("pow.Scalar",)]


_INSTALLED = False


def _fake_logger(name):
    log = logging.getLogger(name)
    if not hasattr(log, "info_once"):
        log.info_once = log.info
        log.warning_once = log.warning
    return log


def _install_fake_vllm():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    _m("vllm", __path__=[], _covfl_stub=True)
    _m("vllm.logger", init_logger=_fake_logger)
    _m("vllm.envs", VLLM_FLOAT32_MATMUL_PRECISION="highest",
       VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=False)
    _m("vllm.config", CUDAGraphMode=CUDAGraphMode, VllmConfig=object,
       set_current_vllm_config=_set_current_vllm_config)
    _m("vllm.config.compilation", CompilationMode=CompilationMode)
    _m("vllm.device_allocator",
       get_mem_allocator_instance=mock.Mock(return_value=None))
    _m("vllm.device_allocator.cumem",
       CuMemAllocator=SimpleNamespace(instance=None))
    _m("vllm.device_allocator.sleep_mode_backend",
       SleepModeBackendFactory=SimpleNamespace(
           create_backend=mock.Mock(return_value=None)))
    _m("vllm.distributed", ensure_model_parallel_initialized=mock.Mock(),
       init_distributed_environment=mock.Mock(),
       set_custom_all_reduce=mock.Mock())
    _m("vllm.distributed.ec_transfer",
       ensure_ec_transfer_initialized=mock.Mock(),
       ensure_ec_transfer_shutdown=mock.Mock())
    _m("vllm.distributed.eplb.eplb_utils", override_envs_for_eplb=mock.Mock())
    _m("vllm.distributed.kv_transfer",
       ensure_kv_transfer_initialized=mock.Mock(),
       ensure_kv_transfer_shutdown=mock.Mock(),
       get_kv_transfer_group=mock.Mock(return_value=None),
       has_kv_transfer_group=mock.Mock(return_value=False))
    _m("vllm.distributed.kv_transfer.kv_connector.v1.base",
       KVConnectorHandshakeMetadata=KVConnectorHandshakeMetadata)
    _m("vllm.distributed.parallel_state", Handle=Handle,
       checkpoint_prepare_distributed_state=mock.Mock(),
       checkpoint_restore_distributed_state=mock.Mock(),
       get_pp_group=mock.Mock(), get_tp_group=mock.Mock(),
       prepare_communication_buffer_for_model=mock.Mock())
    _m("vllm.distributed.weight_transfer", WeightTransferEngine=WeightTransferEngine,
       WeightTransferEngineFactory=WeightTransferEngineFactory)
    _m("vllm.distributed.utils", get_worker_rank_suffix=lambda **kw: "rank0")
    _m("vllm.distributed.elastic_ep.elastic_execute",
       ElasticEPScalingExecutor=ElasticEPScalingExecutor)
    _m("vllm.lora.request", LoRARequest=object)
    _m("vllm.model_executor.model_loader", TensorizerLoader=TensorizerLoader,
       ShardedStateLoader=ShardedStateLoader,
       get_model_loader=mock.Mock(return_value=SimpleNamespace(
           get_external_weight_memory=mock.Mock(return_value=0))))
    _m("vllm.model_executor.warmup.kernel_warmup", kernel_warmup=mock.Mock())
    _m("vllm.model_executor.determinism.batch_invariant",
       init_batch_invariance=mock.Mock())
    _m("vllm.multimodal.gpu_ipc_memory",
       reserve_mm_ipc_gpu_memory=_reserve_mm_ipc_gpu_memory)
    _m("vllm.platforms", current_platform=_PLATFORM)
    _m("vllm.platforms.interface", set_assigned_physical_gpu_ids=mock.Mock())
    _m("vllm.profiler.wrapper", CudaProfilerWrapper=CudaProfilerWrapper,
       ProtonProfilerWrapper=ProtonProfilerWrapper,
       TorchProfilerWrapper=TorchProfilerWrapper)
    _m("vllm.sequence", IntermediateTensors=IntermediateTensors)
    _m("vllm.tasks", SupportedTask=str)
    _m("vllm.tracing", instrument=_identity_instrument)
    _m("vllm.utils.gc_utils", freeze_gc_heap=mock.Mock(),
       maybe_attach_gc_debug_callback=mock.Mock())
    _m("vllm.utils.gpu_sync_debug", enable_gpu_sync_check=mock.Mock(),
       with_gpu_sync_check=_with_gpu_sync_check)
    _m("vllm.utils.mem_constants", GiB_bytes=1 << 30)
    _m("vllm.utils.mem_utils", MemorySnapshot=MemorySnapshot,
       format_gib=_format_gib, memory_profiling=_memory_profiling)
    _m("vllm.utils.torch_utils", set_random_seed=mock.Mock(),
       set_torch_threads_for_runtime=mock.Mock())
    _m("vllm.utils.flashinfer", warn_if_flashinfer_kernels_missing=mock.Mock())
    _m("vllm.utils.jit_monitor", activate=mock.Mock())
    _m("vllm.compilation.compiler_interface",
       trigger_inductor_lazy_init=mock.Mock())
    _m("vllm.compilation.passes.vllm_inductor_pass",
       get_match_table=lambda: {"table": 1})
    _m("vllm.v1.core.sched.output", GrammarOutput=GrammarOutput,
       SchedulerOutput=SchedulerOutput)
    _m("vllm.v1.core.sched.async_scheduler", AsyncScheduler=AsyncScheduler)
    _m("vllm.v1.kv_cache_interface", KVCacheConfig=KVCacheConfig,
       KVCacheSpec=KVCacheSpec)
    _m("vllm.v1.outputs", AsyncModelRunnerOutput=AsyncModelRunnerOutput,
       DraftTokenIds=DraftTokenIds, ModelRunnerOutput=ModelRunnerOutput)
    _m("vllm.v1.utils", compute_iteration_details=_compute_iteration_details,
       report_usage_stats=mock.Mock())
    _m("vllm.v1.worker.gpu.warmup", warmup_kernels=mock.Mock())
    _m("vllm.v1.worker.sentinel.gpu_worker_sentinel",
       WorkerSentinel=WorkerSentinel)
    _m("vllm.v1.worker.startup_plan",
       maybe_apply_startup_plan=mock.Mock(),
       maybe_save_startup_plan=mock.Mock())
    _m("vllm.v1.worker.utils", request_memory=_request_memory)
    _m("vllm.v1.worker.worker_base", CompilationTimes=CompilationTimes,
       WorkerBase=WorkerBase)
    _m("vllm.v1.worker.workspace", init_workspace_manager=mock.Mock())
    _m("vllm.v1.worker.gpu_worker", Worker=GPUWorker,
       init_worker_distributed_environment=mock.Mock())
    _m("vllm.v1.worker.gpu.cudagraph_utils",
       has_compiled_submodule=mock.Mock(return_value=False))
    _m("vllm.v1.worker.gpu.model_runner", GPUModelRunner=GPUWorker)


def _install_fake_vllm_fl():
    base = os.path.join(_SOURCE_ROOT, "vllm_fl")
    _real_pkg("vllm_fl", base)
    _real_pkg("vllm_fl.worker", os.path.join(base, "worker"))
    _real_pkg("vllm_fl.dispatch", os.path.join(base, "dispatch"))
    _real_pkg("vllm_fl.dispatch.backends",
              os.path.join(base, "dispatch", "backends"))
    _real_pkg("vllm_fl.dispatch.backends.vendor",
              os.path.join(base, "dispatch", "backends", "vendor"))
    _real_pkg("vllm_fl.dispatch.backends.vendor.gcu",
              os.path.join(base, "dispatch", "backends", "vendor", "gcu"))
    _real_pkg("vllm_fl.dispatch.backends.vendor.gcu.worker",
              os.path.join(base, "dispatch", "backends", "vendor", "gcu",
                           "worker"))

    _m("vllm_fl.envs", USE_FLAGGEMS=False,
       FLAGGEMS_ENABLE_OPLIST_PATH="/tmp/ops.txt")
    _m("vllm_fl.utils", use_flaggems=_use_flaggems_default,
       get_flag_gems_whitelist_blacklist=_default_whitelist_blacklist)
    _m("vllm_fl.attention", patch_mm_encoder_attention=_patch_mm_encoder_attention_default)
    _m("vllm_fl.attention.utils",
       patch_mm_encoder_attention=_patch_mm_encoder_attention_default)
    _m("vllm_fl.ops", __path__=[])
    _m("vllm_fl.ops.custom_ops", register_oot_ops=_register_oot_ops_default)
    _m("vllm_fl.dispatch.io_common",
       managed_inference_mode=_managed_inference_mode,
       set_io_active=lambda active: None, is_io_active=lambda: False)
    _m("vllm_fl.dispatch.backends.vendor.gcu.worker.model_runner_v2",
       GPUModelRunnerFLV2=type("GPUModelRunnerFLV2", (), {}))
    _m("vllm_fl.worker.model_runner", ModelRunnerFL=type("ModelRunnerFL", (), {}))
    _m("vllm_fl.patches", __path__=[])
    _m("vllm_fl.patches.qsa_cache_hint",
       patch_qsa_cache_hint=_patch_qsa_cache_hint_default)
    _m("vllm_fl.dispatch.io_dumper",
       init_io_dump_from_env=mock.Mock(),
       is_dump_enabled=mock.Mock(return_value=False),
       register_io_module_hooks=mock.Mock(),
       advance_io_step=mock.Mock())
    _m("vllm_fl.dispatch.backends.vendor.iluvatar", __path__=[])
    _m("vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar",
       patch_torch_inductor_for_iluvatar=mock.Mock(),
       patch_sampler_compile_for_iluvatar=mock.Mock())

    _m("vllm_fl.dispatch.backends.vendor.ascend", __path__=[])
    _m("vllm_fl.dispatch.backends.vendor.ascend.impl", __path__=[])
    _m("vllm_fl.dispatch.backends.vendor.ascend.impl.triton_utils",
       init_device_properties_triton=mock.Mock())
    _m("vllm_fl.dispatch.backends.vendor.ascend.patches", __path__=[])
    _m("vllm_fl.dispatch.backends.vendor.ascend.patches.ascend_v2_device",
       install_ascend_v2_device_shims=mock.Mock())
    _m("vllm_fl.dispatch.backends.vendor.ascend.worker", __path__=[])
    _m("vllm_fl.dispatch.backends.vendor.ascend.worker.model_runner_v2",
       GPUModelRunnerFLV2=type("AscendGPUModelRunnerFLV2", (), {}))
    _m("vllm_fl.dispatch.backends.vendor.amd", __path__=[])
    _m("vllm_fl.dispatch.backends.vendor.amd.v2_runner_compat",
       install_v2_runner_compat=mock.Mock())
    _m("vllm_fl.dispatch.backends.vendor.thead", __path__=[])
    _m("vllm_fl.dispatch.backends.vendor.thead.patches", __path__=[])
    _m("vllm_fl.dispatch.backends.vendor.thead.patches.v2_runner_compat",
       install_v2_runner_compat=mock.Mock())
    _m("vllm_fl.dispatch.backends.vendor.iluvatar.worker", __path__=[])
    _m("vllm_fl.dispatch.backends.vendor.iluvatar.worker.model_runner_v2",
       IluvatarGPUModelRunnerV2=type("IluvatarGPUModelRunnerV2", (), {}))

    sys.modules["vllm_fl.dispatch"].warmup_cached_ops = mock.Mock()

    if "xgrammar" not in sys.modules:
        class BatchGrammarMatcher:
            def __init__(self, max_threads=1):
                self.max_threads = max_threads
                self.batch_calls = []

            def batch_fill_next_token_bitmask(self, matchers, bitmask, indices):
                self.batch_calls.append((matchers, bitmask, indices))

        _m("xgrammar", BatchGrammarMatcher=BatchGrammarMatcher)

    if "flag_gems" not in sys.modules:
        sys.modules["flag_gems"] = _FlagGemsStub("flag_gems")

    if "regex" not in sys.modules:
        try:
            import regex  # noqa: F401
        except Exception:  # pragma: no cover
            import re as _stdlib_re
            sys.modules["regex"] = _stdlib_re


def _bootstrap():
    _install_fake_vllm()
    _install_fake_vllm_fl()


_bootstrap()

worker = importlib.import_module("vllm_fl.worker.worker")
scheduler_fl = importlib.import_module("vllm_fl.worker.scheduler_fl")
runtime_mod = importlib.import_module("vllm_fl.worker.runtime")
nvidia_worker = importlib.import_module("vllm_fl.worker.nvidia_worker")
gcu_worker = importlib.import_module(
    "vllm_fl.dispatch.backends.vendor.gcu.worker.worker")

WorkerFL = worker.WorkerFL
AsyncIntermediateTensors = worker.AsyncIntermediateTensors
WorkerGCU = gcu_worker.WorkerGCU
NvidiaWorkerFL = nvidia_worker.NvidiaWorkerFL
platform = _PLATFORM

# --------------------------------------------------------------------------
# Config / worker helpers
# --------------------------------------------------------------------------
def make_config(**over):
    cfg = SimpleNamespace()
    cfg.num_speculative_tokens = 0
    cfg.use_v2_model_runner = False
    cfg.weight_transfer_config = None
    cfg.scheduler_config = SimpleNamespace(
        async_scheduling=False, scheduler_cls=None,
        max_num_seqs=8, max_num_batched_tokens=256)
    cfg.parallel_config = SimpleNamespace(
        assigned_physical_gpu_ids=None, local_world_size=1,
        distributed_executor_backend="mp", data_parallel_backend="mp",
        nnodes_within_dp=1, data_parallel_rank_local=0, data_parallel_index=0,
        enable_fault_tolerance=False, enable_dbo=False,
        disable_custom_all_reduce=True, distributed_timeout_seconds=None,
        world_size=1, tensor_parallel_size=1, pipeline_parallel_size=1,
        prefill_context_parallel_size=1, decode_context_parallel_size=1,
        _api_process_count=1)
    cfg.model_config = SimpleNamespace(
        dtype="float16", enable_cumem_allocator=False, enable_sleep_mode=False,
        enforce_eager=True, seed=0, multimodal_config=None, max_model_len=1024,
        is_hybrid=False, hf_text_config=None, enable_return_routed_experts=False)
    cfg.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=None, gpu_memory_utilization=0.9,
        num_gpu_blocks=0, mamba_cache_mode="none")
    cfg.speculative_config = None
    cfg.compilation_config = SimpleNamespace(
        mode=CompilationMode.NONE, backend="eager", compile_sizes=None,
        cudagraph_mode=CUDAGraphMode.NONE, cudagraph_capture_sizes=None,
        get_compile_ranges=lambda: [], compilation_time=0.0,
        encoder_compilation_time=0.0)
    cfg.device_config = SimpleNamespace(device_type="gcu", device="gcu:0")
    cfg.observability_config = SimpleNamespace(
        jit_monitor_mode="warn", jit_monitor_verbose=False)
    cfg.load_config = SimpleNamespace()
    cfg.profiler_config = SimpleNamespace(
        profiler=None, detailed_trace_annotation=False)
    cfg.kernel_config = SimpleNamespace(enable_jit_warmup=False, moe_backend=None)
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def make_worker(cfg=None, **over):
    cfg = cfg or make_config()
    w = object.__new__(WorkerFL)
    w.vllm_config = cfg
    w.rank = 0
    w.local_rank = 0
    w.is_driver_worker = False
    w.distributed_init_method = "env://"
    w.device = torch.device("cpu")
    w.use_v2_model_runner = cfg.use_v2_model_runner
    w.model_runner = mock.Mock()
    w._pp_send_work = []
    w._sleep_saved_buffers = {}
    w._sleep_saved_draft_buffers = {}
    w.profiler = None
    w.profiler_config = cfg.profiler_config
    w.weight_transfer_engine = None
    w._weight_update_active = False
    w._weight_update_is_draft = False
    w.worker_sentinel = None
    w.elastic_ep_executor = mock.Mock()
    w._sleep_mode_backend = None
    w._fl_io_dump_initialized = False
    for key, value in over.items():
        setattr(w, key, value)
    return w


def make_gcu_worker(cfg=None, **over):
    cfg = cfg or make_config()
    w = object.__new__(WorkerGCU)
    w.vllm_config = cfg
    w.rank = 0
    w.local_rank = 0
    w.distributed_init_method = "env://"
    w.is_driver_worker = False
    w.device = torch.device("cpu")
    w.use_v2_model_runner = cfg.use_v2_model_runner
    w.model_runner = mock.Mock()
    for key, value in over.items():
        setattr(w, key, value)
    return w


class FakeBuffer:
    def __init__(self, value=1):
        self.value = value
        self.data = self

    def cpu(self):
        return self

    def clone(self):
        return FakeBuffer(self.value)

    def copy_(self, other):
        self.value = other.value
        return self


class FakeModel:
    def __init__(self, buffers):
        self._buffers = buffers

    def named_buffers(self):
        return list(self._buffers.items())


def fake_accelerator(monkeypatch, mem_infos=None, device_count=8):
    acc = types.SimpleNamespace()
    acc.synchronize = mock.Mock()
    acc.empty_cache = mock.Mock()
    acc.device_count = mock.Mock(return_value=device_count)
    if mem_infos is not None:
        seq = iter(mem_infos)
        acc.get_memory_info = mock.Mock(side_effect=lambda: next(seq))
    else:
        acc.get_memory_info = mock.Mock(return_value=(0, 1))
    monkeypatch.setattr(torch, "accelerator", acc, raising=False)
    return acc


# --------------------------------------------------------------------------
# scheduler_fl.py
# --------------------------------------------------------------------------
class FakeRow:
    def __init__(self):
        self.value = None

    def fill_(self, value):
        self.value = value


class FakeBitmask:
    def __init__(self, n):
        self.rows = [FakeRow() for _ in range(n)]

    @property
    def shape(self):
        return (len(self.rows),)

    def __getitem__(self, index):
        if isinstance(index, slice):
            out = FakeBitmask(0)
            out.rows = self.rows[index]
            return out
        return self.rows[index]

    def numpy(self):
        return [row.value for row in self.rows]


class FakeMatcher:
    pass


class FakeGrammar:
    def __init__(self, matcher=True, terminated=False, accept_result=True):
        if isinstance(matcher, bool):
            self._seq = [matcher]
        else:
            self._seq = list(matcher)
        self._idx = 0
        self.terminated = terminated
        self.accept_result = accept_result
        self.accepted = []
        self.rollbacks = []

    @property
    def matcher(self):
        if not self._seq:
            return None
        value = self._seq[min(self._idx, len(self._seq) - 1)]
        self._idx += 1
        return FakeMatcher() if value else None

    def is_terminated(self):
        return self.terminated

    def accept_tokens(self, req_id, tokens):
        self.accepted.append((req_id, tokens))
        return self.accept_result

    def rollback(self, count):
        self.rollbacks.append(count)


class FakeBaseManager:
    def __init__(self, num_spec=1, max_num_seqs=4, bitmask=None, backend=None,
                 reasoners=None):
        self.vllm_config = SimpleNamespace(
            num_speculative_tokens=num_spec,
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs))
        self._grammar_bitmask = bitmask
        self.backend = backend
        self._full_mask = 7
        self.reasoners = reasoners or {}
        self.fallback_calls = []

    def _get_reasoner(self, request):
        return self.reasoners.get(request.req_id)

    def grammar_bitmask(self, requests, ids, tokens):
        self.fallback_calls.append((requests, ids, tokens))
        return "FALLBACK"


def make_requests(grammars, tokens_by_id=None):
    requests = {}
    for i, grammar in enumerate(grammars):
        req_id = f"r{i}"
        requests[req_id] = SimpleNamespace(
            req_id=req_id,
            structured_output_request=SimpleNamespace(grammar=grammar))
    return requests


def test_mtp_manager_init_and_getattr():
    base = FakeBaseManager()
    mgr = scheduler_fl._MTPStructuredOutputManager(base)
    assert mgr._base is base
    assert mgr._batch_matcher.max_threads == 8
    assert mgr._logged_active is False
    base.custom_attr = 42
    assert mgr.custom_attr == 42


def test_mtp_manager_fallback_num_spec_not_one():
    base = FakeBaseManager(num_spec=2)
    mgr = scheduler_fl._MTPStructuredOutputManager(base)
    out = mgr.grammar_bitmask({}, ["a", "b"], {})
    assert out == "FALLBACK"
    assert base.fallback_calls


def test_mtp_manager_fallback_single_request():
    base = FakeBaseManager()
    mgr = scheduler_fl._MTPStructuredOutputManager(base)
    out = mgr.grammar_bitmask({}, ["a"], {})
    assert out == "FALLBACK"


def test_mtp_manager_fallback_reasoner_present():
    base = FakeBaseManager(reasoners={"r0": object()})
    mgr = scheduler_fl._MTPStructuredOutputManager(base)
    requests = make_requests([FakeGrammar(), FakeGrammar()])
    out = mgr.grammar_bitmask(requests, ["r0", "r1"], {})
    assert out == "FALLBACK"


def test_mtp_manager_fallback_no_bitmask_no_backend():
    base = FakeBaseManager()
    mgr = scheduler_fl._MTPStructuredOutputManager(base)
    requests = make_requests([FakeGrammar(), FakeGrammar()])
    out = mgr.grammar_bitmask(requests, ["r0", "r1"], {})
    assert out == "FALLBACK"


def test_mtp_manager_allocates_bitmask_and_fills():
    allocate = mock.Mock(return_value=FakeBitmask(8))
    base = FakeBaseManager(backend=SimpleNamespace(allocate_token_bitmask=allocate))
    mgr = scheduler_fl._MTPStructuredOutputManager(base)
    requests = make_requests([FakeGrammar(), FakeGrammar()])
    out = mgr.grammar_bitmask(requests, ["r0", "r1"], {})
    allocate.assert_called_once_with(8)
    assert len(out) == 2
    assert mgr._logged_active is True


def test_mtp_manager_matcher_missing_falls_back():
    base = FakeBaseManager(bitmask=FakeBitmask(8))
    mgr = scheduler_fl._MTPStructuredOutputManager(base)
    requests = make_requests([FakeGrammar(matcher=False), FakeGrammar()])
    out = mgr.grammar_bitmask(requests, ["r0", "r1"], {})
    assert out == "FALLBACK"


def test_mtp_manager_accepts_tokens_and_rolls_back():
    base = FakeBaseManager(bitmask=FakeBitmask(8))
    mgr = scheduler_fl._MTPStructuredOutputManager(base)
    g0 = FakeGrammar()
    g1 = FakeGrammar()
    requests = make_requests([g0, g1])
    tokens = {"r0": [5]}
    out = mgr.grammar_bitmask(requests, ["r0", "r1"], tokens)
    assert len(out) == 3
    assert g0.accepted == [("r0", [5])]
    assert g0.rollbacks == [1]


def test_mtp_manager_negative_token_skips_accept():
    base = FakeBaseManager(bitmask=FakeBitmask(8))
    mgr = scheduler_fl._MTPStructuredOutputManager(base)
    g0 = FakeGrammar()
    requests = make_requests([g0, FakeGrammar()])
    out = mgr.grammar_bitmask(requests, ["r0", "r1"], {"r0": [-1]})
    assert len(out) == 3
    assert g0.accepted == []


def test_mtp_manager_bonus_fill_failure_rolls_back():
    base = FakeBaseManager(bitmask=FakeBitmask(8))
    mgr = scheduler_fl._MTPStructuredOutputManager(base)
    g0 = FakeGrammar(matcher=[True, True])
    g1 = FakeGrammar(matcher=False)
    requests = make_requests([g0, g1])
    out = mgr.grammar_bitmask(requests, ["r0", "r1"], {"r0": [5]})
    assert out == "FALLBACK"
    assert g0.rollbacks == [1]


def test_mtp_manager_terminated_grammar_fills_full_mask():
    base = FakeBaseManager(bitmask=FakeBitmask(8))
    mgr = scheduler_fl._MTPStructuredOutputManager(base)
    requests = make_requests([FakeGrammar(terminated=True), FakeGrammar()])
    out = mgr.grammar_bitmask(requests, ["r0", "r1"], {})
    assert len(out) == 2
    assert out[0] == 7


def test_async_scheduler_fl_wraps_manager_for_mtp1():
    sched = scheduler_fl.AsyncSchedulerFL(
        vllm_config=SimpleNamespace(num_speculative_tokens=1),
        structured_output_manager=FakeBaseManager())
    wrapped = sched.kwargs["structured_output_manager"]
    assert isinstance(wrapped, scheduler_fl._MTPStructuredOutputManager)


def test_async_scheduler_fl_passthrough_when_not_mtp1():
    manager = FakeBaseManager()
    sched = scheduler_fl.AsyncSchedulerFL(
        vllm_config=SimpleNamespace(num_speculative_tokens=2),
        structured_output_manager=manager)
    assert sched.kwargs["structured_output_manager"] is manager


def test_async_scheduler_fl_passthrough_without_config_or_manager():
    sched = scheduler_fl.AsyncSchedulerFL(vllm_config=None,
                                          structured_output_manager=None)
    assert sched.kwargs["structured_output_manager"] is None


# --------------------------------------------------------------------------
# runtime.py
# --------------------------------------------------------------------------
def _flag_gems():
    return sys.modules["flag_gems"]


@pytest.fixture
def runtime_env(monkeypatch):
    patch_mm = mock.Mock()
    reg = mock.Mock()
    monkeypatch.setattr(runtime_mod, "patch_mm_encoder_attention", patch_mm)
    monkeypatch.setattr(sys.modules["vllm_fl.ops.custom_ops"],
                        "register_oot_ops", reg)
    fg = _flag_gems()
    fg.only_enable = mock.Mock()
    fg.enable = mock.Mock()
    return patch_mm, reg, fg


def test_runtime_disabled_returns_early(runtime_env, monkeypatch):
    patch_mm, reg, fg = runtime_env
    monkeypatch.setattr(runtime_mod.fl_envs, "USE_FLAGGEMS", False)
    runtime_mod._initialize_fl_runtime(0)
    patch_mm.assert_called_once()
    reg.assert_called_once()
    fg.enable.assert_not_called()
    fg.only_enable.assert_not_called()


def test_runtime_whitelist_only_enable(runtime_env, monkeypatch):
    patch_mm, reg, fg = runtime_env
    monkeypatch.setattr(runtime_mod.fl_envs, "USE_FLAGGEMS", True)
    monkeypatch.setattr(runtime_mod, "get_flag_gems_whitelist_blacklist",
                        lambda: (["a", "b"], None))
    runtime_mod._initialize_fl_runtime(0)
    fg.only_enable.assert_called_once()
    kwargs = fg.only_enable.call_args.kwargs
    assert kwargs["include"] == ["a", "b"]
    assert kwargs["record"] is True
    assert kwargs["once"] is True


def test_runtime_blacklist_enable_unused(runtime_env, monkeypatch):
    patch_mm, reg, fg = runtime_env
    monkeypatch.setattr(runtime_mod.fl_envs, "USE_FLAGGEMS", True)
    monkeypatch.setattr(runtime_mod, "get_flag_gems_whitelist_blacklist",
                        lambda: (None, ["x"]))
    monkeypatch.setattr(platform, "vendor_name", "enflame")
    runtime_mod._initialize_fl_runtime(1)
    fg.enable.assert_called_once()
    kwargs = fg.enable.call_args.kwargs
    assert kwargs["unused"] == ["x"]
    assert kwargs["record"] is False


def test_runtime_blacklist_amd_uses_enable_except(runtime_env, monkeypatch):
    patch_mm, reg, fg = runtime_env
    monkeypatch.setattr(runtime_mod.fl_envs, "USE_FLAGGEMS", True)
    monkeypatch.setattr(runtime_mod, "get_flag_gems_whitelist_blacklist",
                        lambda: (None, ["x"]))
    monkeypatch.setattr(platform, "vendor_name", "amd")
    fake_except = mock.Mock()
    monkeypatch.setattr(runtime_mod, "_enable_except", fake_except)
    runtime_mod._initialize_fl_runtime(0)
    fake_except.assert_called_once()


def test_runtime_enable_all(runtime_env, monkeypatch):
    patch_mm, reg, fg = runtime_env
    monkeypatch.setattr(runtime_mod.fl_envs, "USE_FLAGGEMS", True)
    monkeypatch.setattr(runtime_mod, "get_flag_gems_whitelist_blacklist",
                        lambda: (None, None))
    runtime_mod._initialize_fl_runtime(0)
    fg.enable.assert_called_once()
    assert "unused" not in fg.enable.call_args.kwargs


def test_enable_except_builds_include_list(monkeypatch):
    fg = mock.Mock()
    fg._FULL_CONFIG = [("add",), ("mul",), ("pow.Scalar",)]
    common = {"record": True, "once": True, "path": "/tmp/x"}
    runtime_mod._enable_except(fg, ["pow.Scalar", "missing"], common)
    fg.only_enable.assert_called_once()
    kwargs = fg.only_enable.call_args.kwargs
    assert kwargs["include"] == ["add", "mul"]
    assert kwargs["path"] == "/tmp/x"


def test_enable_except_falls_back_when_full_config_unavailable():
    class BadList:
        def __iter__(self):
            raise RuntimeError("boom")

    fg = mock.Mock()
    fg._FULL_CONFIG = BadList()
    runtime_mod._enable_except(fg, ["pow.Scalar"], {"record": False})
    fg.enable.assert_called_once()
    assert fg.enable.call_args.kwargs["unused"] == ["pow.Scalar"]
    fg.only_enable.assert_not_called()


# --------------------------------------------------------------------------
# nvidia_worker.py
# --------------------------------------------------------------------------
def _decorated_method(result="ok"):
    def impl(*args, **kwargs):
        return result

    def wrapper(*args, **kwargs):
        return impl(*args, **kwargs)

    wrapper.__wrapped__ = impl
    return wrapper


def test_install_native_runner_io_methods_installs_wrapped():
    runner = SimpleNamespace(
        execute_model=_decorated_method("e"),
        sample_tokens=_decorated_method("s"),
        pool=_decorated_method("p"))
    nvidia_worker._install_native_runner_io_methods(runner)
    assert getattr(runner.execute_model, "_vllm_fl_managed_io", False)
    assert runner.execute_model() == "e"


def test_install_native_runner_io_methods_skips_missing_and_managed():
    managed = _decorated_method("m")
    managed._vllm_fl_managed_io = True
    runner = SimpleNamespace(execute_model=managed)
    nvidia_worker._install_native_runner_io_methods(runner)
    assert runner.execute_model is managed


def test_install_native_runner_io_methods_requires_wrapper():
    runner = SimpleNamespace(execute_model=lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        nvidia_worker._install_native_runner_io_methods(runner)


def test_call_native_worker_with_managed_inference_ok():
    method = _decorated_method("native")
    out = nvidia_worker._call_native_worker_with_managed_inference(
        method, object())
    assert out == "native"


def test_call_native_worker_requires_wrapper():
    def method(self):
        return None

    with pytest.raises(RuntimeError):
        nvidia_worker._call_native_worker_with_managed_inference(method, object())


def test_nvidia_worker_init(monkeypatch):
    init_runtime = mock.Mock()
    monkeypatch.setattr(nvidia_worker, "_initialize_fl_runtime", init_runtime)
    w = NvidiaWorkerFL(vllm_config=make_config(), local_rank=0, rank=3,
                       distributed_init_method="env://")
    assert w.rank == 3
    assert w._fl_io_dump_initialized is False
    init_runtime.assert_called_once_with(3)


@pytest.mark.parametrize("patched", [True, False])
def test_nvidia_worker_init_device(monkeypatch, patched):
    monkeypatch.setattr(GPUWorker, "init_device", mock.Mock())
    qsa = sys.modules["vllm_fl.patches.qsa_cache_hint"]
    patcher = mock.Mock(return_value=patched)
    monkeypatch.setattr(qsa, "patch_qsa_cache_hint", patcher)
    w = object.__new__(NvidiaWorkerFL)
    w.vllm_config = make_config()
    w.device = torch.device("cpu")
    w.init_device()
    patcher.assert_called_once_with(w.model_config, w.device)


def test_nvidia_ensure_io_dump_disabled(monkeypatch):
    w = object.__new__(NvidiaWorkerFL)
    w.vllm_config = make_config()
    w._fl_io_dump_initialized = False
    w.model_runner = mock.Mock()
    dumper = sys.modules["vllm_fl.dispatch.io_dumper"]
    monkeypatch.setattr(dumper, "is_dump_enabled", mock.Mock(return_value=False))
    install = mock.Mock()
    monkeypatch.setattr(nvidia_worker, "_install_native_runner_io_methods", install)
    w._ensure_io_dump_initialized()
    install.assert_not_called()
    assert w._fl_io_dump_initialized is True


def test_nvidia_ensure_io_dump_enabled_installs_hooks(monkeypatch):
    w = object.__new__(NvidiaWorkerFL)
    w.vllm_config = make_config()
    w._fl_io_dump_initialized = False
    w.model_runner = mock.Mock()
    w.model_runner.get_model.return_value = "model"
    dumper = sys.modules["vllm_fl.dispatch.io_dumper"]
    monkeypatch.setattr(dumper, "is_dump_enabled", mock.Mock(return_value=True))
    monkeypatch.setattr(dumper, "register_io_module_hooks", mock.Mock())
    install = mock.Mock()
    monkeypatch.setattr(nvidia_worker, "_install_native_runner_io_methods", install)
    w._ensure_io_dump_initialized()
    install.assert_called_once_with(w.model_runner)
    dumper.register_io_module_hooks.assert_called_once_with("model")


def test_nvidia_ensure_io_dump_already_initialized(monkeypatch):
    w = object.__new__(NvidiaWorkerFL)
    w.vllm_config = make_config()
    w._fl_io_dump_initialized = True
    dumper = sys.modules["vllm_fl.dispatch.io_dumper"]
    called = mock.Mock()
    monkeypatch.setattr(dumper, "init_io_dump_from_env", called)
    w._ensure_io_dump_initialized()
    called.assert_not_called()


def test_nvidia_sample_tokens_advances(monkeypatch):
    w = object.__new__(NvidiaWorkerFL)
    w._fl_io_dump_initialized = True
    w._native_sample_result = "out"
    dumper = sys.modules["vllm_fl.dispatch.io_dumper"]
    advance = mock.Mock()
    monkeypatch.setattr(dumper, "advance_io_step", advance)
    assert w.sample_tokens(None) == "out"
    advance.assert_called_once()


def test_nvidia_sample_tokens_none_no_advance(monkeypatch):
    w = object.__new__(NvidiaWorkerFL)
    w._fl_io_dump_initialized = True
    w._native_sample_result = None
    dumper = sys.modules["vllm_fl.dispatch.io_dumper"]
    advance = mock.Mock()
    monkeypatch.setattr(dumper, "advance_io_step", advance)
    assert w.sample_tokens(None) is None
    advance.assert_not_called()


def test_nvidia_execute_model_advances(monkeypatch):
    w = object.__new__(NvidiaWorkerFL)
    w._fl_io_dump_initialized = True
    w._native_exec_result = "out"
    dumper = sys.modules["vllm_fl.dispatch.io_dumper"]
    advance = mock.Mock()
    monkeypatch.setattr(dumper, "advance_io_step", advance)
    assert w.execute_model(object()) == "out"
    advance.assert_called_once()


def test_nvidia_execute_model_none_no_advance(monkeypatch):
    w = object.__new__(NvidiaWorkerFL)
    w._fl_io_dump_initialized = True
    w._native_exec_result = None
    dumper = sys.modules["vllm_fl.dispatch.io_dumper"]
    advance = mock.Mock()
    monkeypatch.setattr(dumper, "advance_io_step", advance)
    assert w.execute_model(object()) is None
    advance.assert_not_called()


# --------------------------------------------------------------------------
# worker.py module-level helpers
# --------------------------------------------------------------------------
def test_num_workspace_lanes():
    assert worker._num_workspace_lanes(
        SimpleNamespace(speculative_config=None), False) == 1
    assert worker._num_workspace_lanes(
        SimpleNamespace(speculative_config=None), True) == 1
    spec = SimpleNamespace(use_dspark=lambda: True)
    assert worker._num_workspace_lanes(
        SimpleNamespace(speculative_config=spec), True) == 2
    assert worker._num_workspace_lanes(
        SimpleNamespace(speculative_config=spec), False) == 1


@pytest.mark.parametrize("device_type,backend,expected", [
    ("ptpu", "flagcx", "pccl"),
    ("ptpu", "nccl", "pccl"),
    ("npu", "flagcx", "hccl"),
    ("gcu", "flagcx", "flagcx"),
    ("cuda", "nccl", "nccl"),
])
def test_normalize_process_group_backend(monkeypatch, device_type, backend, expected):
    monkeypatch.setattr(platform, "device_type", device_type)
    monkeypatch.setattr(platform, "vendor_name", "enflame")
    assert worker._normalize_process_group_backend(backend) == expected


def test_normalize_process_group_backend_amd():
    orig = platform.device_type
    platform.device_type = "cuda"
    platform.vendor_name = "amd"
    try:
        assert worker._normalize_process_group_backend("flagcx") == "nccl"
    finally:
        platform.device_type = orig
        platform.vendor_name = "enflame"


def test_is_hybrid_kv_model_mamba_mode():
    cfg = make_config()
    cfg.cache_config.mamba_cache_mode = "align"
    assert worker._is_hybrid_kv_model(cfg) is True


def test_is_hybrid_kv_model_flag():
    cfg = make_config()
    cfg.model_config.is_hybrid = True
    assert worker._is_hybrid_kv_model(cfg) is True


def test_is_hybrid_kv_model_exception_then_layer_types():
    class ModelConfig:
        is_hybrid = property(lambda self: (_ for _ in ()).throw(RuntimeError()))
        hf_text_config = SimpleNamespace(layer_types=["linear_attention"])

    cfg = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="none"),
                          model_config=ModelConfig())
    assert worker._is_hybrid_kv_model(cfg) is True


def test_is_hybrid_kv_model_false():
    cfg = make_config()
    cfg.model_config.hf_text_config = SimpleNamespace(layer_types=["full_attention"])
    assert worker._is_hybrid_kv_model(cfg) is False


def test_apply_worker_vendor_patches_skips_non_iluvatar(monkeypatch):
    monkeypatch.setattr(platform, "vendor_name", "enflame")
    worker._apply_worker_vendor_patches()


def test_apply_worker_vendor_patches_iluvatar(monkeypatch):
    monkeypatch.setattr(platform, "vendor_name", "iluvatar")
    mod = sys.modules["vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar"]
    monkeypatch.setattr(mod, "patch_torch_inductor_for_iluvatar", mock.Mock())
    monkeypatch.setattr(mod, "patch_sampler_compile_for_iluvatar", mock.Mock())
    worker._apply_worker_vendor_patches()
    mod.patch_torch_inductor_for_iluvatar.assert_called_once()


def test_apply_worker_vendor_patches_swallows_errors(monkeypatch):
    monkeypatch.setattr(platform, "vendor_name", "iluvatar")
    mod = sys.modules["vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar"]
    monkeypatch.setattr(mod, "patch_torch_inductor_for_iluvatar",
                        mock.Mock(side_effect=RuntimeError("x")))
    worker._apply_worker_vendor_patches()


def test_async_intermediate_tensors_waits_once():
    handle = Handle()
    post = mock.Mock()
    tensors = AsyncIntermediateTensors({"h": 1}, comm_handles=[handle],
                                       comm_postprocess=[post])
    assert tensors.tensors == {"h": 1}
    assert handle.waited is True
    post.assert_called_once()
    assert tensors.tensors == {"h": 1}
    post.assert_called_once()


def test_async_intermediate_tensors_empty_handles():
    tensors = AsyncIntermediateTensors({"h": 1})
    assert tensors.tensors == {"h": 1}
    assert tensors._comm_waited is True


# --------------------------------------------------------------------------
# WorkerFL.__init__ and sleep/wake
# --------------------------------------------------------------------------
def test_workerfl_init_sets_triton_cache_dir(monkeypatch):
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.setattr(platform, "device_type", "gcu")
    monkeypatch.setattr(worker, "_initialize_fl_runtime", mock.Mock())
    w = WorkerFL(make_config(), 0, 5, "env://")
    assert os.environ["TRITON_CACHE_DIR"].endswith("rank-5")
    assert w.use_v2_model_runner is False
    assert w.elastic_ep_executor is not None


def test_workerfl_init_spec_scheduler_override(monkeypatch):
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.setattr(platform, "device_type", "gcu")
    monkeypatch.setattr(worker, "_initialize_fl_runtime", mock.Mock())
    cfg = make_config(num_speculative_tokens=1)
    cfg.scheduler_config.async_scheduling = True
    w = WorkerFL(cfg, 0, 0, "env://")
    assert w.vllm_config.scheduler_config.scheduler_cls == (
        "vllm_fl.worker.scheduler_fl.AsyncSchedulerFL")


def test_workerfl_init_kunlunxin_hybrid_env(monkeypatch):
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.delenv("USE_RESHAPE_AND_CACHE_FLASH", raising=False)
    monkeypatch.setattr(platform, "device_type", "gcu")
    monkeypatch.setattr(platform, "vendor_name", "kunlunxin")
    monkeypatch.setattr(worker, "_initialize_fl_runtime", mock.Mock())
    cfg = make_config()
    cfg.cache_config.mamba_cache_mode = "align"
    WorkerFL(cfg, 0, 0, "env://")
    assert os.environ["USE_RESHAPE_AND_CACHE_FLASH"] == "1"


def test_workerfl_init_fault_tolerance_sentinel(monkeypatch):
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.setattr(platform, "device_type", "gcu")
    monkeypatch.setattr(platform, "vendor_name", "enflame")
    monkeypatch.setattr(worker, "_initialize_fl_runtime", mock.Mock())
    cfg = make_config()
    cfg.parallel_config.enable_fault_tolerance = True
    w = WorkerFL(cfg, 0, 0, "env://")
    assert isinstance(w.worker_sentinel, WorkerSentinel)


def test_get_sleep_mode_backend_caches(monkeypatch):
    backend = object()
    mod = sys.modules["vllm.device_allocator.sleep_mode_backend"]
    factory = SimpleNamespace(create_backend=mock.Mock(return_value=backend))
    monkeypatch.setattr(mod, "SleepModeBackendFactory", factory)
    w = make_worker()
    assert w._get_sleep_mode_backend() is backend
    assert w._get_sleep_mode_backend() is backend
    factory.create_backend.assert_called_once()


def test_sleep_level1_frees_memory(monkeypatch):
    acc = fake_accelerator(monkeypatch, mem_infos=[(100, 200), (150, 200)])
    backend = mock.Mock()
    w = make_worker()
    w._get_sleep_mode_backend = mock.Mock(return_value=backend)
    w.model_runner = SimpleNamespace(model=FakeModel({}))
    w.sleep(1)
    backend.suspend.assert_called_once_with(1)
    assert acc.get_memory_info.call_count == 2
    assert w._sleep_saved_buffers == {}


def test_sleep_level2_saves_buffers_and_draft(monkeypatch):
    fake_accelerator(monkeypatch, mem_infos=[(100, 200), (200, 200)])
    backend = mock.Mock()
    draft = FakeModel({"d": FakeBuffer(4)})
    w = make_worker()
    w._get_sleep_mode_backend = mock.Mock(return_value=backend)
    w.model_runner = SimpleNamespace(
        model=FakeModel({"a": FakeBuffer(3)}),
        get_draft_model=mock.Mock(return_value=draft))
    w.sleep(2)
    assert w._sleep_saved_buffers["a"].value == 3
    assert w._sleep_saved_draft_buffers["d"].value == 4


def test_sleep_negative_freed_asserts(monkeypatch):
    fake_accelerator(monkeypatch, mem_infos=[(100, 200), (50, 200)])
    backend = mock.Mock()
    w = make_worker()
    w._get_sleep_mode_backend = mock.Mock(return_value=backend)
    w.model_runner = SimpleNamespace(model=FakeModel({}))
    with pytest.raises(AssertionError):
        w.sleep(1)


def test_wake_up_restores_weights_and_kv(monkeypatch):
    backend = mock.Mock()
    w = make_worker()
    w._get_sleep_mode_backend = mock.Mock(return_value=backend)
    main_model = FakeModel({"a": FakeBuffer(0)})
    draft = FakeModel({"d": FakeBuffer(0)})
    post_kv = mock.Mock()
    w.model_runner = SimpleNamespace(model=main_model,
                                     get_draft_model=mock.Mock(return_value=draft),
                                     post_kv_cache_wake_up=post_kv)
    w._sleep_saved_buffers = {"a": FakeBuffer(9)}
    w._sleep_saved_draft_buffers = {"d": FakeBuffer(8)}
    w.wake_up(None)
    assert main_model._buffers["a"].value == 9
    assert draft._buffers["d"].value == 8
    assert w._sleep_saved_buffers == {}
    assert w._sleep_saved_draft_buffers == {}
    post_kv.assert_called_once()
    backend.resume.assert_called_once_with(None)


def test_wake_up_weights_only(monkeypatch):
    backend = mock.Mock()
    w = make_worker()
    w._get_sleep_mode_backend = mock.Mock(return_value=backend)
    main_model = FakeModel({"a": FakeBuffer(0)})
    post_kv = mock.Mock()
    w.model_runner = SimpleNamespace(
        model=main_model,
        get_draft_model=mock.Mock(return_value=None),
        post_kv_cache_wake_up=post_kv)
    w._sleep_saved_buffers = {"a": FakeBuffer(5)}
    w.wake_up(["weights"])
    assert main_model._buffers["a"].value == 5
    post_kv.assert_not_called()


def test_wake_up_kv_only(monkeypatch):
    backend = mock.Mock()
    w = make_worker()
    w._get_sleep_mode_backend = mock.Mock(return_value=backend)
    main_model = FakeModel({"a": FakeBuffer(0)})
    post_kv = mock.Mock()
    w.model_runner = SimpleNamespace(model=main_model,
                                     get_draft_model=mock.Mock(return_value=None),
                                     post_kv_cache_wake_up=post_kv)
    w._sleep_saved_buffers = {"a": FakeBuffer(5)}
    w.wake_up(["kv_cache"])
    assert main_model._buffers["a"].value == 0
    post_kv.assert_called_once()


def test_checkpoint_prepare_restore(monkeypatch):
    prep = mock.Mock()
    restore = mock.Mock()
    monkeypatch.setattr(worker, "checkpoint_prepare_distributed_state", prep)
    monkeypatch.setattr(worker, "checkpoint_restore_distributed_state", restore)
    w = make_worker()
    w.checkpoint_prepare()
    w.checkpoint_restore()
    prep.assert_called_once()
    restore.assert_called_once()


def _assert_nullcontext(ctx):
    assert isinstance(ctx, contextlib.nullcontext)


def test_memory_pool_context_npu(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "npu")
    w = make_worker()
    _assert_nullcontext(w._maybe_get_memory_pool_context("weights"))


def test_memory_pool_context_cuda_no_cumem(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "cuda")
    monkeypatch.setattr(platform, "is_cuda_alike", lambda: True)
    w = make_worker()
    w.vllm_config.model_config.enable_cumem_allocator = False
    _assert_nullcontext(w._maybe_get_memory_pool_context("weights"))


def test_memory_pool_context_xpu_no_sleep(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "xpu")
    monkeypatch.setattr(platform, "is_xpu", lambda: True)
    w = make_worker()
    _assert_nullcontext(w._maybe_get_memory_pool_context("weights"))


def test_memory_pool_context_cpu(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "cpu")
    monkeypatch.setattr(platform, "is_cpu", lambda: True)
    w = make_worker()
    _assert_nullcontext(w._maybe_get_memory_pool_context("weights"))


def test_memory_pool_context_no_sleep_support(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "gcu")
    monkeypatch.setattr(platform, "is_sleep_mode_available", lambda: False)
    w = make_worker()
    _assert_nullcontext(w._maybe_get_memory_pool_context("weights"))


def test_memory_pool_context_allocator(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "gcu")
    monkeypatch.setattr(platform, "is_sleep_mode_available", lambda: True)
    sentinel = object()
    alloc = mock.Mock()
    alloc.get_current_usage.return_value = 0
    alloc.use_memory_pool.return_value = sentinel
    monkeypatch.setattr(worker, "get_mem_allocator_instance", lambda: alloc)
    w = make_worker()
    assert w._maybe_get_memory_pool_context("weights") is sentinel
    alloc.use_memory_pool.assert_called_once_with(tag="weights")


def test_scoped_allocator_max_split_non_cuda(monkeypatch):
    monkeypatch.setattr(platform, "is_cuda", lambda: False)
    w = make_worker()
    with w._scoped_allocator_max_split(20):
        pass


def test_scoped_allocator_max_split_restores(monkeypatch):
    monkeypatch.setattr(platform, "is_cuda", lambda: True)
    setter = mock.Mock()
    monkeypatch.setattr(torch._C, "_accelerator_setAllocatorSettings",
                        setter, raising=False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
    w = make_worker()
    with w._scoped_allocator_max_split(20):
        pass
    assert setter.call_args_list[0].args[0] == "max_split_size_mb:20"
    assert setter.call_args_list[1].args[0] == "max_split_size_mb:64"


# --------------------------------------------------------------------------
# WorkerFL.init_device
# --------------------------------------------------------------------------
def _stub_init_device(monkeypatch, free=2000, total=3000, requested=1500,
                      device_count=8):
    monkeypatch.setattr(worker, "init_worker_distributed_environment", mock.Mock())
    monkeypatch.setattr(worker, "MemorySnapshot",
                        lambda device=None: SimpleNamespace(
                            free_memory=free, total_memory=total))
    req = mock.Mock(return_value=requested)
    monkeypatch.setattr(worker, "request_memory", req)
    ws = mock.Mock()
    monkeypatch.setattr(worker, "init_workspace_manager", ws)
    rep = mock.Mock()
    monkeypatch.setattr(worker, "report_usage_stats", rep)
    monkeypatch.setattr(worker, "set_random_seed", mock.Mock())
    acc = fake_accelerator(monkeypatch, device_count=device_count)
    return req, ws, rep, acc


def test_init_device_unsupported_type(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "cpu")
    cfg = make_config()
    cfg.device_config.device_type = "cuda"
    w = make_worker(cfg)
    with pytest.raises(RuntimeError):
        w.init_device()


def test_init_device_v1_constructs_model_runner(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "cpu")
    monkeypatch.setattr(platform, "vendor_name", "enflame")
    cfg = make_config()
    cfg.device_config.device_type = "cpu"
    _, ws, rep, _ = _stub_init_device(monkeypatch)
    sentinel = object()
    mr_mod = sys.modules["vllm_fl.worker.model_runner"]
    factory = mock.Mock(return_value=sentinel)
    monkeypatch.setattr(mr_mod, "ModelRunnerFL", factory)
    w = make_worker(cfg)
    w.init_device()
    factory.assert_called_once()
    assert w.model_runner is sentinel
    ws.assert_called_once()
    rep.assert_called_once()
    assert w.init_snapshot.free_memory == 2000
    assert w.requested_memory == 1500


def test_init_device_v2_constructs_runner(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "cpu")
    monkeypatch.setattr(platform, "vendor_name", "enflame")
    cfg = make_config(use_v2_model_runner=True)
    cfg.device_config.device_type = "cpu"
    _stub_init_device(monkeypatch)
    sentinel = object()
    mr_mod = sys.modules["vllm.v1.worker.gpu.model_runner"]
    factory = mock.Mock(return_value=sentinel)
    monkeypatch.setattr(mr_mod, "GPUModelRunner", factory)
    w = make_worker(cfg)
    w.init_device()
    factory.assert_called_once()
    assert w.model_runner is sentinel


def test_init_device_assigned_ids_branch(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "cpu")
    monkeypatch.setattr(platform, "vendor_name", "enflame")
    cfg = make_config()
    cfg.device_config.device_type = "cpu"
    cfg.parallel_config.assigned_physical_gpu_ids = [0, 1]
    cfg.parallel_config.local_world_size = 1
    _stub_init_device(monkeypatch)
    mr_mod = sys.modules["vllm_fl.worker.model_runner"]
    monkeypatch.setattr(mr_mod, "ModelRunnerFL", mock.Mock(return_value=object()))
    set_assigned = mock.Mock()
    iface = sys.modules["vllm.platforms.interface"]
    monkeypatch.setattr(iface, "set_assigned_physical_gpu_ids", set_assigned)
    w = make_worker(cfg)
    w.init_device()
    set_assigned.assert_called_once_with([0, 1])


def test_init_device_assigned_ids_out_of_bounds(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "cpu")
    cfg = make_config()
    cfg.device_config.device_type = "cpu"
    cfg.parallel_config.assigned_physical_gpu_ids = [0]
    cfg.parallel_config.local_world_size = 2
    _stub_init_device(monkeypatch)
    iface = sys.modules["vllm.platforms.interface"]
    monkeypatch.setattr(iface, "set_assigned_physical_gpu_ids", mock.Mock())
    w = make_worker(cfg)
    with pytest.raises(AssertionError):
        w.init_device()


def test_init_device_ray_backend_skips_local_world_size_check(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "cpu")
    monkeypatch.setattr(platform, "vendor_name", "enflame")
    cfg = make_config()
    cfg.device_config.device_type = "cpu"
    cfg.parallel_config.assigned_physical_gpu_ids = [0]
    cfg.parallel_config.local_world_size = 2
    cfg.parallel_config.distributed_executor_backend = "ray"
    _stub_init_device(monkeypatch)
    mr_mod = sys.modules["vllm_fl.worker.model_runner"]
    monkeypatch.setattr(mr_mod, "ModelRunnerFL", mock.Mock(return_value=object()))
    iface = sys.modules["vllm.platforms.interface"]
    monkeypatch.setattr(iface, "set_assigned_physical_gpu_ids", mock.Mock())
    w = make_worker(cfg)
    w.init_device()


def test_init_device_dp_offset(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "cpu")
    monkeypatch.setattr(platform, "vendor_name", "enflame")
    cfg = make_config()
    cfg.device_config.device_type = "cpu"
    cfg.parallel_config.data_parallel_rank_local = None
    cfg.parallel_config.data_parallel_index = 2
    cfg.parallel_config.local_world_size = 2
    _stub_init_device(monkeypatch, device_count=10)
    mr_mod = sys.modules["vllm_fl.worker.model_runner"]
    monkeypatch.setattr(mr_mod, "ModelRunnerFL", mock.Mock(return_value=object()))
    w = make_worker(cfg)
    w.init_device()
    assert w.local_rank == 4


def test_handle_ft_command(monkeypatch):
    w = make_worker()
    w.worker_sentinel = WorkerSentinel()
    assert w.handle_ft_command("cmd") == ("handled", "cmd")
    w.worker_sentinel = None
    with pytest.raises(AssertionError):
        w.handle_ft_command("cmd")


def test_load_model_basic(monkeypatch):
    w = make_worker()
    w._maybe_get_memory_pool_context = mock.Mock(
        return_value=contextlib.nullcontext())
    w._scoped_allocator_max_split = mock.Mock(
        return_value=contextlib.nullcontext())
    w.model_runner.load_model = mock.Mock()
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    w.load_model(load_dummy_weights=True)
    w.model_runner.load_model.assert_called_once_with(load_dummy_weights=True)


def test_load_model_creates_weight_transfer_engine(monkeypatch):
    cfg = make_config()
    cfg.weight_transfer_config = object()
    w = make_worker(cfg)
    w._maybe_get_memory_pool_context = mock.Mock(
        return_value=contextlib.nullcontext())
    w._scoped_allocator_max_split = mock.Mock(
        return_value=contextlib.nullcontext())
    w.model_runner.load_model = mock.Mock()
    w.model_runner.get_model.return_value = "model"
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    engine = object()
    monkeypatch.setattr(worker.WeightTransferEngineFactory, "create_engine",
                        mock.Mock(return_value=engine))
    w.load_model()
    assert w.weight_transfer_engine is engine


def test_load_model_npu_restores_thread_settings(monkeypatch):
    w = make_worker()
    w.device = SimpleNamespace(type="npu")
    w._maybe_get_memory_pool_context = mock.Mock(
        return_value=contextlib.nullcontext())
    w._scoped_allocator_max_split = mock.Mock(
        return_value=contextlib.nullcontext())
    w.model_runner.load_model = mock.Mock()
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    setter = mock.Mock()
    monkeypatch.setattr(worker, "set_torch_threads_for_runtime", setter)
    w.load_model()
    setter.assert_called_once()


def test_update_config_and_reload_weights(monkeypatch):
    w = make_worker()
    w.model_runner.update_config = mock.Mock()
    w.model_runner.reload_weights = mock.Mock()
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    w.update_config({"a": 1})
    w.reload_weights(1, b=2)
    w.model_runner.update_config.assert_called_once_with({"a": 1})
    w.model_runner.reload_weights.assert_called_once_with(1, b=2)


# --------------------------------------------------------------------------
# determine_available_memory
# --------------------------------------------------------------------------
def test_determine_memory_txda_preset(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "txda")
    cfg = make_config()
    cfg.cache_config.kv_cache_memory_bytes = 1234
    w = make_worker(cfg)
    assert w.determine_available_memory() == 1234


def test_determine_memory_txda_env_default(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "txda")
    monkeypatch.delenv("VLLM_TXDA_KV_CACHE_SIZE", raising=False)
    w = make_worker(make_config())
    assert w.determine_available_memory() == 20 * (1 << 30)


def test_determine_memory_txda_env_value(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "txda")
    monkeypatch.setenv("VLLM_TXDA_KV_CACHE_SIZE", "4096")
    w = make_worker(make_config())
    assert w.determine_available_memory() == 4096


def test_determine_memory_txda_env_invalid(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "txda")
    monkeypatch.setenv("VLLM_TXDA_KV_CACHE_SIZE", "notint")
    w = make_worker(make_config())
    with pytest.raises(ValueError):
        w.determine_available_memory()


def test_determine_memory_txda_non_positive(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "txda")
    cfg = make_config()
    cfg.cache_config.kv_cache_memory_bytes = 0
    w = make_worker(cfg)
    with pytest.raises(ValueError):
        w.determine_available_memory()


def test_determine_memory_explicit_kv_bytes(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "gcu")
    monkeypatch.setattr(platform, "vendor_name", "enflame")
    cfg = make_config()
    cfg.cache_config.kv_cache_memory_bytes = 1000
    reserve = mock.Mock(side_effect=lambda b, *a, **k: b)
    monkeypatch.setattr(worker, "maybe_apply_startup_plan", mock.Mock())
    monkeypatch.setattr(worker, "reserve_mm_ipc_gpu_memory", reserve)
    w = make_worker(cfg)
    w.init_snapshot = SimpleNamespace(free_memory=2000, total_memory=3000)
    w.model_runner.profile_run = mock.Mock()
    assert w.determine_available_memory() == 1000
    w.model_runner.profile_run.assert_called_once()
    reserve.assert_called_once()


def test_determine_memory_amd_warmup_and_profile_config(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "gcu")
    monkeypatch.setattr(platform, "vendor_name", "amd")
    cfg = make_config(use_v2_model_runner=True)
    cfg.cache_config.kv_cache_memory_bytes = 1000
    monkeypatch.setattr(worker, "maybe_apply_startup_plan", mock.Mock())
    monkeypatch.setattr(worker, "reserve_mm_ipc_gpu_memory",
                        mock.Mock(side_effect=lambda b, *a, **k: b))
    warm = mock.Mock()
    sys.modules["vllm_fl.dispatch"].warmup_cached_ops = warm
    w = make_worker(cfg)
    w.init_snapshot = SimpleNamespace(free_memory=2000, total_memory=3000)
    w.model_runner.profile_run = mock.Mock()
    assert w.determine_available_memory() == 1000
    warm.assert_called_once_with(amd_compile=True)


def _patch_profiling(monkeypatch, profile_result):
    @contextlib.contextmanager
    def _ctx(snapshot, weights_memory=0):
        yield profile_result

    monkeypatch.setattr(worker, "memory_profiling", _ctx)


def test_determine_memory_profiling_path(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "gcu")
    monkeypatch.setattr(platform, "vendor_name", "enflame")
    monkeypatch.setattr(platform, "is_cuda_alike", lambda: True)
    cfg = make_config()
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    monkeypatch.setattr(worker.envs, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
                        True)
    monkeypatch.setattr(worker, "maybe_apply_startup_plan", mock.Mock())
    reserve = mock.Mock(side_effect=lambda b, *a, **k: b)
    monkeypatch.setattr(worker, "reserve_mm_ipc_gpu_memory", reserve)
    profile_result = SimpleNamespace(
        total_consumed=200, transient_peak_headroom=50,
        non_kv_cache_memory=100,
        after_profile=SimpleNamespace(free_memory=1900))
    _patch_profiling(monkeypatch, profile_result)
    w = make_worker(cfg)
    w.init_snapshot = SimpleNamespace(free_memory=2000, total_memory=3000)
    w.requested_memory = 1500
    w.model_runner.model_memory_usage = 100
    w.model_runner.profile_run = mock.Mock()
    w.model_runner.profile_cudagraph_memory = mock.Mock(return_value=30)
    assert w.determine_available_memory() == 1370
    assert w.total_consumed == 200
    assert w.peak_activation_memory == 80
    reserve.assert_called_once()
    assert reserve.call_args.args[0] == 1370


def test_determine_memory_profiling_no_cudagraph_estimate(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "gcu")
    monkeypatch.setattr(platform, "vendor_name", "enflame")
    monkeypatch.setattr(platform, "is_cuda_alike", lambda: True)
    cfg = make_config()
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    monkeypatch.setattr(worker.envs, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
                        False)
    monkeypatch.setattr(worker, "maybe_apply_startup_plan", mock.Mock())
    reserve = mock.Mock(side_effect=lambda b, *a, **k: b)
    monkeypatch.setattr(worker, "reserve_mm_ipc_gpu_memory", reserve)
    profile_result = SimpleNamespace(
        total_consumed=10, transient_peak_headroom=1,
        non_kv_cache_memory=100,
        after_profile=SimpleNamespace(free_memory=1900))
    _patch_profiling(monkeypatch, profile_result)
    w = make_worker(cfg)
    w.init_snapshot = SimpleNamespace(free_memory=2000, total_memory=3000)
    w.requested_memory = 1500
    w.model_runner.model_memory_usage = 100
    w.model_runner.profile_run = mock.Mock()
    w.model_runner.profile_cudagraph_memory = mock.Mock(return_value=30)
    assert w.determine_available_memory() == 1400


def test_determine_memory_musa_logs(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "musa")
    monkeypatch.setattr(platform, "vendor_name", "enflame")
    monkeypatch.setattr(platform, "is_cuda_alike", lambda: False)
    platform.torch_device_fn = SimpleNamespace(
        mem_get_info=mock.Mock(return_value=(100, 200)))
    monkeypatch.setattr(worker, "maybe_apply_startup_plan", mock.Mock())
    monkeypatch.setattr(worker, "reserve_mm_ipc_gpu_memory",
                        mock.Mock(side_effect=lambda b, *a, **k: b))
    profile_result = SimpleNamespace(
        total_consumed=10, transient_peak_headroom=1,
        non_kv_cache_memory=100,
        after_profile=SimpleNamespace(free_memory=1900))
    _patch_profiling(monkeypatch, profile_result)
    w = make_worker(make_config())
    w.init_snapshot = SimpleNamespace(free_memory=2000, total_memory=3000)
    w.requested_memory = 1500
    w.model_runner.model_memory_usage = 100
    w.model_runner.profile_run = mock.Mock()
    assert w.determine_available_memory() == 1400


# --------------------------------------------------------------------------
# Small delegation methods
# --------------------------------------------------------------------------
def test_get_kv_connector_no_group(monkeypatch):
    monkeypatch.setattr(worker, "has_kv_transfer_group", lambda: False)
    w = make_worker()
    assert w.get_kv_connector_handshake_metadata() is None


def test_get_kv_connector_no_metadata(monkeypatch):
    monkeypatch.setattr(worker, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(worker, "get_kv_transfer_group",
                        lambda: SimpleNamespace(
                            get_handshake_metadata=lambda: None))
    w = make_worker()
    assert w.get_kv_connector_handshake_metadata() is None


def test_get_kv_connector_metadata(monkeypatch):
    monkeypatch.setattr(worker, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(worker, "get_kv_transfer_group",
                        lambda: SimpleNamespace(
                            get_handshake_metadata=lambda: "meta"))
    monkeypatch.setattr(worker, "get_pp_group",
                        lambda: SimpleNamespace(rank_in_group=1))
    monkeypatch.setattr(worker, "get_tp_group",
                        lambda: SimpleNamespace(rank_in_group=2))
    w = make_worker()
    assert w.get_kv_connector_handshake_metadata() == {(1, 2): "meta"}


def test_get_kv_cache_spec_and_layouts():
    w = make_worker()
    w.model_runner.get_kv_cache_spec = mock.Mock(return_value={"kv": 1})
    assert w.get_kv_cache_spec() == {"kv": 1}
    assert w.get_supported_kv_cache_layouts() == []


def test_update_max_model_len():
    w = make_worker()
    w.model_runner.update_max_model_len = mock.Mock()
    w.update_max_model_len(4096)
    assert w.model_config.max_model_len == 4096
    w.model_runner.update_max_model_len.assert_called_once_with(4096)
    w.model_runner = None
    w.update_max_model_len(8192)
    assert w.model_config.max_model_len == 8192


def test_initialize_from_config(monkeypatch):
    monkeypatch.setattr(worker, "ensure_kv_transfer_initialized", mock.Mock())
    w = make_worker()
    w._maybe_get_memory_pool_context = mock.Mock(
        return_value=contextlib.nullcontext())
    w.model_runner.initialize_kv_cache = mock.Mock()
    w.model_runner.init_routed_experts_capturer = mock.Mock()
    w.model_runner._init_kv_zero_meta = mock.Mock()
    w.vllm_config.model_config.enable_return_routed_experts = True
    kvcfg = KVCacheConfig(num_blocks=5, needs_kv_cache_zeroing=True)
    w.initialize_from_config(kvcfg)
    assert w.cache_config.num_gpu_blocks == 5
    w.model_runner.initialize_kv_cache.assert_called_once_with(kvcfg)
    w.model_runner.init_routed_experts_capturer.assert_called_once()
    w.model_runner._init_kv_zero_meta.assert_called_once()


def test_reset_and_get_models():
    w = make_worker()
    w.model_runner.reset_mm_cache = mock.Mock()
    w.model_runner.reset_encoder_cache = mock.Mock()
    w.model_runner.get_model = mock.Mock(return_value="m")
    w.model_runner.get_draft_model = mock.Mock(return_value="d")
    w.reset_mm_cache()
    w.reset_encoder_cache()
    assert w.get_model() == "m"
    assert w.get_draft_model() == "d"
    w.model_runner.reset_mm_cache.assert_called_once()
    w.model_runner.reset_encoder_cache.assert_called_once()


def test_supports_draft_weight_updates_combinations():
    w = make_worker()
    w.weight_transfer_engine = None
    assert w.supports_draft_weight_updates() is False

    engine = SimpleNamespace(supports_draft_weight_update=True)
    w.weight_transfer_engine = engine
    w.vllm_config.speculative_config = SimpleNamespace(
        draft_model_config=object())
    w.model_runner.get_draft_model = mock.Mock(return_value=object())
    assert w.supports_draft_weight_updates() is True

    w.model_runner.get_draft_model = mock.Mock(return_value=None)
    assert w.supports_draft_weight_updates() is False

    w.model_runner.get_draft_model = mock.Mock(return_value=object())
    w.vllm_config.speculative_config = SimpleNamespace(draft_model_config=None)
    assert w.supports_draft_weight_updates() is False


def test_set_draft_weight_update_target_errors():
    w = make_worker()
    w.weight_transfer_engine = None
    with pytest.raises(AssertionError):
        w._set_draft_weight_update_target()

    engine = mock.Mock()
    w.weight_transfer_engine = engine
    w.model_runner.get_draft_model = mock.Mock(return_value=None)
    with pytest.raises(RuntimeError):
        w._set_draft_weight_update_target()

    w.model_runner.get_draft_model = mock.Mock(return_value="draft")
    w.vllm_config.speculative_config = None
    with pytest.raises(RuntimeError):
        w._set_draft_weight_update_target()

    cfg = SimpleNamespace(draft_model_config="cfg")
    w.vllm_config.speculative_config = cfg
    w._set_draft_weight_update_target()
    engine.set_weight_update_target.assert_called_once_with("draft", "cfg")


def test_get_supported_tasks_and_match_table_and_stats():
    w = make_worker()
    w.model_runner.get_supported_tasks = mock.Mock(return_value=("generate",))
    w.model_runner.get_encoder_timing_stats = mock.Mock(return_value={"a": 1})
    assert w.get_supported_tasks() == ("generate",)
    assert w.get_compilation_match_table() == {"table": 1}
    assert w.get_encoder_timing_stats() == {"a": 1}


# --------------------------------------------------------------------------
# compile_or_warm_up_model
# --------------------------------------------------------------------------
class _CompileRange:
    def __init__(self, lo, end):
        self.lo = lo
        self.end = end

    def __contains__(self, value):
        return self.lo <= value <= self.end


def _stub_compile(monkeypatch, pp_last=True):
    monkeypatch.setattr(worker, "kernel_warmup", mock.Mock())
    monkeypatch.setattr(worker, "freeze_gc_heap", mock.Mock())
    monkeypatch.setattr(worker, "maybe_attach_gc_debug_callback", mock.Mock())
    monkeypatch.setattr(worker, "enable_gpu_sync_check", mock.Mock())
    monkeypatch.setattr(worker, "set_random_seed", mock.Mock())
    monkeypatch.setattr(worker, "set_torch_threads_for_runtime", mock.Mock())
    monkeypatch.setattr(worker, "warmup_kernels", mock.Mock())
    monkeypatch.setattr(worker, "maybe_save_startup_plan", mock.Mock())
    monkeypatch.setattr(worker, "get_pp_group",
                        lambda: SimpleNamespace(is_last_rank=pp_last))
    jit = sys.modules["vllm.utils.jit_monitor"]
    activate = mock.Mock()
    monkeypatch.setattr(jit, "activate", activate)
    return activate


def _make_compile_runner(w):
    w.device = SimpleNamespace(type="cpu")
    mr = mock.Mock()
    mr._dummy_run.return_value = ("h", "l")
    mr.is_pooling_model = False
    mr.lora_config = object()
    mr.capture_model.return_value = 0
    w.model_runner = mr
    return mr


def test_compile_eager_v1_basic(monkeypatch):
    activate = _stub_compile(monkeypatch)
    w = make_worker()
    mr = _make_compile_runner(w)
    out = w.compile_or_warm_up_model()
    assert out.language_model == 0.0
    mr._dummy_sampler_run.assert_called_once()
    activate.assert_called_once()
    worker.set_torch_threads_for_runtime.assert_called_once()


def test_compile_vllm_compile_sizes_and_ranges(monkeypatch):
    _stub_compile(monkeypatch)
    cfg = make_config()
    cfg.compilation_config.mode = CompilationMode.VLLM_COMPILE
    cfg.compilation_config.compile_sizes = [2, 4]
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    cfg.compilation_config.cudagraph_capture_sizes = [4]
    cfg.compilation_config.get_compile_ranges = lambda: [_CompileRange(5, 10)]
    cfg.model_config.enforce_eager = False
    w = make_worker(cfg)
    mr = _make_compile_runner(w)
    w.total_consumed = 1
    w.peak_activation_memory = 2
    w.available_kv_cache_memory_bytes = 3
    w.init_snapshot = SimpleNamespace(free_memory=100, total_memory=200)
    w.requested_memory = 50
    w.compile_or_warm_up_model()
    sizes = [c.args[0] for c in mr._dummy_run.call_args_list if c.args]
    assert 10 in sizes
    assert 2 in sizes
    mr.capture_model.assert_called_once()
    worker.maybe_save_startup_plan.assert_called_once()


def test_compile_v2_warmup_kernels(monkeypatch):
    _stub_compile(monkeypatch)
    cfg = make_config(use_v2_model_runner=True)
    w = make_worker(cfg)
    w.use_v2_model_runner = True
    _make_compile_runner(w)
    w.compile_or_warm_up_model()
    worker.warmup_kernels.assert_called_once()


def test_compile_kernel_warmup_import_error(monkeypatch):
    _stub_compile(monkeypatch)
    monkeypatch.setattr(worker, "kernel_warmup",
                        mock.Mock(side_effect=ImportError("no torchvision")))
    w = make_worker()
    _make_compile_runner(w)
    w.compile_or_warm_up_model()


def test_compile_kunlunxin_skips_kernel_warmup(monkeypatch):
    _stub_compile(monkeypatch)
    monkeypatch.setattr(platform, "vendor_name", "kunlunxin")
    w = make_worker()
    _make_compile_runner(w)
    w.compile_or_warm_up_model()
    worker.kernel_warmup.assert_not_called()


def test_compile_pooling_model_dummy_pooler(monkeypatch):
    _stub_compile(monkeypatch)
    w = make_worker()
    mr = _make_compile_runner(w)
    mr.is_pooling_model = True
    w.compile_or_warm_up_model()
    mr._dummy_pooler_run.assert_called_once()


# --------------------------------------------------------------------------
# annotate_profile
# --------------------------------------------------------------------------
def test_annotate_profile_no_profiler():
    w = make_worker()
    w.profiler = None
    _assert_nullcontext(w.annotate_profile(SimpleNamespace()))


def test_annotate_profile_not_running():
    w = make_worker()
    w.profiler = SimpleNamespace(step=mock.Mock(), is_running=False)
    _assert_nullcontext(w.annotate_profile(SimpleNamespace()))
    w.profiler.step.assert_called_once()


def _scheduler_output_for_profile():
    cached = SimpleNamespace(
        req_ids=["b"], num_computed_tokens=[10],
        is_context_phase=lambda req_id: req_id == "b")
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="a", num_computed_tokens=0)],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={"a": 5, "b": 1})


def test_annotate_profile_detailed(monkeypatch):
    w = make_worker()
    w.vllm_config.profiler_config.detailed_trace_annotation = True
    annotate = mock.Mock(return_value="CTX")
    w.profiler = SimpleNamespace(step=mock.Mock(), is_running=True,
                                 annotate_context_manager=annotate)
    assert w.annotate_profile(_scheduler_output_for_profile()) == "CTX"
    text = annotate.call_args.args[0]
    assert text.startswith("execute_6_context_")


def test_annotate_profile_simple():
    w = make_worker()
    w.vllm_config.profiler_config.detailed_trace_annotation = False
    annotate = mock.Mock(return_value="CTX")
    w.profiler = SimpleNamespace(step=mock.Mock(), is_running=True,
                                 annotate_context_manager=annotate)
    assert w.annotate_profile(_scheduler_output_for_profile()) == "CTX"
    assert annotate.call_args.args[0].startswith("execute_context_")


# --------------------------------------------------------------------------
# sample_tokens / execute_model
# --------------------------------------------------------------------------
def test_worker_sample_tokens():
    w = make_worker()
    w.model_runner.sample_tokens = mock.Mock(return_value="S")
    assert w.sample_tokens(None) == "S"


def test_execute_model_returns_runner_output(monkeypatch):
    w = make_worker()
    out = ModelRunnerOutput()
    w.model_runner.execute_model = mock.Mock(return_value=out)
    sched = SimpleNamespace(total_num_scheduled_tokens=0)
    assert w.execute_model(sched) is out


def test_execute_model_waits_pp_send_work(monkeypatch):
    w = make_worker()
    h1, h2 = Handle(), Handle()
    w._pp_send_work = [h1, h2]
    out = ModelRunnerOutput()
    w.model_runner.execute_model = mock.Mock(return_value=out)
    sched = SimpleNamespace(total_num_scheduled_tokens=0)
    assert w.execute_model(sched) is out
    assert h1.waited and h2.waited
    assert w._pp_send_work == []


def test_execute_model_v2_pooling_none(monkeypatch):
    cfg = make_config(use_v2_model_runner=True)
    w = make_worker(cfg)
    w.use_v2_model_runner = True
    w.model_runner.is_pooling_model = True
    w.model_runner.execute_model = mock.Mock(return_value=None)
    pooled = ModelRunnerOutput()
    w.model_runner.pool = mock.Mock(return_value=pooled)
    sched = SimpleNamespace(total_num_scheduled_tokens=0)
    assert w.execute_model(sched) is pooled
    w.model_runner.pool.assert_called_once()


def test_execute_model_intermediate_tensors_pp_send(monkeypatch):
    w = make_worker()
    handle = Handle()
    post = mock.Mock()
    pp = SimpleNamespace(
        is_first_rank=False, is_last_rank=False,
        irecv_tensor_dict=mock.Mock(return_value=({"t": 1}, [handle], [post])),
        isend_tensor_dict=mock.Mock(return_value=[Handle()]))
    monkeypatch.setattr(worker, "get_pp_group", lambda: pp)
    monkeypatch.setattr(worker, "get_tp_group", lambda: SimpleNamespace())
    w.model_runner.execute_model = mock.Mock(
        return_value=IntermediateTensors({"x": 2}))
    sched = SimpleNamespace(total_num_scheduled_tokens=1)
    assert w.execute_model(sched) is None
    pp.irecv_tensor_dict.assert_called_once()
    pp.isend_tensor_dict.assert_called_once()
    assert len(w._pp_send_work) == 1


# --------------------------------------------------------------------------
# profile()
# --------------------------------------------------------------------------
def test_profile_not_configured(monkeypatch):
    w = make_worker()
    w.profiler_config = None
    with pytest.raises(RuntimeError):
        w.profile(True)
    w.profiler_config = SimpleNamespace(profiler=None)
    with pytest.raises(RuntimeError):
        w.profile(True)


def test_profile_torch_start(monkeypatch):
    w = make_worker()
    w.profiler_config = SimpleNamespace(profiler="torch")
    w.profiler = None
    wrapper = mock.Mock()
    monkeypatch.setattr(worker, "TorchProfilerWrapper",
                        mock.Mock(return_value=wrapper))
    monkeypatch.setattr(sys.modules["vllm.distributed.utils"],
                        "get_worker_rank_suffix", lambda **kw: "rank-0")
    w.profile(True, "prefix")
    wrapper.start.assert_called_once()
    assert w.profiler is wrapper


def test_profile_cuda_start(monkeypatch):
    w = make_worker()
    w.profiler_config = SimpleNamespace(profiler="cuda")
    w.profiler = None
    wrapper = mock.Mock()
    monkeypatch.setattr(worker, "CudaProfilerWrapper",
                        mock.Mock(return_value=wrapper))
    monkeypatch.setattr(sys.modules["vllm.distributed.utils"],
                        "get_worker_rank_suffix", lambda **kw: "r")
    w.profile(True)
    wrapper.start.assert_called_once()


def test_profile_proton_start(monkeypatch):
    w = make_worker()
    w.profiler_config = SimpleNamespace(profiler="proton")
    w.profiler = None
    wrapper = mock.Mock()
    monkeypatch.setattr(worker, "ProtonProfilerWrapper",
                        mock.Mock(return_value=wrapper))
    monkeypatch.setattr(sys.modules["vllm.distributed.utils"],
                        "get_worker_rank_suffix", lambda **kw: "r")
    w.profile(True)
    wrapper.start.assert_called_once()


def test_profile_invalid_type(monkeypatch):
    w = make_worker()
    w.profiler_config = SimpleNamespace(profiler="bogus")
    w.profiler = None
    monkeypatch.setattr(sys.modules["vllm.distributed.utils"],
                        "get_worker_rank_suffix", lambda **kw: "r")
    with pytest.raises(ValueError):
        w.profile(True)


def test_profile_stop_without_start():
    w = make_worker()
    w.profiler_config = SimpleNamespace(profiler="torch")
    w.profiler = None
    assert w.profile(False) is None


def test_profile_stop_proton_resets():
    w = make_worker()
    w.profiler_config = SimpleNamespace(profiler="proton")
    wrapper = mock.Mock()
    w.profiler = wrapper
    w.profile(False)
    wrapper.stop.assert_called_once()
    assert w.profiler is None


def test_profile_stop_torch_keeps_profiler():
    w = make_worker()
    w.profiler_config = SimpleNamespace(profiler="torch")
    wrapper = mock.Mock()
    w.profiler = wrapper
    w.profile(False)
    wrapper.stop.assert_called_once()
    assert w.profiler is wrapper


# --------------------------------------------------------------------------
# Misc delegations / savers
# --------------------------------------------------------------------------
def test_execute_dummy_batch():
    w = make_worker()
    del w.model_runner.uniform_decode_query_len
    w.model_runner.uniform_decode_query_len = 4
    w.execute_dummy_batch()
    w.model_runner._dummy_run.assert_called_once_with(4, uniform_decode=True)


def test_lora_and_health():
    w = make_worker()
    w.model_runner.add_lora = mock.Mock(return_value=True)
    w.model_runner.remove_lora = mock.Mock(return_value=True)
    w.model_runner.list_loras = mock.Mock(return_value={1})
    w.model_runner.pin_lora = mock.Mock(return_value=True)
    assert w.add_lora("lr") is True
    assert w.remove_lora(1) is True
    assert w.list_loras() == {1}
    assert w.pin_lora(1) is True
    assert w.check_health() is None


def test_save_sharded_state(monkeypatch):
    w = make_worker()
    w.model_runner.model = "model"
    loader = sys.modules["vllm.model_executor.model_loader"]
    save = mock.Mock()
    monkeypatch.setattr(loader, "ShardedStateLoader",
                        SimpleNamespace(save_model=save))
    w.save_sharded_state("path", pattern="p", max_size=1)
    save.assert_called_once_with("model", "path", pattern="p", max_size=1)


def test_save_tensorized_model(monkeypatch):
    w = make_worker()
    w.get_model = mock.Mock(return_value="model")
    save = mock.Mock()
    monkeypatch.setattr(worker.TensorizerLoader, "save_model", save)
    w.save_tensorized_model("cfg")
    save.assert_called_once_with("model", tensorizer_config="cfg",
                                 model_config=w.model_config)


def test_elastic_ep_execute():
    w = make_worker()
    w.elastic_ep_executor = SimpleNamespace(execute=mock.Mock(return_value="E"))
    assert w.elastic_ep_execute("method", 1) == "E"
    w.elastic_ep_executor.execute.assert_called_once_with("method", 1)


# --------------------------------------------------------------------------
# Weight transfer
# --------------------------------------------------------------------------
def test_check_weight_transfer_engine_none():
    w = make_worker()
    with pytest.raises(RuntimeError):
        w._check_weight_transfer_engine()


def test_init_weight_transfer_engine():
    w = make_worker()
    engine = SimpleNamespace(parse_init_info=mock.Mock(return_value="T"),
                             init_transfer_engine=mock.Mock())
    w.weight_transfer_engine = engine
    w.init_weight_transfer_engine({"k": 1})
    engine.init_transfer_engine.assert_called_once_with("T")


def test_init_weight_transfer_engine_none():
    w = make_worker()
    with pytest.raises(RuntimeError):
        w.init_weight_transfer_engine({})


def _wt_engine(draft_support=True):
    return SimpleNamespace(
        supports_draft_weight_update=draft_support,
        start_weight_update=mock.Mock(),
        update_weights=mock.Mock(),
        finish_weight_update=mock.Mock(),
        reset_weight_update_target=mock.Mock(),
        set_weight_update_target=mock.Mock())


def test_start_weight_update(monkeypatch):
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    w = make_worker()
    engine = _wt_engine()
    w.weight_transfer_engine = engine
    w.start_weight_update()
    engine.start_weight_update.assert_called_once()
    assert w._weight_update_active is True
    assert w._weight_update_is_draft is False


def test_start_draft_weight_update_unsupported(monkeypatch):
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    w = make_worker()
    w.weight_transfer_engine = _wt_engine(draft_support=False)
    with pytest.raises(RuntimeError):
        w.start_draft_weight_update()


def test_start_weight_update_already_active(monkeypatch):
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    w = make_worker()
    w.weight_transfer_engine = _wt_engine()
    w._weight_update_active = True
    with pytest.raises(RuntimeError):
        w.start_weight_update()


def test_start_weight_update_failure_resets(monkeypatch):
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    w = make_worker()
    engine = _wt_engine()
    engine.start_weight_update = mock.Mock(side_effect=RuntimeError("boom"))
    w.weight_transfer_engine = engine
    with pytest.raises(RuntimeError):
        w.start_weight_update()
    engine.reset_weight_update_target.assert_called_once()
    assert w._weight_update_active is False


def test_update_weights_requires_start(monkeypatch):
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    w = make_worker()
    w.weight_transfer_engine = _wt_engine()
    with pytest.raises(RuntimeError):
        w.update_weights({})


def test_update_weights_success(monkeypatch):
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    w = make_worker()
    engine = _wt_engine()
    w.weight_transfer_engine = engine
    w._weight_update_active = True
    w.update_weights({"chunk": 1})
    engine.update_weights.assert_called_once_with({"chunk": 1})


def test_update_weights_failure_resets(monkeypatch):
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    w = make_worker()
    engine = _wt_engine()
    engine.update_weights = mock.Mock(side_effect=RuntimeError("bad"))
    w.weight_transfer_engine = engine
    w._weight_update_active = True
    with pytest.raises(RuntimeError):
        w.update_weights({})
    assert w._weight_update_active is False
    engine.reset_weight_update_target.assert_called_once()


def test_finish_weight_update_requires_start(monkeypatch):
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    w = make_worker()
    w.weight_transfer_engine = _wt_engine()
    with pytest.raises(RuntimeError):
        w.finish_weight_update()


def test_finish_weight_update_resets_lora(monkeypatch):
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    w = make_worker()
    engine = _wt_engine()
    w.weight_transfer_engine = engine
    w._weight_update_active = True
    w._weight_update_is_draft = False
    w.finish_weight_update()
    engine.finish_weight_update.assert_called_once()
    w.model_runner.reset_lora_state.assert_called_once()
    assert w._weight_update_active is False


def test_finish_draft_weight_update_skips_lora(monkeypatch):
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    w = make_worker()
    w.weight_transfer_engine = _wt_engine()
    w._weight_update_active = True
    w._weight_update_is_draft = True
    w.finish_weight_update()
    w.model_runner.reset_lora_state.assert_not_called()


# --------------------------------------------------------------------------
# shutdown
# --------------------------------------------------------------------------
def test_shutdown_releases_resources(monkeypatch):
    monkeypatch.setattr(worker.gc, "unfreeze", mock.Mock())
    kv_shutdown = mock.Mock()
    ec_shutdown = mock.Mock()
    monkeypatch.setattr(worker, "ensure_kv_transfer_shutdown", kv_shutdown)
    monkeypatch.setattr(worker, "ensure_ec_transfer_shutdown", ec_shutdown)
    monkeypatch.setattr(platform, "is_cuda_alike", lambda: False)
    w = make_worker()
    w.profiler = mock.Mock()
    w.weight_transfer_engine = mock.Mock()
    w.model_runner = mock.Mock()
    w.shutdown()
    kv_shutdown.assert_called_once()
    ec_shutdown.assert_called_once()
    w.profiler.shutdown.assert_called_once()
    w.weight_transfer_engine.shutdown.assert_called_once()
    w.model_runner.shutdown.assert_called_once()


def test_shutdown_releases_cumem_pools(monkeypatch):
    monkeypatch.setattr(worker.gc, "unfreeze", mock.Mock())
    monkeypatch.setattr(worker, "ensure_kv_transfer_shutdown", mock.Mock())
    monkeypatch.setattr(worker, "ensure_ec_transfer_shutdown", mock.Mock())
    monkeypatch.setattr(platform, "is_cuda_alike", lambda: True)
    instance = mock.Mock()
    cumem = sys.modules["vllm.device_allocator.cumem"]
    monkeypatch.setattr(cumem, "CuMemAllocator",
                        SimpleNamespace(instance=instance))
    w = make_worker()
    w.profiler = None
    w.weight_transfer_engine = None
    w.model_runner = mock.Mock()
    w.shutdown()
    instance.release_pools.assert_called_once()


# --------------------------------------------------------------------------
# init_worker_distributed_environment
# --------------------------------------------------------------------------
def _stub_distributed_env(monkeypatch):
    init = mock.Mock()
    mp = mock.Mock()
    car = mock.Mock()
    eplb = mock.Mock()
    ec = mock.Mock()
    monkeypatch.setattr(worker, "init_distributed_environment", init)
    monkeypatch.setattr(worker, "ensure_model_parallel_initialized", mp)
    monkeypatch.setattr(worker, "set_custom_all_reduce", car)
    monkeypatch.setattr(worker, "override_envs_for_eplb", eplb)
    monkeypatch.setattr(worker, "ensure_ec_transfer_initialized", ec)
    bib = sys.modules["vllm.model_executor.determinism.batch_invariant"]
    monkeypatch.setattr(bib, "init_batch_invariance", mock.Mock())
    return init, mp, car, eplb, ec


def test_init_worker_distributed_environment_defaults(monkeypatch):
    init, mp, car, eplb, ec = _stub_distributed_env(monkeypatch)
    monkeypatch.setattr(platform, "device_type", "gcu")
    cfg = make_config()
    worker.init_worker_distributed_environment(
        cfg, rank=1, distributed_init_method=None, local_rank=0,
        backend="flagcx")
    init.assert_called_once_with(1, 1, "env://", 0, "flagcx", None)
    mp.assert_called_once_with(1, 1, 1, 1)
    ec.assert_called_once_with(cfg)


def test_init_worker_distributed_environment_timeout_and_backend(monkeypatch):
    init, mp, car, eplb, ec = _stub_distributed_env(monkeypatch)
    monkeypatch.setattr(platform, "device_type", "ptpu")
    cfg = make_config()
    cfg.parallel_config.distributed_timeout_seconds = 5
    worker.init_worker_distributed_environment(
        cfg, rank=0, distributed_init_method="tcp://x", local_rank=0,
        backend="flagcx")
    args = init.call_args.args
    assert args[2] == "tcp://x"
    assert args[4] == "pccl"
    assert args[5].total_seconds() == 5


# --------------------------------------------------------------------------
# GCU worker
# --------------------------------------------------------------------------
def test_gcu_worker_init(monkeypatch):
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    init_runtime = mock.Mock()
    monkeypatch.setattr(gcu_worker, "_initialize_fl_runtime", init_runtime)
    w = WorkerGCU(make_config(), 0, 3, "env://")
    assert w.rank == 3
    init_runtime.assert_called_once_with(3)
    assert os.environ["TRITON_CACHE_DIR"].endswith("rank-3")


def test_gcu_worker_init_spec_scheduler(monkeypatch):
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.setattr(gcu_worker, "_initialize_fl_runtime", mock.Mock())
    cfg = make_config(num_speculative_tokens=1)
    cfg.scheduler_config.async_scheduling = True
    w = WorkerGCU(cfg, 0, 0, "env://")
    assert w.vllm_config.scheduler_config.scheduler_cls == (
        "vllm_fl.worker.scheduler_fl.AsyncSchedulerFL")


def test_gcu_make_model_runner_v1(monkeypatch):
    sentinel = object()
    mr_mod = sys.modules["vllm_fl.worker.model_runner"]
    factory = mock.Mock(return_value=sentinel)
    monkeypatch.setattr(mr_mod, "ModelRunnerFL", factory)
    w = make_gcu_worker()
    w.use_v2_model_runner = False
    assert w._make_model_runner() is sentinel
    factory.assert_called_once()


def test_gcu_make_model_runner_v2(monkeypatch):
    sentinel = object()
    mod = sys.modules[
        "vllm_fl.dispatch.backends.vendor.gcu.worker.model_runner_v2"]
    factory = mock.Mock(return_value=sentinel)
    monkeypatch.setattr(mod, "GPUModelRunnerFLV2", factory)
    w = make_gcu_worker()
    w.use_v2_model_runner = True
    assert w._make_model_runner() is sentinel


def test_gcu_memory_pool_context_no_sleep(monkeypatch):
    monkeypatch.setattr(platform, "is_sleep_mode_available", lambda: False)
    w = make_gcu_worker()
    _assert_nullcontext(w._maybe_get_memory_pool_context("weights"))


def test_gcu_memory_pool_context_allocator(monkeypatch):
    monkeypatch.setattr(platform, "is_sleep_mode_available", lambda: True)
    sentinel = object()
    alloc = mock.Mock()
    alloc.get_current_usage.return_value = 0
    alloc.use_memory_pool.return_value = sentinel
    monkeypatch.setattr(gcu_worker, "get_mem_allocator_instance", lambda: alloc)
    w = make_gcu_worker()
    assert w._maybe_get_memory_pool_context("weights") is sentinel
    assert w._maybe_get_memory_pool_context("kv_cache") is sentinel


def test_gcu_scoped_allocator_max_split_is_nullcontext():
    w = make_gcu_worker()
    _assert_nullcontext(w._scoped_allocator_max_split(20))


def test_gcu_init_device_unsupported_type(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "cpu")
    cfg = make_config()
    cfg.device_config.device_type = "cuda"
    w = make_gcu_worker(cfg)
    with pytest.raises(RuntimeError):
        w.init_device()


def test_gcu_init_device_success(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "cpu")
    monkeypatch.setattr(platform, "logical_device_id_to_visible_device_id",
                        lambda rank: rank)
    monkeypatch.setattr(gcu_worker, "init_worker_distributed_environment",
                        mock.Mock())
    monkeypatch.setattr(gcu_worker, "warn_if_flashinfer_kernels_missing",
                        mock.Mock())
    monkeypatch.setattr(gcu_worker, "set_random_seed", mock.Mock())
    monkeypatch.setattr(gcu_worker, "MemorySnapshot",
                        lambda device=None: SimpleNamespace(
                            free_memory=2000, total_memory=3000))
    monkeypatch.setattr(gcu_worker, "request_memory",
                        mock.Mock(return_value=1500))
    monkeypatch.setattr(gcu_worker, "get_model_loader",
                        mock.Mock(return_value=SimpleNamespace(
                            get_external_weight_memory=mock.Mock(
                                return_value=0))))
    fake_accelerator(monkeypatch)
    cfg = make_config()
    cfg.device_config.device_type = "cpu"
    w = make_gcu_worker(cfg)
    w._init_workspace_and_model_runner = mock.Mock()
    w.init_device()
    assert w.init_snapshot.free_memory == 2000
    assert w.requested_memory == 1500
    w._init_workspace_and_model_runner.assert_called_once()


def test_gcu_init_device_dp_offset_and_assigned(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "cpu")
    monkeypatch.setattr(platform, "logical_device_id_to_visible_device_id",
                        lambda rank: rank)
    monkeypatch.setattr(gcu_worker, "init_worker_distributed_environment",
                        mock.Mock())
    monkeypatch.setattr(gcu_worker, "warn_if_flashinfer_kernels_missing",
                        mock.Mock())
    monkeypatch.setattr(gcu_worker, "set_random_seed", mock.Mock())
    monkeypatch.setattr(gcu_worker, "MemorySnapshot",
                        lambda device=None: SimpleNamespace(
                            free_memory=2000, total_memory=3000))
    monkeypatch.setattr(gcu_worker, "request_memory",
                        mock.Mock(return_value=1500))
    monkeypatch.setattr(gcu_worker, "get_model_loader",
                        mock.Mock(return_value=SimpleNamespace(
                            get_external_weight_memory=mock.Mock(
                                return_value=0))))
    iface = sys.modules["vllm.platforms.interface"]
    set_assigned = mock.Mock()
    monkeypatch.setattr(iface, "set_assigned_physical_gpu_ids", set_assigned)
    fake_accelerator(monkeypatch, device_count=10)
    cfg = make_config()
    cfg.device_config.device_type = "cpu"
    cfg.parallel_config.assigned_physical_gpu_ids = [0, 1, 2, 3]
    cfg.parallel_config.local_world_size = 4
    cfg.parallel_config.data_parallel_index = 1
    w = make_gcu_worker(cfg)
    w._init_workspace_and_model_runner = mock.Mock()
    w.init_device()
    set_assigned.assert_called_once_with([0, 1, 2, 3])


def _stub_gcu_compile(monkeypatch, pp_last=True):
    monkeypatch.setattr(gcu_worker, "kernel_warmup", mock.Mock())
    monkeypatch.setattr(gcu_worker, "freeze_gc_heap", mock.Mock())
    monkeypatch.setattr(gcu_worker, "maybe_attach_gc_debug_callback", mock.Mock())
    monkeypatch.setattr(gcu_worker, "enable_gpu_sync_check", mock.Mock())
    monkeypatch.setattr(gcu_worker, "set_random_seed", mock.Mock())
    monkeypatch.setattr(gcu_worker, "warmup_kernels", mock.Mock())
    monkeypatch.setattr(gcu_worker, "maybe_save_startup_plan", mock.Mock())
    monkeypatch.setattr(gcu_worker, "get_pp_group",
                        lambda: SimpleNamespace(is_last_rank=pp_last))


def _make_gcu_compile_runner(w):
    w.device = SimpleNamespace(type="cpu")
    w.randomize_dummy_inputs = False
    mr = mock.Mock()
    mr._dummy_run.return_value = ("h", "l")
    mr.is_pooling_model = False
    mr.lora_config = object()
    mr.capture_model.return_value = 0
    w.model_runner = mr
    w._maybe_activate_jit_monitor = mock.Mock()
    w._get_cudagraph_capture_context = mock.Mock(
        return_value=contextlib.nullcontext())
    return mr


def test_gcu_compile_basic(monkeypatch):
    _stub_gcu_compile(monkeypatch)
    w = make_gcu_worker()
    w.use_v2_model_runner = False
    mr = _make_gcu_compile_runner(w)
    out = w.compile_or_warm_up_model()
    assert out.language_model == 0.0
    mr._dummy_sampler_run.assert_called_once()
    w._maybe_activate_jit_monitor.assert_called_once()


def test_gcu_compile_compile_mode(monkeypatch):
    _stub_gcu_compile(monkeypatch)
    monkeypatch.setattr(gcu_worker, "has_compiled_submodule",
                        mock.Mock(return_value=True))
    cfg = make_config()
    cfg.compilation_config.mode = CompilationMode.VLLM_COMPILE
    cfg.compilation_config.compile_sizes = [2, 4]
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    cfg.compilation_config.cudagraph_capture_sizes = [4]
    cfg.compilation_config.get_compile_ranges = lambda: []
    cfg.model_config.enforce_eager = False
    w = make_gcu_worker(cfg)
    w.use_v2_model_runner = False
    mr = _make_gcu_compile_runner(w)
    w.compile_or_warm_up_model()
    mr.capture_model.assert_called_once()
    sizes = [c.args[0] for c in mr._dummy_run.call_args_list if c.args]
    assert 2 in sizes


def test_gcu_compile_kernel_warmup_import_error(monkeypatch):
    _stub_gcu_compile(monkeypatch)
    monkeypatch.setattr(gcu_worker, "kernel_warmup",
                        mock.Mock(side_effect=ImportError("missing")))
    w = make_gcu_worker()
    w.use_v2_model_runner = False
    _make_gcu_compile_runner(w)
    w.compile_or_warm_up_model()


def test_gcu_determine_available_memory_delegates(monkeypatch):
    monkeypatch.setattr(GPUWorker, "determine_available_memory",
                        mock.Mock(return_value="M"))
    w = make_gcu_worker()
    assert w.determine_available_memory() == "M"


def test_gcu_sample_tokens_delegates(monkeypatch):
    monkeypatch.setattr(GPUWorker, "sample_tokens",
                        lambda self, grammar_output: "S")
    w = make_gcu_worker()
    assert w.sample_tokens(None) == "S"


def test_gcu_execute_model_delegates(monkeypatch):
    monkeypatch.setattr(GPUWorker, "execute_model",
                        lambda self, scheduler_output: "E")
    w = make_gcu_worker()
    assert w.execute_model(object()) == "E"


# --------------------------------------------------------------------------
# Extra branch coverage
# --------------------------------------------------------------------------
def test_scheduler_current_row_fill_failure():
    base = FakeBaseManager(bitmask=FakeBitmask(8))
    mgr = scheduler_fl._MTPStructuredOutputManager(base)
    requests = make_requests([FakeGrammar(matcher=False), FakeGrammar()])
    out = mgr.grammar_bitmask(requests, ["r0", "r1"], {"r0": [5]})
    assert out == "FALLBACK"
    assert base.fallback_calls


def test_async_intermediate_wait_for_comm_idempotent():
    tensors = AsyncIntermediateTensors({"x": 1})
    tensors.wait_for_comm()
    tensors.wait_for_comm()


def test_sleep_rocm_retries_until_freed(monkeypatch):
    monkeypatch.setattr(platform, "is_rocm", lambda: True)
    fake_accelerator(monkeypatch, mem_infos=[(100, 200), (50, 200), (150, 200)])
    monkeypatch.setattr(worker.time, "sleep", mock.Mock())
    w = make_worker()
    w._get_sleep_mode_backend = mock.Mock(return_value=mock.Mock())
    w.model_runner = SimpleNamespace(model=FakeModel({}))
    w.sleep(1)
    worker.time.sleep.assert_called_once_with(0.1)


def _patch_torch_device(monkeypatch):
    monkeypatch.setattr(worker.torch, "device",
                        lambda spec: SimpleNamespace(type=str(spec).split(":")[0]),
                        raising=False)


def test_init_device_npu_branch(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "npu")
    monkeypatch.setattr(platform, "vendor_name", "ascend")
    _patch_torch_device(monkeypatch)
    cfg = make_config()
    cfg.device_config.device_type = "npu"
    _stub_init_device(monkeypatch)
    triton = mock.Mock()
    sys.modules["vllm_fl.dispatch.backends.vendor.ascend.impl.triton_utils"] \
        .init_device_properties_triton = triton
    mr_mod = sys.modules["vllm_fl.worker.model_runner"]
    monkeypatch.setattr(mr_mod, "ModelRunnerFL", mock.Mock(return_value=object()))
    w = make_worker(cfg)
    w.init_device()
    triton.assert_called_once()


def _v2_init_cfg(monkeypatch, vendor, device_type="cpu"):
    monkeypatch.setattr(platform, "device_type", device_type)
    monkeypatch.setattr(platform, "vendor_name", vendor)
    cfg = make_config(use_v2_model_runner=True)
    cfg.device_config.device_type = device_type
    _stub_init_device(monkeypatch)
    return cfg


def test_init_device_v2_npu(monkeypatch):
    cfg = _v2_init_cfg(monkeypatch, "ascend", "npu")
    _patch_torch_device(monkeypatch)
    sentinel = object()
    mod = sys.modules[
        "vllm_fl.dispatch.backends.vendor.ascend.worker.model_runner_v2"]
    factory = mock.Mock(return_value=sentinel)
    monkeypatch.setattr(mod, "GPUModelRunnerFLV2", factory)
    shim = mock.Mock()
    sys.modules[
        "vllm_fl.dispatch.backends.vendor.ascend.patches.ascend_v2_device"
    ].install_ascend_v2_device_shims = shim
    w = make_worker(cfg)
    w.init_device()
    factory.assert_called_once()
    shim.assert_called_once()
    assert w.model_runner is sentinel


def test_init_device_v2_amd(monkeypatch):
    cfg = _v2_init_cfg(monkeypatch, "amd")
    install = mock.Mock()
    sys.modules["vllm_fl.dispatch.backends.vendor.amd.v2_runner_compat"] \
        .install_v2_runner_compat = install
    sentinel = object()
    factory = mock.Mock(return_value=sentinel)
    monkeypatch.setattr(sys.modules["vllm.v1.worker.gpu.model_runner"],
                        "GPUModelRunner", factory)
    w = make_worker(cfg)
    w.init_device()
    install.assert_called_once()
    assert w.model_runner is sentinel


def test_init_device_v2_thead(monkeypatch):
    cfg = _v2_init_cfg(monkeypatch, "thead")
    install = mock.Mock()
    sys.modules["vllm_fl.dispatch.backends.vendor.thead.patches.v2_runner_compat"] \
        .install_v2_runner_compat = install
    sentinel = object()
    factory = mock.Mock(return_value=sentinel)
    monkeypatch.setattr(sys.modules["vllm.v1.worker.gpu.model_runner"],
                        "GPUModelRunner", factory)
    w = make_worker(cfg)
    w.init_device()
    install.assert_called_once()
    assert w.model_runner is sentinel


def test_init_device_v2_iluvatar(monkeypatch):
    cfg = _v2_init_cfg(monkeypatch, "iluvatar")
    sentinel = object()
    mod = sys.modules[
        "vllm_fl.dispatch.backends.vendor.iluvatar.worker.model_runner_v2"]
    factory = mock.Mock(return_value=sentinel)
    monkeypatch.setattr(mod, "IluvatarGPUModelRunnerV2", factory)
    w = make_worker(cfg)
    w.init_device()
    factory.assert_called_once()
    assert w.model_runner is sentinel


def test_determine_memory_musa_budget_exhausted(monkeypatch):
    monkeypatch.setattr(platform, "device_type", "musa")
    monkeypatch.setattr(platform, "vendor_name", "enflame")
    platform.torch_device_fn = SimpleNamespace(
        mem_get_info=mock.Mock(return_value=(100, 200)))
    monkeypatch.setattr(worker, "maybe_apply_startup_plan", mock.Mock())
    monkeypatch.setattr(worker, "reserve_mm_ipc_gpu_memory",
                        mock.Mock(side_effect=lambda b, *a, **k: b))
    profile_result = SimpleNamespace(
        total_consumed=10, transient_peak_headroom=1,
        non_kv_cache_memory=500,
        after_profile=SimpleNamespace(free_memory=1900))
    _patch_profiling(monkeypatch, profile_result)
    w = make_worker(make_config())
    w.init_snapshot = SimpleNamespace(free_memory=2000, total_memory=3000)
    w.requested_memory = 100
    w.model_runner.model_memory_usage = 100
    w.model_runner.profile_run = mock.Mock()
    assert w.determine_available_memory() == -400


def test_compile_cudagraph_estimate_diff(monkeypatch):
    _stub_compile(monkeypatch)
    cfg = make_config()
    cfg.model_config.enforce_eager = False
    w = make_worker(cfg)
    mr = _make_compile_runner(w)
    w.cudagraph_memory_estimate = 5
    w.compile_or_warm_up_model()
    mr.capture_model.assert_called_once()


def test_compile_triggers_inductor(monkeypatch):
    _stub_compile(monkeypatch)
    trigger = mock.Mock()
    sys.modules["vllm.compilation.compiler_interface"] \
        .trigger_inductor_lazy_init = trigger
    cfg = make_config()
    cfg.compilation_config.mode = CompilationMode.VLLM_COMPILE
    cfg.compilation_config.backend = "inductor"
    w = make_worker(cfg)
    _make_compile_runner(w)
    w.compile_or_warm_up_model()
    trigger.assert_called_once()


def test_annotate_profile_detailed_generation(monkeypatch):
    w = make_worker()
    w.vllm_config.profiler_config.detailed_trace_annotation = True
    annotate = mock.Mock(return_value="CTX")
    w.profiler = SimpleNamespace(step=mock.Mock(), is_running=True,
                                 annotate_context_manager=annotate)
    sched = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["g"], num_computed_tokens=[10],
            is_context_phase=lambda req_id: False),
        num_scheduled_tokens={"g": 1})
    assert w.annotate_profile(sched) == "CTX"
    assert "_generation_" in annotate.call_args.args[0]


def test_execute_model_v2_pooling_after_intermediate(monkeypatch):
    cfg = make_config(use_v2_model_runner=True)
    w = make_worker(cfg)
    w.use_v2_model_runner = True
    w.model_runner.is_pooling_model = True
    w.model_runner.pool = mock.Mock()
    pp = SimpleNamespace(
        is_first_rank=False, is_last_rank=False,
        irecv_tensor_dict=mock.Mock(return_value=({}, [], [])),
        isend_tensor_dict=mock.Mock(return_value=[]))
    monkeypatch.setattr(worker, "get_pp_group", lambda: pp)
    monkeypatch.setattr(worker, "get_tp_group", lambda: SimpleNamespace())
    w.model_runner.execute_model = mock.Mock(
        return_value=IntermediateTensors({"x": 1}))
    assert w.execute_model(SimpleNamespace(total_num_scheduled_tokens=1)) is None
    w.model_runner.pool.assert_called_once()


def test_take_draft_token_ids():
    w = make_worker()
    w.model_runner.take_draft_token_ids = mock.Mock(return_value="D")
    assert w.take_draft_token_ids() == "D"


def test_start_draft_weight_update_success(monkeypatch):
    monkeypatch.setattr(worker, "set_current_vllm_config",
                        _set_current_vllm_config)
    w = make_worker()
    engine = _wt_engine(draft_support=True)
    w.weight_transfer_engine = engine
    w.model_runner.get_draft_model = mock.Mock(return_value="draft")
    w.vllm_config.speculative_config = SimpleNamespace(draft_model_config="cfg")
    w.start_draft_weight_update()
    engine.set_weight_update_target.assert_called_once_with("draft", "cfg")
    assert w._weight_update_is_draft is True


def _stub_gcu_full_init(monkeypatch, device_count=8):
    monkeypatch.setattr(platform, "device_type", "cpu")
    monkeypatch.setattr(platform, "logical_device_id_to_visible_device_id",
                        lambda rank: rank)
    monkeypatch.setattr(gcu_worker, "init_worker_distributed_environment",
                        mock.Mock())
    monkeypatch.setattr(gcu_worker, "warn_if_flashinfer_kernels_missing",
                        mock.Mock())
    monkeypatch.setattr(gcu_worker, "set_random_seed", mock.Mock())
    monkeypatch.setattr(gcu_worker, "MemorySnapshot",
                        lambda device=None: SimpleNamespace(
                            free_memory=2000, total_memory=3000))
    monkeypatch.setattr(gcu_worker, "request_memory",
                        mock.Mock(return_value=1500))
    monkeypatch.setattr(gcu_worker, "get_model_loader",
                        mock.Mock(return_value=SimpleNamespace(
                            get_external_weight_memory=mock.Mock(
                                return_value=0))))
    fake_accelerator(monkeypatch, device_count=device_count)


def test_gcu_init_device_dp_offset_no_assigned(monkeypatch):
    _stub_gcu_full_init(monkeypatch, device_count=10)
    cfg = make_config()
    cfg.device_config.device_type = "cpu"
    cfg.parallel_config.data_parallel_rank_local = None
    cfg.parallel_config.data_parallel_index = 1
    cfg.parallel_config.local_world_size = 2
    w = make_gcu_worker(cfg)
    w._init_workspace_and_model_runner = mock.Mock()
    w.init_device()
    assert w.local_rank == 2


def test_gcu_init_device_v2_logs(monkeypatch):
    _stub_gcu_full_init(monkeypatch)
    cfg = make_config(use_v2_model_runner=True)
    cfg.device_config.device_type = "cpu"
    w = make_gcu_worker(cfg)
    w._init_workspace_and_model_runner = mock.Mock()
    w.init_device()
    w._init_workspace_and_model_runner.assert_called_once()


def test_gcu_compile_range_appends_end(monkeypatch):
    _stub_gcu_compile(monkeypatch)
    monkeypatch.setattr(gcu_worker, "has_compiled_submodule",
                        mock.Mock(return_value=True))
    cfg = make_config()
    cfg.compilation_config.mode = CompilationMode.VLLM_COMPILE
    cfg.compilation_config.get_compile_ranges = lambda: [_CompileRange(5, 10)]
    w = make_gcu_worker(cfg)
    w.use_v2_model_runner = False
    mr = _make_gcu_compile_runner(w)
    w.compile_or_warm_up_model()
    sizes = [c.args[0] for c in mr._dummy_run.call_args_list if c.args]
    assert 10 in sizes


def test_gcu_compile_kv_suggestion_and_estimate(monkeypatch):
    _stub_gcu_compile(monkeypatch)
    cfg = make_config()
    cfg.model_config.enforce_eager = False
    w = make_gcu_worker(cfg)
    w.use_v2_model_runner = False
    _make_gcu_compile_runner(w)
    w.cudagraph_memory_estimate = 5
    w.total_consumed = 1
    w.peak_activation_memory = 2
    w.available_kv_cache_memory_bytes = 3
    w.init_snapshot = SimpleNamespace(free_memory=100, total_memory=200)
    w.requested_memory = 50
    w.compile_or_warm_up_model()
    gcu_worker.maybe_save_startup_plan.assert_called_once()


def test_gcu_compile_pooling_model(monkeypatch):
    _stub_gcu_compile(monkeypatch)
    w = make_gcu_worker()
    w.use_v2_model_runner = False
    mr = _make_gcu_compile_runner(w)
    mr.is_pooling_model = True
    w.compile_or_warm_up_model()
    mr._dummy_pooler_run.assert_called_once()


def test_gcu_compile_v2_jit_warmup(monkeypatch):
    _stub_gcu_compile(monkeypatch)
    cfg = make_config(use_v2_model_runner=True)
    cfg.kernel_config.enable_jit_warmup = True
    w = make_gcu_worker(cfg)
    w.use_v2_model_runner = True
    _make_gcu_compile_runner(w)
    w.compile_or_warm_up_model()
    gcu_worker.warmup_kernels.assert_called_once()


def test_gcu_compile_triggers_inductor(monkeypatch):
    _stub_gcu_compile(monkeypatch)
    trigger = mock.Mock()
    sys.modules["vllm.compilation.compiler_interface"] \
        .trigger_inductor_lazy_init = trigger
    cfg = make_config()
    cfg.compilation_config.mode = CompilationMode.VLLM_COMPILE
    cfg.compilation_config.backend = "inductor"
    w = make_gcu_worker(cfg)
    w.use_v2_model_runner = False
    _make_gcu_compile_runner(w)
    w.compile_or_warm_up_model()
    trigger.assert_called_once()


def test_worker_module_import_fallback_for_comm_buffer(monkeypatch):
    """Cover the vLLM>=0.28 compat path that defines a no-op comm buffer."""
    parallel_state = sys.modules["vllm.distributed.parallel_state"]
    monkeypatch.delattr(parallel_state, "prepare_communication_buffer_for_model")
    path = os.path.join(_SOURCE_ROOT, "vllm_fl", "worker", "worker.py")
    spec = importlib.util.spec_from_file_location(
        "vllm_fl.worker._worker_comm_fallback", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._prepare_comm_buffer(1, 2, x=3) is None







