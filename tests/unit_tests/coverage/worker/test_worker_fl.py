# Copyright (c) 2025 BAAI. All rights reserved.

"""Supplementary CPU tests for ``vllm_fl.worker.worker``.

``worker.py`` is the FL GPU worker.  It cannot be imported on CPU-only
machines because it pulls the real ``vllm`` package at import time, so these
tests install minimal *external-boundary* stubs (a fake ``vllm`` module tree
plus a stub ``WorkerBase``) in ``sys.modules`` and then re-import the real
module, following the pattern established in ``test_runtime.py``.  Only the
vLLM surface is faked; every line of the module under test runs for real:

* module-level helpers: ``_num_workspace_lanes``, the ``_prepare_comm_buffer``
  version-compat shim, ``_normalize_process_group_backend`` per platform,
  ``_is_hybrid_kv_model`` (mamba_cache_mode / is_hybrid / layer_types),
  ``_apply_worker_vendor_patches`` (iluvatar gate, success and failure);
* ``AsyncIntermediateTensors``: handle waits, postprocess callbacks, the
  ``tensors`` lazy-synchronization ``__getattribute__`` gate;
* ``WorkerFL.__init__`` (gcu triton cache dir, MTP=1 scheduler override,
  kunlunxin hybrid-KV env default, sentinel/weight-transfer state);
* worker methods driven through a recording model-runner double:
  sleep/wake buffer save/restore, checkpoint prepare/restore, weight-update
  state machine (start/update/finish, draft retargeting, error recovery),
  ``init_device`` rank arithmetic, lora/save/shutdown delegation, profiler
  start/stop, ``execute_model`` PP plumbing, ``annotate_profile`` annotations;
* ``init_worker_distributed_environment`` forwarding to the distributed
  boundary with the normalized backend and the EPLB env override.
"""

import builtins
import contextlib
import enum
import gc
import logging
import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

# ---------------------------------------------------------------------------
# Fake vllm module tree (external boundary)
# ---------------------------------------------------------------------------


class _CUDAGraphMode(enum.Enum):
    NONE = "none"
    FULL = "full"


class _CompilationMode(enum.Enum):
    NONE = "none"
    VLLM_COMPILE = "vllm_compile"


class _OnceLogger:
    """Logger wrapper providing the ``*_once`` helpers vLLM loggers have."""

    def __init__(self, name):
        self._log = logging.getLogger(name)

    def __getattr__(self, item):
        return getattr(self._log, item)

    def info_once(self, *args, **kwargs):
        self._log.info(*args, **kwargs)

    def warning_once(self, *args, **kwargs):
        self._log.warning(*args, **kwargs)

    def debug_once(self, *args, **kwargs):
        self._log.debug(*args, **kwargs)


class _StubIntermediateTensors:
    """Stub of vllm.sequence.IntermediateTensors (attribute carrier)."""

    def __init__(self, tensors):
        self.tensors = tensors


class _SleepModeBackendStub:
    """Recording double of the device sleep-mode backend."""

    def __init__(self):
        self.calls = []

    def suspend(self, level):
        self.calls.append(("suspend", level))

    def resume(self, tags):
        self.calls.append(("resume", tags))


class _RecordingEngine:
    """Recording double of a WeightTransferEngine."""

    def __init__(self, supports_draft=False):
        self.calls = []
        self.supports_draft_weight_update = supports_draft

    def parse_init_info(self, info):
        self.calls.append(("parse_init_info", (info,), {}))
        return ("typed", info)

    def _rec(self, name):
        def fn(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return fn

    def __getattr__(self, name):
        if name.startswith("_") or name in ("calls", "supports_draft_weight_update"):
            raise AttributeError(name)
        return self._rec(name)


class _WorkerBaseStub:
    """Stub of vllm WorkerBase: mirrors attribute wiring on real subclassing."""

    instances = []

    def __init__(
        self,
        vllm_config=None,
        local_rank=None,
        rank=None,
        distributed_init_method=None,
        is_driver_worker=None,
    ):
        self.vllm_config = vllm_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        cfg = vllm_config
        platform = sys.modules["vllm.platforms"].current_platform
        self.device_config = SimpleNamespace(
            device_type=platform.device_type, device=f"{platform.device_type}:0"
        )
        self.parallel_config = cfg.parallel_config
        self.model_config = cfg.model_config
        self.cache_config = cfg.cache_config
        self.scheduler_config = cfg.scheduler_config
        self.speculative_config = cfg.speculative_config
        self.compilation_config = cfg.compilation_config
        self.observability_config = SimpleNamespace(
            jit_monitor_mode="off", jit_monitor_verbose=False
        )
        self.kernel_config = cfg.kernel_config
        _WorkerBaseStub.instances.append(self)


class WorkerVllmTree:
    """Install/restore the fake ``vllm`` tree ``vllm_fl.worker.worker`` needs.

    Both the import-time surface (every ``vllm.*`` leaf the module imports)
    and the runtime surface the FL bootstrap (``_initialize_fl_runtime``)
    touches are provided.  ``calls`` records every stubbed-function call so
    tests have an independent execution oracle.
    """

    def __init__(self, vendor_name="nvidia", device_type="cuda"):
        self._installed = []
        self._previous = {}
        self.calls = []
        self.sleep_backend = None
        self.platform = SimpleNamespace(
            vendor_name=vendor_name,
            device_type=device_type,
            dist_backend="nccl",
            is_cuda=lambda: device_type == "cuda",
            is_rocm=lambda: False,
            is_cuda_alike=lambda: device_type in ("cuda", "hip"),
            is_cpu=lambda: device_type == "cpu",
            is_xpu=lambda: False,
            is_sleep_mode_available=lambda: False,
            logical_device_id_to_visible_device_id=lambda i: i,
            set_device=lambda d: None,
            check_if_supports_dtype=lambda d: None,
            is_out_of_tree=lambda: True,
        )
        self._build()

    # -- tree plumbing -----------------------------------------------------

    def _put(self, name, **attrs):
        parts = name.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                self._put(parent)
        module = ModuleType(name)
        # Mark as package so ``from x.y.leaf import name`` resolves through
        # sys.modules even when only the leaf was installed explicitly.
        module.__path__ = []
        for key, value in attrs.items():
            setattr(module, key, value)
        if name not in self._previous:
            self._previous[name] = sys.modules.get(name)
        sys.modules[name] = module
        self._installed.append(name)
        return module

    def rec(self, name, ret=None):
        """Stub function recording (name, args, kwargs) into ``self.calls``."""

        def fn(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return ret

        return fn

    # -- stub surface ------------------------------------------------------

    def _build(self):
        nullctx = contextlib.nullcontext
        self._put("vllm")
        # Leaves pulled in by _initialize_fl_runtime -> vllm_fl.ops.custom_ops
        # and vllm_fl.attention.utils (identical set to test_runtime.py).
        self._put("vllm.triton_utils", tl=object(), triton=object())
        self._put("vllm._aiter_ops", rocm_aiter_ops=object())
        self._put(
            "vllm.model_executor.custom_op",
            CustomOp=type("CustomOp", (), {"register_oot": lambda **k: None}),
            PluggableLayer=type(
                "PluggableLayer", (), {"register_oot": lambda **k: None}
            ),
        )
        self._put("vllm.model_executor.layers")
        self._put(
            "vllm.model_executor.layers.activation",
            SiluAndMul=type("SiluAndMul", (), {}),
            GeluAndMul=type("GeluAndMul", (), {}),
        )
        self._put(
            "vllm.model_executor.layers.layernorm",
            RMSNorm=type("RMSNorm", (), {}),
        )
        self._put(
            "vllm.model_executor.layers.rotary_embedding",
            RotaryEmbedding=type("RotaryEmbedding", (), {}),
        )
        self.mm_encoder = self._put(
            "vllm.model_executor.layers.attention.mm_encoder_attention",
            MMEncoderAttention=type(
                "MMEncoderAttention",
                (),
                {
                    "forward_native": lambda s, *a: None,
                    "forward_cuda": lambda s, *a: None,
                },
            ),
            maybe_get_vit_flash_attn_backend=lambda backend: None,
        )
        self._put("vllm.v1.attention.backends.registry")
        self._put(
            "vllm.v1.attention.backends.registry",
            AttentionBackendEnum=SimpleNamespace(
                FLASH_ATTN=object(), ROCM_AITER_FA=object(), TORCH_SDPA=object()
            ),
        )
        # The iluvatar module-level patch calls find_spec on this name; the
        # stub needs a __spec__ whose origin is a real readable file, or
        # find_spec raises (missing parent chain / None __spec__). Pointing
        # the origin at this test file is safe: the patch reads it, finds no
        # chained-or to rewrite, and skips — exactly the graceful branch.
        from importlib.util import spec_from_file_location

        _helpers = self._put("vllm.v1.attention.ops.triton_attention_helpers")
        _helpers.__spec__ = spec_from_file_location(
            "vllm.v1.attention.ops.triton_attention_helpers", __file__
        )
        self._put(
            "vllm.model_executor.layers.fused_moe",
            FusedMoEFactory=type("FusedMoEFactory", (), {}),
            FusedMoE=type("FusedMoE", (), {}),
        )
        self.fused_moe_layer = self._put(
            "vllm.model_executor.layers.fused_moe.layer",
            FusedMoEFactory=type("UpstreamFactory", (), {}),
            FusedMoE=type("UpstreamMoE", (), {}),
        )
        self._put("vllm.model_executor.layers.fused_moe.runner")
        self._put(
            "vllm.model_executor.layers.fused_moe.runner.moe_runner",
            MoERunner=type("MoERunner", (), {}),
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.modular_kernel",
            FusedMoEExperts=type("FusedMoEExperts", (), {}),
            FusedMoEActivationFormat=SimpleNamespace(
                BatchedExperts=object(), Standard=object()
            ),
            ExpertTokensMetadata=type("ExpertTokensMetadata", (), {}),
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.config",
            FusedMoEConfig=type("FusedMoEConfig", (), {}),
            FUSED_MOE_UNQUANTIZED_CONFIG={},
            FusedMoEQuantConfig=type("FusedMoEQuantConfig", (), {}),
            _get_config_dtype_str=lambda *a, **k: "fp16",
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method",
            UnquantizedFusedMoEMethod=type("UnquantizedFusedMoEMethod", (), {}),
            select_unquantized_moe_backend=object(),
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.activation",
            ApplyMoEActivationConfig=type("ApplyMoEActivationConfig", (), {}),
            MoEActivation=type("MoEActivation", (), {}),
            apply_moe_activation=lambda *a, **k: None,
        )
        self._put("vllm.model_executor.layers.fused_moe.experts")
        self._put(
            "vllm.model_executor.layers.fused_moe.experts.triton_moe",
            TritonExperts=type("TritonExperts", (), {}),
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe",
            rocm_aiter_grouped_topk=lambda *a, **k: None,
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.fused_moe",
            try_get_optimal_moe_config=lambda *a, **k: None,
            _get_config_quant_dtype=lambda *a, **k: None,
        )
        ume_backend = type(
            "UnquantizedMoeBackend",
            (),
            {
                "AITER": object(),
                "TRITON": object(),
                "BATCHED_TRITON": object(),
                "TPU": object(),
                "FLASHINFER_TRTLLM": object(),
                "FLASHINFER_CUTLASS": object(),
            },
        )
        self._put("vllm.model_executor.layers.fused_moe.oracle")
        self._put(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized",
            UnquantizedMoeBackend=ume_backend,
            _trtllm_bf16_lora_supported=lambda *a, **k: True,
            backend_to_kernel_cls={},
            map_unquantized_backend=lambda *a, **k: None,
            select_unquantized_moe_backend=object(),
        )
        self._put("vllm.model_executor.layers.fused_moe.router")
        self._put(
            "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router",
            FusedTopKBiasRouter=type("FusedTopKBiasRouter", (), {}),
            fused_topk_bias=lambda *a, **k: None,
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.router.fused_topk_router",
            FusedTopKRouter=type("FusedTopKRouter", (), {}),
            _get_padding_mask=lambda *a, **k: None,
            fused_topk=lambda *a, **k: None,
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.router.grouped_topk_router",
            GroupedTopKRouter=type("GroupedTopKRouter", (), {}),
        )
        self._put(
            "vllm.model_executor.layers.fused_moe.utils",
            _resize_cache=lambda *a, **k: None,
            moe_kernel_quantize_input=lambda *a, **k: None,
        )

        # Direct imports of worker.py.
        self._put(
            "vllm.envs",
            VLLM_FLOAT32_MATMUL_PRECISION="high",
            VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=True,
        )
        self._put(
            "vllm.config",
            CUDAGraphMode=_CUDAGraphMode,
            VllmConfig=object,
            set_current_vllm_config=lambda *a, **k: nullctx(),
        )
        self._put("vllm.config.compilation", CompilationMode=_CompilationMode)
        self._put("vllm.device_allocator", get_mem_allocator_instance=lambda: None)
        self.sleep_backend_factory = _SleepModeBackendStub()
        self._put(
            "vllm.device_allocator.sleep_mode_backend",
            SleepModeBackendFactory=SimpleNamespace(
                create_backend=lambda model_config: self.sleep_backend_factory
            ),
        )
        self._put(
            "vllm.distributed",
            ensure_model_parallel_initialized=self.rec(
                "ensure_model_parallel_initialized"
            ),
            init_distributed_environment=self.rec("init_distributed_environment"),
            set_custom_all_reduce=self.rec("set_custom_all_reduce"),
        )
        self._put(
            "vllm.distributed.ec_transfer",
            ensure_ec_transfer_initialized=self.rec("ensure_ec_transfer_initialized"),
            ensure_ec_transfer_shutdown=self.rec("ensure_ec_transfer_shutdown"),
        )
        self._put("vllm.distributed.eplb")
        self._put(
            "vllm.distributed.eplb.eplb_utils",
            override_envs_for_eplb=self.rec("override_envs_for_eplb"),
        )
        self._put("vllm.distributed.elastic_ep")
        self._put(
            "vllm.distributed.elastic_ep.elastic_execute",
            ElasticEPScalingExecutor=lambda worker=None: SimpleNamespace(
                execute=lambda m, *a, **k: ("elastic", m, a, k)
            ),
        )
        self._put(
            "vllm.distributed.kv_transfer",
            ensure_kv_transfer_initialized=self.rec("ensure_kv_transfer_initialized"),
            ensure_kv_transfer_shutdown=self.rec("ensure_kv_transfer_shutdown"),
            get_kv_transfer_group=self.rec("get_kv_transfer_group"),
            has_kv_transfer_group=lambda: False,
        )
        self._put(
            "vllm.distributed.kv_transfer.kv_connector.v1.base",
            KVConnectorHandshakeMetadata=object,
        )
        self._put(
            "vllm.distributed.parallel_state",
            Handle=object,
            checkpoint_prepare_distributed_state=self.rec("cp_prepare"),
            checkpoint_restore_distributed_state=self.rec("cp_restore"),
            get_pp_group=self.rec("get_pp_group"),
            get_tp_group=self.rec("get_tp_group"),
            prepare_communication_buffer_for_model=self.rec("_prepare_comm_buffer"),
        )
        self.factory_calls = []

        def _create_engine(cfg, vc, dev, model):
            self.factory_calls.append(("create_engine", cfg, dev, model))
            return _RecordingEngine()

        self._put(
            "vllm.distributed.weight_transfer",
            WeightTransferEngine=object,
            WeightTransferEngineFactory=SimpleNamespace(create_engine=_create_engine),
        )
        self._put("vllm.logger", init_logger=lambda name: _OnceLogger(name))
        self._put("vllm.lora.request", LoRARequest=object)
        self._put("vllm.model_executor")
        # Lazily imported by init_worker_distributed_environment.
        self._put(
            "vllm.model_executor.determinism",
            init_batch_invariance=self.rec("init_batch_invariance"),
        )
        self._put(
            "vllm.model_executor.determinism.batch_invariant",
            init_batch_invariance=sys.modules[
                "vllm.model_executor.determinism"
            ].init_batch_invariance,
        )
        self._put(
            "vllm.model_executor.model_loader",
            TensorizerLoader=SimpleNamespace(save_model=lambda *a, **k: None),
            ShardedStateLoader=SimpleNamespace(save_model=lambda *a, **k: None),
        )
        self._put(
            "vllm.model_executor.warmup.kernel_warmup",
            kernel_warmup=self.rec("kernel_warmup"),
        )
        self._put(
            "vllm.multimodal.gpu_ipc_memory",
            reserve_mm_ipc_gpu_memory=lambda n, mm, c: n,
        )
        self._put("vllm.platforms", current_platform=self.platform)
        self.platform_iface = self._put(
            "vllm.platforms.interface", set_assigned_physical_gpu_ids=lambda ids: None
        )
        self._put(
            "vllm.profiler.wrapper",
            CudaProfilerWrapper=object,
            ProtonProfilerWrapper=object,
            TorchProfilerWrapper=object,
        )
        self._put("vllm.sequence", IntermediateTensors=_StubIntermediateTensors)
        self._put("vllm.tasks", SupportedTask=object)
        self._put("vllm.tracing", instrument=lambda **k: lambda fn: fn)
        self._put(
            "vllm.utils.gc_utils",
            freeze_gc_heap=self.rec("freeze_gc_heap"),
            maybe_attach_gc_debug_callback=self.rec("maybe_attach_gc_debug_callback"),
        )
        self._put(
            "vllm.utils.gpu_sync_debug",
            enable_gpu_sync_check=self.rec("enable_gpu_sync_check"),
            with_gpu_sync_check=lambda fn: fn,
        )
        self._put("vllm.utils.mem_constants", GiB_bytes=1024**3)
        self._put(
            "vllm.utils.mem_utils",
            MemorySnapshot=object,
            format_gib=lambda b: f"{b}",
            memory_profiling=lambda *a, **k: nullctx(),
        )
        self._put(
            "vllm.utils.torch_utils",
            set_random_seed=self.rec("set_random_seed"),
            set_torch_threads_for_runtime=self.rec("set_torch_threads_for_runtime"),
        )
        self._put(
            "vllm.v1.core.sched.output", GrammarOutput=object, SchedulerOutput=object
        )
        self._put(
            "vllm.v1.kv_cache_interface", KVCacheConfig=object, KVCacheSpec=object
        )
        self._put(
            "vllm.v1.outputs",
            AsyncModelRunnerOutput=object,
            DraftTokenIds=object,
            ModelRunnerOutput=object,
        )
        self._put(
            "vllm.v1.utils",
            compute_iteration_details=lambda so: so,
            report_usage_stats=self.rec("report_usage_stats"),
        )
        self._put(
            "vllm.v1.worker.gpu.warmup", warmup_kernels=self.rec("warmup_kernels")
        )
        self._put(
            "vllm.v1.worker.sentinel.gpu_worker_sentinel",
            WorkerSentinel=lambda worker=None: SimpleNamespace(
                worker=worker,
                handle_command=lambda req: ("handled", req),
            ),
        )
        self._put(
            "vllm.v1.worker.startup_plan",
            maybe_apply_startup_plan=self.rec("maybe_apply_startup_plan"),
            maybe_save_startup_plan=self.rec("maybe_save_startup_plan"),
        )
        self._put("vllm.v1.worker.utils", request_memory=lambda snap, cc: 0)
        # CuMemAllocator leaf: shutdown imports it on cuda-like platforms.
        self._put(
            "vllm.device_allocator.cumem",
            CuMemAllocator=SimpleNamespace(instance=None),
        )

        class _CompilationTimes:
            def __init__(self, language_model, encoder):
                self.language_model = language_model
                self.encoder = encoder

        self._put(
            "vllm.v1.worker.worker_base",
            CompilationTimes=_CompilationTimes,
            WorkerBase=_WorkerBaseStub,
        )
        self._put(
            "vllm.v1.worker.workspace",
            init_workspace_manager=self.rec("init_workspace_manager"),
        )
        built = []
        self.gpu_runner_builds = built

        class _GPUModelRunnerV2:
            def __init__(self, vllm_config, device):
                built.append((vllm_config, device))

        self.gpu_model_runner_cls = _GPUModelRunnerV2
        self._put(
            "vllm.v1.worker.gpu.model_runner",
            GPUModelRunner=_GPUModelRunnerV2,
        )

    def restore(self):
        """Remove/restore every sys.modules entry this tree installed."""
        for name in self._installed:
            previous = self._previous[name]
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        self._installed.clear()
        self._previous.clear()


# ---------------------------------------------------------------------------
# Config / model-runner doubles
# ---------------------------------------------------------------------------


class _RunnerDouble:
    """Recording model-runner double.

    Unknown method calls are recorded and return a fresh SimpleNamespace so
    delegation-style worker methods run to completion; known attributes used
    by branching logic are declared explicitly.
    """

    def __init__(self, vllm_config=None, device=None):
        self.vllm_config = vllm_config
        self.device = device
        self.calls = []
        self.is_pooling_model = False
        self.lora_config = None
        self.model_memory_usage = 0
        self.uniform_decode_query_len = 3
        self.model = SimpleNamespace(model="model")
        self.draft_model = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def fn(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return SimpleNamespace()

        return fn

    def get_draft_model(self):
        return self.draft_model

    def get_model(self):
        return self.model

    def _dummy_run(self, *args, **kwargs):
        """Warmup dummy-run recording (underscore names bypass __getattr__)."""
        self.calls.append(("_dummy_run", args, kwargs))
        return SimpleNamespace(), SimpleNamespace()

    def _dummy_sampler_run(self, *args, **kwargs):
        """Sampler warmup recording."""
        self.calls.append(("_dummy_sampler_run", args, kwargs))

    def _dummy_pooler_run(self, *args, **kwargs):
        """Pooler warmup recording."""
        self.calls.append(("_dummy_pooler_run", args, kwargs))


def make_vllm_config(**overrides: Any) -> SimpleNamespace:
    """VllmConfig double covering every sub-config WorkerFL reads."""
    cfg: SimpleNamespace = SimpleNamespace(
        use_v2_model_runner=True,
        num_speculative_tokens=0,
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            scheduler_cls="vllm.v1.core.sched.scheduler.Scheduler",
            max_num_seqs=4,
            max_num_batched_tokens=8,
        ),
        parallel_config=SimpleNamespace(
            enable_dbo=False,
            enable_fault_tolerance=False,
            assigned_physical_gpu_ids=None,
            local_world_size=1,
            distributed_executor_backend="mp",
            data_parallel_backend="mp",
            nnodes_within_dp=1,
            data_parallel_rank_local=None,
            data_parallel_index=0,
            world_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            disable_custom_all_reduce=False,
            distributed_timeout_seconds=None,
        ),
        model_config=SimpleNamespace(
            dtype=torch.float16,
            seed=42,
            max_model_len=128,
            enforce_eager=False,
            enable_cumem_allocator=False,
            enable_sleep_mode=False,
            enable_return_routed_experts=False,
            multimodal_config=SimpleNamespace(),
        ),
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None,
            num_gpu_blocks=None,
            gpu_memory_utilization=0.9,
            block_size=16,
        ),
        speculative_config=None,
        profiler_config=None,
        weight_transfer_config=None,
        compilation_config=SimpleNamespace(
            mode=_CompilationMode.NONE,
            compile_sizes=None,
            cudagraph_mode=_CUDAGraphMode.NONE,
            cudagraph_capture_sizes=None,
            get_compile_ranges=lambda: [],
            backend="",
            compilation_time=1.5,
            encoder_compilation_time=0.5,
        ),
        kernel_config=SimpleNamespace(moe_backend="triton", enable_jit_warmup=True),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture
def worker_env(fl_sys_modules, monkeypatch):
    """Yield a factory installing a fresh worker-module environment.

    Each call re-imports ``vllm_fl.worker.worker`` against a newly installed
    stub tree so import-time bindings always reference this test's objects.
    The module under test itself is never stubbed.
    """
    trees = []
    monkeypatch.setenv("USE_FLAGGEMS", "0")
    # Keep the FL bootstrap's OOT registration from reaching vendor patch
    # chains (kunlunxin/ascend/...) that need far deeper vllm surfaces; the
    # bootstrap is an external boundary here, not the module under test.
    monkeypatch.setenv("VLLM_FL_OOT_ENABLED", "0")

    def _install(vendor_name="nvidia", device_type="cuda"):
        for name in list(sys.modules):
            if name == "vllm_fl.worker" or name.startswith("vllm_fl.worker."):
                sys.modules.pop(name, None)
        tree = WorkerVllmTree(vendor_name=vendor_name, device_type=device_type)
        trees.append(tree)
        import importlib

        import vllm_fl.worker.worker as worker_module

        return importlib.reload(worker_module), tree

    yield _install
    for tree in reversed(trees):
        tree.restore()


@pytest.fixture(autouse=True)
def _worker_test_env(monkeypatch):
    """Neutralize env vars steering the FL runtime bootstrap."""
    for key in (
        "TRITON_CACHE_DIR",
        "USE_RESHAPE_AND_CACHE_FLASH",
        "PYTORCH_CUDA_ALLOC_CONF",
        "VLLM_TXDA_KV_CACHE_SIZE",
        "NCCL_ASYNC_ERROR_HANDLING",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
        "VLLM_FL_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)
    yield


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestNumWorkspaceLanes:
    """``_num_workspace_lanes`` picks lane count from DSpark speculation."""

    def test_single_lane_without_dspark(self, worker_env):
        """Plain V2 runner and non-DSpark speculation keep one workspace lane."""
        worker, _ = worker_env()
        plain = make_vllm_config()
        assert worker._num_workspace_lanes(plain, use_v2_model_runner=True) == 1
        assert worker._num_workspace_lanes(plain, use_v2_model_runner=False) == 1

    def test_two_lanes_for_dspark_v2(self, worker_env):
        """V2 runner + spec_config.use_dspark() doubles the lanes."""
        worker, _ = worker_env()
        cfg = make_vllm_config(
            speculative_config=SimpleNamespace(use_dspark=lambda: True)
        )
        assert worker._num_workspace_lanes(cfg, use_v2_model_runner=True) == 2
        # V1 runner never gets the second lane even with dspark.
        assert worker._num_workspace_lanes(cfg, use_v2_model_runner=False) == 1

    def test_none_spec_config_is_single_lane(self, worker_env):
        """A missing speculative config degrades to one lane."""
        worker, _ = worker_env()
        cfg = make_vllm_config()
        cfg.speculative_config = None
        assert worker._num_workspace_lanes(cfg, use_v2_model_runner=True) == 1


class TestNormalizeProcessGroupBackend:
    """``_normalize_process_group_backend`` maps vendor backends."""

    @pytest.mark.parametrize(
        ("vendor_name", "device_type", "backend", "expected"),
        [
            ("nvidia", "cuda", "nccl", "nccl"),
            ("nvidia", "cuda", "flagcx", "flagcx"),
            ("kunlunxin", "npu", "flagcx", "hccl"),
            ("kunlunxin", "npu", "nccl", "nccl"),
            ("kunlunxin", "xpu", "flagcx", "flagcx"),
            ("sunrise", "ptpu", "flagcx", "pccl"),
            ("sunrise", "ptpu", "nccl", "pccl"),
            ("sunrise", "ptpu", "hccl", "hccl"),
            ("amd", "cuda", "flagcx", "nccl"),
            ("amd", "cuda", "nccl", "nccl"),
            ("nvidia", "cuda", "gloo", "gloo"),
        ],
    )
    def test_backend_matrix(
        self, worker_env, vendor_name, device_type, backend, expected
    ):
        """Each (vendor, device, backend) normalizes to the torch backend."""
        worker, _ = worker_env(vendor_name=vendor_name, device_type=device_type)
        assert worker._normalize_process_group_backend(backend) == expected


class TestIsHybridKvModel:
    """``_is_hybrid_kv_model`` detects hybrid attention layers."""

    def test_mamba_cache_mode_align_or_all(self, worker_env):
        """mamba_cache_mode 'align'/'all' forces hybrid KV on any vendor."""
        worker, _ = worker_env()
        for mode in ("align", "all"):
            cfg = make_vllm_config()
            cfg.cache_config = SimpleNamespace(mamba_cache_mode=mode)
            assert worker._is_hybrid_kv_model(cfg) is True

    def test_mamba_cache_mode_none_falls_through(self, worker_env):
        """mamba_cache_mode 'none' does not by itself make the model hybrid."""
        worker, _ = worker_env()
        cfg = make_vllm_config()
        cfg.cache_config = SimpleNamespace(mamba_cache_mode="none")
        cfg.model_config = SimpleNamespace(is_hybrid=False, hf_text_config=None)
        assert worker._is_hybrid_kv_model(cfg) is False

    def test_model_config_is_hybrid_flag(self, worker_env):
        """A truthy model_config.is_hybrid short-circuits to True."""
        worker, _ = worker_env()
        cfg = make_vllm_config()
        cfg.cache_config = SimpleNamespace(mamba_cache_mode="none")
        cfg.model_config = SimpleNamespace(is_hybrid=True, hf_text_config=None)
        assert worker._is_hybrid_kv_model(cfg) is True

    def test_is_hybrid_attribute_raising_falls_back_to_layer_types(self, worker_env):
        """A raising is_hybrid property falls back to the layer-type scan."""
        worker, _ = worker_env()
        cfg = make_vllm_config()
        cfg.cache_config = SimpleNamespace(mamba_cache_mode="none")

        class _Raising:
            @property
            def is_hybrid(self):
                raise RuntimeError("unresolved before init")

            hf_text_config = SimpleNamespace(
                layer_types=["full_attention", "linear_attention", "full_attention"]
            )

        cfg.model_config = _Raising()
        assert worker._is_hybrid_kv_model(cfg) is True

    def test_layer_types_scan(self, worker_env):
        """A non full_attention layer_type marks the model hybrid."""
        worker, _ = worker_env()
        cfg = make_vllm_config()
        cfg.cache_config = SimpleNamespace(mamba_cache_mode="none")
        cfg.model_config = SimpleNamespace(
            is_hybrid=False,
            hf_text_config=SimpleNamespace(
                layer_types=["full_attention", "full_attention"]
            ),
        )
        assert worker._is_hybrid_kv_model(cfg) is False

    def test_layer_types_empty_tuple(self, worker_env):
        """Missing or empty layer_types leaves the model non-hybrid."""
        worker, _ = worker_env()
        cfg = make_vllm_config()
        cfg.cache_config = SimpleNamespace(mamba_cache_mode="none")
        cfg.model_config = SimpleNamespace(is_hybrid=False, hf_text_config=None)
        # hf_text_config None -> layer_types () -> any() False.
        assert worker._is_hybrid_kv_model(cfg) is False
        cfg2 = make_vllm_config()
        cfg2.cache_config = SimpleNamespace(mamba_cache_mode="none")
        cfg2.model_config = SimpleNamespace(
            is_hybrid=False, hf_text_config=SimpleNamespace(layer_types=())
        )
        assert worker._is_hybrid_kv_model(cfg2) is False

    def test_is_hybrid_no_attribute(self, worker_env):
        """A model config without is_hybrid at all still works via getattr."""
        worker, _ = worker_env()
        cfg = make_vllm_config()
        cfg.cache_config = SimpleNamespace(mamba_cache_mode="none")
        cfg.model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(layer_types=["sliding_window"])
        )
        assert worker._is_hybrid_kv_model(cfg) is True


class TestPrepareCommBufferShim:
    """The vLLM >= 0.28 userbuffer-compat import shim."""

    def test_shim_present_and_callable(self, worker_env):
        """The shim resolves to the stubbed vllm symbol and forwards."""
        worker, tree = worker_env()
        ps_mod = sys.modules["vllm.distributed.parallel_state"]
        assert (
            worker._prepare_comm_buffer is ps_mod.prepare_communication_buffer_for_model
        )
        ret = worker._prepare_comm_buffer(1, 2, kw=3)
        assert ret is None
        # The recording stub captured the forwarded call.
        assert ("_prepare_comm_buffer", (1, 2), {"kw": 3}) in tree.calls


class TestApplyWorkerVendorPatches:
    """``_apply_worker_vendor_patches`` gates iluvatar-only patches."""

    def test_non_iluvatar_vendor_returns_early(self, worker_env, caplog):
        """Any other vendor imports nothing and logs nothing."""
        worker, tree = worker_env(vendor_name="nvidia")
        with caplog.at_level(logging.DEBUG, logger="vllm_fl.worker.worker"):
            worker._apply_worker_vendor_patches()
        assert caplog.records == []
        assert tree.calls == []

    def test_iluvatar_vendor_applies_real_patches(self, worker_env, caplog):
        """vendor_name=iluvatar runs the real vendor patch functions."""
        worker, _ = worker_env(vendor_name="iluvatar")
        with caplog.at_level(logging.WARNING, logger="vllm_fl.worker.worker"):
            worker._apply_worker_vendor_patches()
        # Both patch functions tolerate CPU: the sampler patch imports
        # vllm.v1.sample.ops.* which does not exist -> its own warning
        # branch; the inductor patch sees no triton backends -> skipped.
        # Neither propagates a failure out of _apply_worker_vendor_patches.
        assert not any(
            "Iluvatar worker patch failed" in rec.message for rec in caplog.records
        )

    def test_iluvatar_patch_failure_logs_warning(self, worker_env, caplog, monkeypatch):
        """An exception inside the patch block is caught and logged."""
        worker, _ = worker_env(vendor_name="iluvatar")
        import vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar as ilu_mod

        def _boom():
            raise RuntimeError("boom")

        monkeypatch.setattr(ilu_mod, "patch_torch_inductor_for_iluvatar", _boom)
        with caplog.at_level(logging.WARNING, logger="vllm_fl.worker.worker"):
            worker._apply_worker_vendor_patches()
        assert any(
            "Iluvatar worker patch failed" in rec.message for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# AsyncIntermediateTensors
# ---------------------------------------------------------------------------


class _HandleDouble:
    """Recording double of a distributed Handle."""

    def __init__(self):
        self.waited = 0

    def wait(self):
        self.waited += 1


class TestAsyncIntermediateTensors:
    """Lazy comm synchronization for pipeline-parallel intermediates."""

    def test_wait_for_comm_runs_handles_then_postprocess(self, worker_env):
        """Handles wait in order, then postprocessors run in order."""
        worker, _ = worker_env()
        h1, h2 = _HandleDouble(), _HandleDouble()
        order = []
        handles = worker.AsyncIntermediateTensors(
            {"t": torch.zeros(1)}, comm_handles=[h1, h2]
        )
        handles._comm_postprocess = [
            lambda: order.append("p1"),
            lambda: order.append("p2"),
        ]
        handles.wait_for_comm()
        # Independent oracle: each handle waited exactly once, postprocess
        # ran after both waits.
        assert (h1.waited, h2.waited) == (1, 1)
        assert order == ["p1", "p2"]
        assert handles._comm_waited is True

    def test_wait_for_comm_is_idempotent(self, worker_env):
        """A second wait_for_comm does not re-run handles or callbacks."""
        worker, _ = worker_env()
        h1 = _HandleDouble()
        calls = []
        handles = worker.AsyncIntermediateTensors(
            {"t": torch.zeros(1)}, comm_handles=[h1]
        )
        handles._comm_postprocess = [lambda: calls.append(1)]
        handles.wait_for_comm()
        handles.wait_for_comm()
        assert h1.waited == 1
        assert calls == [1]

    def test_empty_handles_and_callbacks(self, worker_env):
        """None handles/postprocess leave wait_for_comm a cheap no-op."""
        worker, _ = worker_env()
        handles = worker.AsyncIntermediateTensors({"t": torch.zeros(1)})
        handles.wait_for_comm()
        assert handles._comm_waited is True

    def test_getattribute_tensors_triggers_lazy_wait(self, worker_env):
        """Reading ``.tensors`` first synchronizes pending communication."""
        worker, _ = worker_env()
        h1 = _HandleDouble()
        handles = worker.AsyncIntermediateTensors(
            {"t": torch.zeros(2)}, comm_handles=[h1]
        )
        # Accessing .tensors through the __getattribute__ gate waits.
        before = h1.waited
        assert handles.tensors["t"].shape[0] == 2
        assert h1.waited == before + 1
        # A second attribute read does not wait again.
        _ = handles.tensors
        assert h1.waited == before + 1

    def test_getattribute_unrelated_names_skip_wait(self, worker_env):
        """Non-tensors attributes do not trigger synchronization."""
        worker, _ = worker_env()
        h1 = _HandleDouble()
        handles = worker.AsyncIntermediateTensors(
            {"t": torch.zeros(1)}, comm_handles=[h1]
        )
        # dunder / private lookups bypass the gate
        object.__getattribute__(handles, "_comm_handles")
        assert h1.waited == 0
        assert handles._comm_waited is False

    def test_is_intermediate_tensors_subclass(self, worker_env):
        """AsyncIntermediateTensors extends the vllm base class."""
        worker, _ = worker_env()
        seq_mod = sys.modules["vllm.sequence"]
        assert issubclass(worker.AsyncIntermediateTensors, seq_mod.IntermediateTensors)

    def test_tensors_untouched_by_wait(self, worker_env):
        """The tensors dict passes through the base class unchanged."""
        worker, _ = worker_env()
        tensors = {"a": torch.arange(4), "b": torch.zeros(2, 2)}
        handles = worker.AsyncIntermediateTensors(tensors)
        assert handles.tensors is tensors


# ---------------------------------------------------------------------------
# WorkerFL.__init__
# ---------------------------------------------------------------------------


class TestWorkerFlInit:
    """Constructor branches: env setup, scheduler override, state defaults."""

    def test_default_construction(self, worker_env):
        """Plain nvidia construction wires the documented defaults."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        assert w.use_v2_model_runner is True
        assert w._sleep_saved_buffers == {}
        assert w._sleep_saved_draft_buffers == {}
        assert w.weight_transfer_engine is None
        assert w._weight_update_active is False
        assert w._weight_update_is_draft is False
        assert w.profiler is None
        assert w._pp_send_work == []
        assert w.worker_sentinel is None
        assert w.elastic_ep_executor is not None
        # init_device was not called yet: no distributed init happened.
        assert tree.calls == []

    def test_gcu_sets_triton_cache_dir(self, worker_env, monkeypatch):
        """GCU vendors give each rank its own triton cache directory."""
        worker, _ = worker_env(device_type="gcu")
        monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
        cfg = make_vllm_config()
        worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=3, distributed_init_method="tcp://x:1"
        )
        assert os.environ["TRITON_CACHE_DIR"] == "/tmp/triton-cache-fl-rank-3"

    def test_gcu_respects_existing_triton_cache_dir(self, worker_env, monkeypatch):
        """An operator-provided TRITON_CACHE_DIR is not overwritten."""
        worker, _ = worker_env(device_type="gcu")
        monkeypatch.setenv("TRITON_CACHE_DIR", "/shared/cache")
        cfg = make_vllm_config()
        worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=1, distributed_init_method="tcp://x:1"
        )
        assert os.environ["TRITON_CACHE_DIR"] == "/shared/cache"

    def test_mtp1_async_scheduling_swaps_scheduler_cls(self, worker_env):
        """num_speculative_tokens==1 + async scheduling installs AsyncSchedulerFL."""
        worker, _ = worker_env()
        cfg = make_vllm_config(num_speculative_tokens=1)
        cfg.scheduler_config.async_scheduling = True
        worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        assert (
            cfg.scheduler_config.scheduler_cls
            == "vllm_fl.worker.scheduler_fl.AsyncSchedulerFL"
        )

    def test_mtp1_sync_scheduling_keeps_scheduler_cls(self, worker_env):
        """num_speculative_tokens==1 without async scheduling changes nothing."""
        worker, _ = worker_env()
        cfg = make_vllm_config(num_speculative_tokens=1)
        cfg.scheduler_config.async_scheduling = False
        worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        assert (
            cfg.scheduler_config.scheduler_cls
            == "vllm.v1.core.sched.scheduler.Scheduler"
        )

    def test_kunlunxin_hybrid_model_defaults_reshape_env(self, worker_env):
        """kunlunxin + hybrid KV model sets USE_RESHAPE_AND_CACHE_FLASH."""
        worker, _ = worker_env(vendor_name="kunlunxin", device_type="xpu")
        cfg = make_vllm_config()
        cfg.cache_config = SimpleNamespace(mamba_cache_mode="all")
        worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        assert os.environ.get("USE_RESHAPE_AND_CACHE_FLASH") == "1"

    def test_kunlunxin_non_hybrid_model_leaves_env_alone(self, worker_env):
        """kunlunxin + a pure full-attention model does not set the env var."""
        worker, _ = worker_env(vendor_name="kunlunxin", device_type="xpu")
        cfg = make_vllm_config()
        cfg.cache_config = SimpleNamespace(mamba_cache_mode="none")
        cfg.model_config = SimpleNamespace(
            is_hybrid=False, hf_text_config=SimpleNamespace(layer_types=[])
        )
        worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        assert "USE_RESHAPE_AND_CACHE_FLASH" not in os.environ

    def test_kunlunxin_env_not_overwritten_when_present(self, worker_env, monkeypatch):
        """An existing USE_RESHAPE_AND_CACHE_FLASH value survives (setdefault)."""
        worker, _ = worker_env(vendor_name="kunlunxin", device_type="xpu")
        monkeypatch.setenv("USE_RESHAPE_AND_CACHE_FLASH", "0")
        cfg = make_vllm_config()
        cfg.cache_config = SimpleNamespace(mamba_cache_mode="all")
        worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        assert os.environ["USE_RESHAPE_AND_CACHE_FLASH"] == "0"

    def test_fault_tolerance_builds_sentinel(self, worker_env):
        """enable_fault_tolerance attaches a (stubbed) WorkerSentinel."""
        worker, _ = worker_env()
        cfg = make_vllm_config()
        cfg.parallel_config.enable_fault_tolerance = True
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        assert w.worker_sentinel is not None

    def test_float32_matmul_precision_applied(self, worker_env, monkeypatch):
        """The vllm env var value is pushed into torch matmul precision."""
        worker, _ = worker_env()
        envs_mod = sys.modules["vllm.envs"]
        monkeypatch.setattr(
            envs_mod, "VLLM_FLOAT32_MATMUL_PRECISION", "medium", raising=False
        )
        cfg = make_vllm_config()
        worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        assert torch.get_float32_matmul_precision() == "medium"


# ---------------------------------------------------------------------------
# Sleep / wake / checkpoint
# ---------------------------------------------------------------------------


def _fake_accelerator(free_values=(120,), total=256):
    """Accelerator namespace double with per-call memory info readings.

    ``get_memory_info`` returns each entry of *free_values* in turn (the last
    repeats), mirroring distinct before/after readings inside ``sleep()``.
    """
    readings = list(free_values)
    state = {"i": 0}

    def get_memory_info():
        idx = min(state["i"], len(readings) - 1)
        state["i"] += 1
        return (readings[idx], total)

    return SimpleNamespace(
        synchronize=lambda: None,
        get_memory_info=get_memory_info,
        empty_cache=lambda: None,
    )


class TestSleepWake:
    """Level-based sleep and wake with buffer save/restore."""

    def _make_worker(self, worker_env, level_free=(100, 120, 256)):
        """Build a worker whose model has named CPU buffers; patch torch.accelerator."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        model = torch.nn.Linear(2, 2)
        model.register_buffer("buf", torch.tensor([1.0, 2.0]))
        draft = torch.nn.Linear(2, 2)
        draft.register_buffer("dbuf", torch.tensor([9.0]))
        runner = _RunnerDouble()
        runner.model = model
        runner.draft_model = draft
        w.model_runner = runner
        return w, worker, tree

    def test_sleep_level1_skips_buffer_save(self, worker_env, monkeypatch):
        """Level 1 suspends without copying model buffers to host."""
        w, worker, tree = self._make_worker(worker_env)
        backend = _SleepModeBackendStub()
        monkeypatch.setattr(
            w, "_get_sleep_mode_backend", lambda: backend, raising=False
        )
        acc = _fake_accelerator()
        monkeypatch.setattr(worker.torch.accelerator, "synchronize", acc.synchronize)
        monkeypatch.setattr(
            worker.torch.accelerator,
            "get_memory_info",
            acc.get_memory_info,
            raising=False,
        )
        w.sleep(level=1)
        # Independent oracle: suspend received the level; no buffers were saved.
        assert backend.calls == [("suspend", 1)]
        assert w._sleep_saved_buffers == {}
        assert w._sleep_saved_draft_buffers == {}

    def test_sleep_level2_saves_buffers_and_draft(self, worker_env, monkeypatch):
        """Level 2 snapshots every named buffer (model and draft) to host."""
        w, worker, tree = self._make_worker(worker_env)
        backend = _SleepModeBackendStub()
        monkeypatch.setattr(
            w, "_get_sleep_mode_backend", lambda: backend, raising=False
        )
        acc = _fake_accelerator()
        monkeypatch.setattr(worker.torch.accelerator, "synchronize", acc.synchronize)
        monkeypatch.setattr(
            worker.torch.accelerator,
            "get_memory_info",
            acc.get_memory_info,
            raising=False,
        )
        w.sleep(level=2)
        assert backend.calls == [("suspend", 2)]
        # The saved copies are detached host clones, not live buffer views.
        assert set(w._sleep_saved_buffers) == {"buf"}
        assert torch.equal(w._sleep_saved_buffers["buf"], torch.tensor([1.0, 2.0]))
        assert w._sleep_saved_buffers["buf"] is not w.model_runner.model.buf
        assert set(w._sleep_saved_draft_buffers) == {"dbuf"}
        assert torch.equal(w._sleep_saved_draft_buffers["dbuf"], torch.tensor([9.0]))

    def test_sleep_asserts_when_memory_grows(self, worker_env, monkeypatch):
        """A negative freed-bytes result trips the post-sleep assertion."""
        w, worker, tree = self._make_worker(worker_env)
        backend = _SleepModeBackendStub()
        monkeypatch.setattr(
            w, "_get_sleep_mode_backend", lambda: backend, raising=False
        )
        acc = _fake_accelerator(free_values=(100, 90))
        monkeypatch.setattr(worker.torch.accelerator, "synchronize", acc.synchronize)
        monkeypatch.setattr(
            worker.torch.accelerator,
            "get_memory_info",
            acc.get_memory_info,
            raising=False,
        )
        with pytest.raises(AssertionError, match="increased after sleeping"):
            w.sleep(level=1)

    def test_wake_up_restores_buffers(self, worker_env, monkeypatch):
        """wake_up writes the saved values back and clears the store."""
        w, worker, tree = self._make_worker(worker_env)
        backend = _SleepModeBackendStub()
        monkeypatch.setattr(
            w, "_get_sleep_mode_backend", lambda: backend, raising=False
        )
        w._sleep_saved_buffers = {"buf": torch.tensor([7.0, 8.0])}
        w.model_runner.model.buf.zero_()
        runner = w.model_runner
        post_wake = []
        runner.post_kv_cache_wake_up = lambda: post_wake.append(True)
        w.wake_up()
        assert backend.calls == [("resume", None)]
        # Buffer content restored from the saved clone via copy_ semantics.
        assert torch.equal(runner.model.buf, torch.tensor([7.0, 8.0]))
        assert w._sleep_saved_buffers == {}
        assert post_wake == [True]

    def test_wake_up_weight_tags_skip_kv(self, worker_env, monkeypatch):
        """tags=['weights'] restores buffers but skips the KV cache wake."""
        w, worker, tree = self._make_worker(worker_env)
        backend = _SleepModeBackendStub()
        monkeypatch.setattr(
            w, "_get_sleep_mode_backend", lambda: backend, raising=False
        )
        w._sleep_saved_buffers = {"buf": torch.tensor([7.0, 8.0])}
        runner = w.model_runner
        post_wake = []
        runner.post_kv_cache_wake_up = lambda: post_wake.append(True)
        w.wake_up(tags=["weights"])
        assert backend.calls == [("resume", ["weights"])]
        assert torch.equal(runner.model.buf, torch.tensor([7.0, 8.0]))
        assert post_wake == []

    def test_wake_up_kv_tags_skip_buffer_restore(self, worker_env, monkeypatch):
        """tags=['kv_cache'] wakes KV but does not touch saved buffers."""
        w, worker, tree = self._make_worker(worker_env)
        backend = _SleepModeBackendStub()
        monkeypatch.setattr(
            w, "_get_sleep_mode_backend", lambda: backend, raising=False
        )
        w._sleep_saved_buffers = {"buf": torch.tensor([7.0, 8.0])}
        runner = w.model_runner
        runner.model.buf.zero_()
        post_wake = []
        runner.post_kv_cache_wake_up = lambda: post_wake.append(True)
        w.wake_up(tags=["kv_cache"])
        assert post_wake == [True]
        # Buffer not restored: stays zeroed, store still populated.
        assert torch.equal(runner.model.buf, torch.tensor([0.0, 0.0]))
        assert "buf" in w._sleep_saved_buffers

    def test_wake_up_draft_buffers(self, worker_env, monkeypatch):
        """Draft buffers restore through get_draft_model when present."""
        w, worker, tree = self._make_worker(worker_env)
        backend = _SleepModeBackendStub()
        monkeypatch.setattr(
            w, "_get_sleep_mode_backend", lambda: backend, raising=False
        )
        w._sleep_saved_draft_buffers = {"dbuf": torch.tensor([5.0])}
        runner = w.model_runner
        runner.draft_model.dbuf.zero_()
        w.wake_up()
        assert torch.equal(runner.draft_model.dbuf, torch.tensor([5.0]))
        assert w._sleep_saved_draft_buffers == {}

    def test_wake_up_draft_saved_but_draft_missing(self, worker_env, monkeypatch):
        """A vanished draft model leaves the saved store untouched."""
        w, worker, tree = self._make_worker(worker_env)
        backend = _SleepModeBackendStub()
        monkeypatch.setattr(
            w, "_get_sleep_mode_backend", lambda: backend, raising=False
        )
        w._sleep_saved_draft_buffers = {"dbuf": torch.tensor([5.0])}
        w.model_runner.draft_model = None
        w.wake_up()
        # The store clears even when the draft disappeared mid-sleep.
        assert w._sleep_saved_draft_buffers == {}

    def test_get_sleep_mode_backend_caches(self, worker_env, monkeypatch):
        """The backend instance is resolved lazily and then reused."""
        w, worker, tree = self._make_worker(worker_env)
        created = []

        def _factory(model_config):
            backend = _SleepModeBackendStub()
            created.append(backend)
            return backend

        factory_mod = sys.modules["vllm.device_allocator.sleep_mode_backend"]
        monkeypatch.setattr(
            factory_mod,
            "SleepModeBackendFactory",
            SimpleNamespace(create_backend=_factory),
        )
        first = w._get_sleep_mode_backend()
        second = w._get_sleep_mode_backend()
        assert first is second
        assert len(created) == 1


class TestCheckpointState:
    """Distributed checkpoint prepare/restore delegation."""

    def test_checkpoint_prepare_calls_boundary(self, worker_env):
        """prepare forwards to checkpoint_prepare_distributed_state."""
        worker, tree = worker_env()
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        w.checkpoint_prepare()
        assert ("cp_prepare", (), {}) in tree.calls

    def test_checkpoint_restore_calls_boundary(self, worker_env):
        """restore forwards to checkpoint_restore_distributed_state."""
        worker, tree = worker_env()
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        w.checkpoint_restore()
        assert ("cp_restore", (), {}) in tree.calls


# ---------------------------------------------------------------------------
# Weight-transfer state machine
# ---------------------------------------------------------------------------


class TestWeightTransferStateMachine:
    """start/update/finish lifecycle with draft retargeting and recovery."""

    def _make(self, worker_env, engine=None, spec=None):
        worker, tree = worker_env()
        cfg = make_vllm_config(
            speculative_config=spec, weight_transfer_config=SimpleNamespace(x=1)
        )
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        w.weight_transfer_engine = (
            engine if engine is not None else _RecordingEngine(supports_draft=True)
        )
        w.model_runner = _RunnerDouble()
        return w, worker, tree

    def test_init_weight_transfer_engine_parses_and_inits(self, worker_env):
        """parse_init_info feeds init_transfer_engine with its typed result."""
        engine = _RecordingEngine()
        w, _, _ = self._make(worker_env, engine=engine)
        w.init_weight_transfer_engine({"backend": "nccl"})
        assert engine.calls == [
            ("parse_init_info", ({"backend": "nccl"},), {}),
            ("init_transfer_engine", (("typed", {"backend": "nccl"}),), {}),
        ]

    def test_start_weight_update_happy_path(self, worker_env):
        """A fresh start activates the session and marks it non-draft."""
        engine = _RecordingEngine()
        w, _, _ = self._make(worker_env, engine=engine)
        w.start_weight_update()
        assert engine.calls == [("start_weight_update", (), {})]
        assert w._weight_update_active is True
        assert w._weight_update_is_draft is False

    def test_start_weight_update_twice_raises(self, worker_env):
        """Starting an active session again is rejected."""
        engine = _RecordingEngine()
        w, _, _ = self._make(worker_env, engine=engine)
        w.start_weight_update()
        with pytest.raises(RuntimeError, match="already active"):
            w.start_weight_update()

    def test_start_draft_without_support_raises(self, worker_env):
        """A non-draft-capable engine rejects draft sessions."""
        engine = _RecordingEngine(supports_draft=False)
        w, _, _ = self._make(worker_env, engine=engine)
        with pytest.raises(RuntimeError, match="does not support"):
            w.start_draft_weight_update()

    def test_start_draft_with_model_retargets_engine(self, worker_env):
        """Draft start sets the draft model as the update target."""
        engine = _RecordingEngine(supports_draft=True)
        spec = SimpleNamespace(draft_model_config=SimpleNamespace(name="draft"))
        w, _, _ = self._make(worker_env, engine=engine, spec=spec)
        draft_model = SimpleNamespace(draft=True)
        w.model_runner.draft_model = draft_model
        w.start_draft_weight_update()
        assert (
            "set_weight_update_target",
            (draft_model, spec.draft_model_config),
            {},
        ) in (engine.calls)
        assert w._weight_update_is_draft is True

    def test_draft_without_draft_model_raises_and_resets(self, worker_env):
        """A missing draft model aborts the session start cleanly."""
        engine = _RecordingEngine(supports_draft=True)
        spec = SimpleNamespace(draft_model_config=SimpleNamespace(name="draft"))
        w, _, _ = self._make(worker_env, engine=engine, spec=spec)
        with pytest.raises(RuntimeError, match="no draft model is configured"):
            w.start_draft_weight_update()
        assert engine.calls == [("reset_weight_update_target", (), {})]
        assert w._weight_update_active is False

    def test_draft_without_spec_config_raises(self, worker_env):
        """A missing speculative config aborts the draft session."""
        engine = _RecordingEngine(supports_draft=True)
        w, _, _ = self._make(worker_env, engine=engine, spec=None)
        w.model_runner.draft_model = SimpleNamespace(draft=True)
        with pytest.raises(RuntimeError, match="no draft model config"):
            w.start_draft_weight_update()
        assert engine.calls == [("reset_weight_update_target", (), {})]
        assert w._weight_update_active is False

    def test_update_weights_requires_active_session(self, worker_env):
        """update_weights before start_weight_update is rejected."""
        engine = _RecordingEngine()
        w, _, _ = self._make(worker_env, engine=engine)
        with pytest.raises(RuntimeError, match="must be called before"):
            w.update_weights({"chunk": 1})

    def test_update_weights_forwards_payload(self, worker_env):
        """An active session forwards the chunk payload to the engine."""
        engine = _RecordingEngine()
        w, _, _ = self._make(worker_env, engine=engine)
        w.start_weight_update()
        engine.calls.clear()
        w.update_weights({"chunk": 7})
        assert ("update_weights", ({"chunk": 7},), {}) in engine.calls

    def test_update_weights_failure_deactivates_and_resets(self, worker_env):
        """A raising update both clears the active flag and resets the target."""

        class _FailingEngine(_RecordingEngine):
            def update_weights(self, info):
                raise RuntimeError("chunk corrupted")

        engine = _FailingEngine()
        w, _, _ = self._make(worker_env, engine=engine)
        w.start_weight_update()
        with pytest.raises(RuntimeError, match="chunk corrupted"):
            w.update_weights({"chunk": 1})
        assert w._weight_update_active is False
        assert ("reset_weight_update_target", (), {}) in engine.calls
        # The session is gone: a follow-up update is refused.
        with pytest.raises(RuntimeError, match="must be called before"):
            w.update_weights({"chunk": 2})

    def test_finish_weight_update_resets_state(self, worker_env):
        """finish stops the session, resets the target and lora state."""
        engine = _RecordingEngine()
        w, _, _ = self._make(worker_env, engine=engine)
        w.start_weight_update()
        engine.calls.clear()
        w.finish_weight_update()
        assert ("finish_weight_update", (), {}) in engine.calls
        assert ("reset_weight_update_target", (), {}) in engine.calls
        assert w._weight_update_active is False
        # Non-draft finishes reset the runner's lora state.
        assert any(name == "reset_lora_state" for name, _, _ in w.model_runner.calls)

    def test_finish_draft_update_skips_lora_reset(self, worker_env):
        """A draft session finish leaves the main model's lora state alone."""
        engine = _RecordingEngine(supports_draft=True)
        spec = SimpleNamespace(draft_model_config=SimpleNamespace(name="draft"))
        w, _, _ = self._make(worker_env, engine=engine, spec=spec)
        w.model_runner.draft_model = SimpleNamespace(draft=True)
        w.start_draft_weight_update()
        w.model_runner.calls.clear()
        w.finish_weight_update()
        assert w._weight_update_active is False
        assert not any(
            name == "reset_lora_state" for name, _, _ in w.model_runner.calls
        )

    def test_finish_without_start_raises(self, worker_env):
        """finish_weight_update without a session is rejected."""
        engine = _RecordingEngine()
        w, _, _ = self._make(worker_env, engine=engine)
        with pytest.raises(RuntimeError, match="without a matching start"):
            w.finish_weight_update()

    def test_requires_engine(self, worker_env):
        """Every entry point rejects a worker without weight transfer config."""
        worker, _ = worker_env()
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        with pytest.raises(RuntimeError, match="Weight transfer not configured"):
            w.init_weight_transfer_engine({})
        with pytest.raises(RuntimeError, match="Weight transfer not configured"):
            w.start_weight_update()
        with pytest.raises(RuntimeError, match="Weight transfer not configured"):
            w.update_weights({})
        with pytest.raises(RuntimeError, match="Weight transfer not configured"):
            w.finish_weight_update()

    def test_supports_draft_weight_updates_matrix(self, worker_env):
        """The capability probe requires engine, draft model and spec config."""
        engine = _RecordingEngine(supports_draft=True)
        spec = SimpleNamespace(draft_model_config=SimpleNamespace(name="draft"))
        w, _, _ = self._make(worker_env, engine=engine, spec=spec)
        # Everything present -> True.
        w.model_runner.draft_model = SimpleNamespace(draft=True)
        assert w.supports_draft_weight_updates() is True
        # No draft model -> False.
        w.model_runner.draft_model = None
        assert w.supports_draft_weight_updates() is False
        # Engine without draft support -> False.
        w.weight_transfer_engine = _RecordingEngine(supports_draft=False)
        w.model_runner.draft_model = SimpleNamespace(draft=True)
        assert w.supports_draft_weight_updates() is False
        # No speculative config at all -> False.
        w.speculative_config = None
        assert w.supports_draft_weight_updates() is False
        # No engine -> False.
        w.weight_transfer_engine = None
        assert w.supports_draft_weight_updates() is False


def _stub_allocator_settings(worker, monkeypatch):
    """Provide the accelerator-only allocator knob on CPU-only torch builds.

    ``load_model`` always enters ``_scoped_allocator_max_split``, whose real
    body calls ``torch._C._accelerator_setAllocatorSettings`` -- a private API
    that only exists on accelerator-enabled torch builds.  Stubbing it lets
    the load_model code under test run unmodified on a CPU-only build.
    """
    monkeypatch.setattr(
        worker.torch._C,
        "_accelerator_setAllocatorSettings",
        lambda setting: None,
        raising=False,
    )


class TestLoadModel:
    """load_model engine creation and npu-only thread pinning."""

    def test_engine_created_when_configured(self, worker_env, monkeypatch):
        """A weight_transfer_config builds the engine around the model."""
        worker, tree = worker_env()
        _stub_allocator_settings(worker, monkeypatch)
        cfg = make_vllm_config(weight_transfer_config=SimpleNamespace(kind="nccl"))
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        w.device = torch.device("cpu")
        w.model_runner = _RunnerDouble()
        loaded = []
        w.model_runner.load_model = lambda **kw: loaded.append(kw)
        w.load_model()
        assert loaded == [{"load_dummy_weights": False}]
        assert isinstance(w.weight_transfer_engine, _RecordingEngine)
        # The factory consumed the runner's model and this worker's device.
        (name, got_cfg, got_dev, got_model), *_ = tree.factory_calls
        assert name == "create_engine"
        assert got_cfg is cfg.weight_transfer_config
        assert got_dev is w.device
        assert got_model is w.model_runner.get_model()

    def test_engine_not_created_without_config(self, worker_env, monkeypatch):
        """No weight_transfer_config leaves the engine unset."""
        worker, tree = worker_env()
        _stub_allocator_settings(worker, monkeypatch)
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        w.device = torch.device("cpu")
        w.model_runner = _RunnerDouble()
        w.model_runner.load_model = lambda **kw: None
        w.load_model()
        assert w.weight_transfer_engine is None

    def test_npu_load_pins_torch_threads(self, worker_env, monkeypatch):
        """An npu device pins torch threads after loading weights."""
        worker, tree = worker_env()
        _stub_allocator_settings(worker, monkeypatch)
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        w.device = SimpleNamespace(type="npu")
        w.model_runner = _RunnerDouble()
        w.model_runner.load_model = lambda **kw: None
        w.load_model()
        assert ("set_torch_threads_for_runtime", (), {}) in tree.calls


# ---------------------------------------------------------------------------
# init_device
# ---------------------------------------------------------------------------


def _setup_device_fakes(worker, monkeypatch, tree, device_count=8):
    """Patch the torch/vllm surfaces init_device touches.

    Worker.py bound ``MemorySnapshot`` / ``request_memory`` by name at import
    time, so the doubles are patched onto the worker module's own globals —
    the module under test keeps running its real code.
    """
    acc = SimpleNamespace(
        device_count=lambda: device_count,
        synchronize=lambda: None,
        get_memory_info=lambda: (10, 20),
        empty_cache=lambda: None,
    )
    monkeypatch.setattr(worker.torch.accelerator, "device_count", acc.device_count)
    monkeypatch.setattr(worker.torch.accelerator, "empty_cache", acc.empty_cache)
    monkeypatch.setattr(worker.torch, "device", torch.device)
    monkeypatch.setattr(gc, "collect", lambda: None)
    # Memory snapshot double.
    tree.memory_snapshot_calls = []

    def _snapshot(device=None):
        tree.memory_snapshot_calls.append(device)
        return SimpleNamespace(free_memory=100, total_memory=256)

    monkeypatch.setattr(worker, "MemorySnapshot", _snapshot)
    monkeypatch.setattr(worker, "request_memory", lambda snap, cc: 90)


class TestInitDevice:
    """init_device rank arithmetic, distributed init, and runner choice."""

    def test_basic_init_device_flow(self, worker_env, monkeypatch):
        """Plain nvidia flow: dist init, snapshot, V2 runner construction."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        _setup_device_fakes(worker, monkeypatch, tree)
        w.init_device()
        names = [name for name, _, _ in tree.calls]
        # Distributed init happens before the memory snapshot.
        assert "init_distributed_environment" in names
        assert "init_worker_distributed_environment" not in names
        assert names.index("init_distributed_environment") < len(names) - 1
        # Dist init received the normalized backend and the rank tuple.
        dist_call = next(
            c for c in tree.calls if c[0] == "init_distributed_environment"
        )
        assert dist_call[1] == (1, 0, "tcp://x:1", 0, "nccl", None)
        # Workspace manager saw the device, ubatch and lane counts.
        ws_call = next(c for c in tree.calls if c[0] == "init_workspace_manager")
        _, args, kwargs = ws_call
        assert kwargs == {} or args[-1] == 1
        # V2 runner was built with this worker's config and device.
        assert w.model_runner.__class__ is tree.gpu_model_runner_cls
        assert tree.gpu_runner_builds == [(cfg, w.device)]

    def test_init_device_rank0_reports_usage(self, worker_env, monkeypatch):
        """rank 0 collects usage stats after the runner is built."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        _setup_device_fakes(worker, monkeypatch, tree)
        w.init_device()
        assert ("report_usage_stats", (cfg,), {}) in tree.calls

    def test_init_device_nonzero_rank_skips_usage(self, worker_env, monkeypatch):
        """Non-driver ranks do not report usage stats."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=1, rank=1, distributed_init_method="tcp://x:1"
        )
        _setup_device_fakes(worker, monkeypatch, tree, device_count=8)
        w.init_device()
        assert not any(name == "report_usage_stats" for name, _, _ in tree.calls)

    def test_init_device_dp_rank_offset_applied(self, worker_env, monkeypatch):
        """A full device list adds the DP offset to local_rank."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        cfg.parallel_config.data_parallel_rank_local = 2
        cfg.parallel_config.local_world_size = 4
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=1, rank=0, distributed_init_method="tcp://x:1"
        )
        _setup_device_fakes(worker, monkeypatch, tree, device_count=16)
        w.init_device()
        # Independent oracle: 1 + 2*4 = 9.
        assert w.local_rank == 9

    def test_init_device_dp_offset_uses_global_when_local_none(
        self, worker_env, monkeypatch
    ):
        """A missing local DP rank falls back to the global DP index."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        cfg.parallel_config.data_parallel_rank_local = None
        cfg.parallel_config.data_parallel_index = 3
        cfg.parallel_config.local_world_size = 2
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=1, rank=0, distributed_init_method="tcp://x:1"
        )
        _setup_device_fakes(worker, monkeypatch, tree, device_count=16)
        w.init_device()
        # Independent oracle: 1 + 3*2 = 7.
        assert w.local_rank == 7

    def test_init_device_rank_slice_skips_dp_offset(self, worker_env, monkeypatch):
        """assigned ids sized to local_world_size skip the DP adjustment."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        cfg.parallel_config.assigned_physical_gpu_ids = [0, 1]
        cfg.parallel_config.local_world_size = 2
        cfg.parallel_config.data_parallel_rank_local = 5
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=1, rank=0, distributed_init_method="tcp://x:1"
        )
        _setup_device_fakes(worker, monkeypatch, tree, device_count=16)
        w.init_device()
        assert w.local_rank == 1

    def test_init_device_ray_backend_skips_dp_offset(self, worker_env, monkeypatch):
        """ray / external_launcher backends never receive the DP offset."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        cfg.parallel_config.distributed_executor_backend = "ray"
        cfg.parallel_config.data_parallel_rank_local = 5
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=1, rank=0, distributed_init_method="tcp://x:1"
        )
        _setup_device_fakes(worker, monkeypatch, tree, device_count=16)
        w.init_device()
        assert w.local_rank == 1

    def test_init_device_publishes_assigned_ids(self, worker_env, monkeypatch):
        """assigned_physical_gpu_ids installs the logical-to-physical map."""
        worker, tree = worker_env()
        published = []
        monkeypatch.setattr(
            sys.modules["vllm.platforms.interface"],
            "set_assigned_physical_gpu_ids",
            published.append,
        )
        cfg = make_vllm_config()
        cfg.parallel_config.assigned_physical_gpu_ids = [4, 5]
        cfg.parallel_config.local_world_size = 2
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=1, rank=0, distributed_init_method="tcp://x:1"
        )
        _setup_device_fakes(worker, monkeypatch, tree, device_count=16)
        w.init_device()
        assert published == [[4, 5]]

    def test_init_device_mismatched_device_type_raises(self, worker_env):
        """A device_config mismatch is rejected with the upstream error."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        w.device_config = SimpleNamespace(device_type="cpu", device="cpu")
        with pytest.raises(RuntimeError, match="Unsupported device type"):
            w.init_device()

    def test_init_device_local_rank_out_of_bounds(self, worker_env, monkeypatch):
        """A local rank beyond the visible device count is rejected."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=3, rank=0, distributed_init_method="tcp://x:1"
        )
        _setup_device_fakes(worker, monkeypatch, tree, device_count=2)
        with pytest.raises(AssertionError, match="out of"):
            w.init_device()

    def test_init_device_v1_runner(self, worker_env, monkeypatch):
        """use_v2_model_runner False builds the FL ModelRunnerFL."""
        worker, tree = worker_env()
        built = []

        class _MarkerRunner:
            def __init__(self, vllm_config, device):
                built.append((vllm_config, device))

        fake_mr = ModuleType("vllm_fl.worker.model_runner")
        fake_mr.ModelRunnerFL = _MarkerRunner
        monkeypatch.setitem(sys.modules, "vllm_fl.worker.model_runner", fake_mr)
        cfg = make_vllm_config(use_v2_model_runner=False)
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        _setup_device_fakes(worker, monkeypatch, tree)
        w.init_device()
        assert isinstance(w.model_runner, _MarkerRunner)
        # Independent oracle: the runner got this worker's config and device.
        assert built == [(cfg, w.device)]

    def test_init_device_npu_device_runs_ascend_property_init(
        self, worker_env, monkeypatch
    ):
        """npu devices run the ascend triton property init after dist setup."""
        worker, tree = worker_env(vendor_name="ascend", device_type="npu")
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        _setup_device_fakes(worker, monkeypatch, tree)
        # torch.device rejects unknown device strings on a CPU build; give
        # this test a permissive stand-in that only formats the string.
        monkeypatch.setattr(
            worker.torch, "device", lambda spec: SimpleNamespace(type="npu", dev=spec)
        )

        class MarkerRunnerV2Npu:
            def __init__(self, vllm_config, device):
                self.vllm_config = vllm_config
                self.device = device

        fake_shims = ModuleType(
            "vllm_fl.dispatch.backends.vendor.ascend.patches.ascend_v2_device"
        )
        fake_shims.install_ascend_v2_device_shims = lambda: None
        monkeypatch.setitem(
            sys.modules,
            "vllm_fl.dispatch.backends.vendor.ascend.patches.ascend_v2_device",
            fake_shims,
        )
        fake_v2 = ModuleType(
            "vllm_fl.dispatch.backends.vendor.ascend.worker.model_runner_v2"
        )
        fake_v2.GPUModelRunnerFLV2 = MarkerRunnerV2Npu
        monkeypatch.setitem(
            sys.modules,
            "vllm_fl.dispatch.backends.vendor.ascend.worker.model_runner_v2",
            fake_v2,
        )
        # Stub the deep ascend triton init module the branch imports.
        ascend_utils = ModuleType(
            "vllm_fl.dispatch.backends.vendor.ascend.impl.triton_utils"
        )
        calls = []
        ascend_utils.init_device_properties_triton = lambda: calls.append(True)
        monkeypatch.setitem(
            sys.modules,
            "vllm_fl.dispatch.backends.vendor.ascend.impl.triton_utils",
            ascend_utils,
        )
        w.init_device()
        assert calls == [True]
        # The npu V2 runner comes from the ascend vendor package (stubbed
        # below as a marker class built with config + device).
        assert w.model_runner.__class__.__name__ == "MarkerRunnerV2Npu"


# ---------------------------------------------------------------------------
# init_worker_distributed_environment
# ---------------------------------------------------------------------------


class TestInitWorkerDistributedEnvironment:
    """Free-function distributed bootstrap with EPLB and backend mapping."""

    def test_full_forwarding(self, worker_env):
        """Every argument reaches the vllm distributed boundary."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        worker.init_worker_distributed_environment(
            cfg, rank=3, distributed_init_method="tcp://h:9", local_rank=2
        )
        assert ("set_custom_all_reduce", (True,), {}) in tree.calls
        dist = next(c for c in tree.calls if c[0] == "init_distributed_environment")
        # world_size, rank, init_method, local_rank, backend, timeout
        assert dist[1][0] == 1
        assert dist[1][1] == 3
        assert dist[1][2] == "tcp://h:9"
        assert dist[1][3] == 2
        assert dist[1][4] == "nccl"
        assert dist[1][5] is None
        assert ("ensure_model_parallel_initialized", (1, 1, 1, 1), {}) in tree.calls
        assert any(
            name == "ensure_ec_transfer_initialized" for name, _, _ in tree.calls
        )

    def test_default_init_method_env(self, worker_env):
        """A missing distributed_init_method falls back to env://."""
        worker, tree = worker_env()
        worker.init_worker_distributed_environment(
            make_vllm_config(), rank=0, distributed_init_method=None, local_rank=0
        )
        dist = next(c for c in tree.calls if c[0] == "init_distributed_environment")
        assert dist[1][2] == "env://"

    def test_timeout_forwarded(self, worker_env):
        """distributed_timeout_seconds becomes a timedelta."""
        from datetime import timedelta

        worker, tree = worker_env()
        cfg = make_vllm_config()
        cfg.parallel_config.distributed_timeout_seconds = 30
        worker.init_worker_distributed_environment(
            cfg, rank=0, distributed_init_method="env://", local_rank=0
        )
        dist = next(c for c in tree.calls if c[0] == "init_distributed_environment")
        assert dist[1][5] == timedelta(seconds=30)

    def test_custom_all_reduce_disabled(self, worker_env):
        """disable_custom_all_reduce flips the set_custom_all_reduce flag."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        cfg.parallel_config.disable_custom_all_reduce = True
        worker.init_worker_distributed_environment(
            cfg, rank=0, distributed_init_method="env://", local_rank=0
        )
        assert ("set_custom_all_reduce", (False,), {}) in tree.calls

    def test_eplb_env_override(self, worker_env):
        """override_envs_for_eplb receives the kernel moe_backend."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        worker.init_worker_distributed_environment(
            cfg, rank=0, distributed_init_method="env://", local_rank=0
        )
        eplb = next(c for c in tree.calls if c[0] == "override_envs_for_eplb")
        assert eplb[1][0] is cfg.parallel_config
        assert eplb[2] == {"moe_backend": "triton"}

    def test_ptpu_backend_normalized_to_pccl(self, worker_env):
        """A ptpu platform maps the nccl default to pccl."""
        worker, tree = worker_env(vendor_name="sunrise", device_type="ptpu")
        cfg = make_vllm_config()
        worker.init_worker_distributed_environment(
            cfg, rank=0, distributed_init_method="env://", local_rank=0
        )
        dist = next(c for c in tree.calls if c[0] == "init_distributed_environment")
        assert dist[1][4] == "pccl"


# ---------------------------------------------------------------------------
# execute_model / annotate_profile / profile
# ---------------------------------------------------------------------------


class _PPGroupStub:
    """PP group double with configurable first/last rank flags."""

    def __init__(self, is_first_rank=True, is_last_rank=True):
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank
        self.irecv_calls = []
        self.isend_calls = []

    def irecv_tensor_dict(self, all_gather_group=None, all_gather_tensors=None):
        self.irecv_calls.append(all_gather_tensors)
        return ({"hidden": torch.zeros(1)}, [], [])

    def isend_tensor_dict(
        self, tensors, all_gather_group=None, all_gather_tensors=None
    ):
        self.isend_calls.append((tensors, all_gather_tensors))
        return []


class TestExecuteModel:
    """execute_model pipeline plumbing and PP send bookkeeping."""

    def _make(self, worker_env, pp_group):
        worker, tree = worker_env()
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        w.model_runner = _RunnerDouble()
        w._pp_groups = pp_group
        return w, worker, tree

    def test_pending_pp_sends_waited_first(self, worker_env, monkeypatch):
        """Leftover non-blocking sends complete before the new forward."""
        worker, tree = worker_env()
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        waited = []

        def _handle(idx):
            return SimpleNamespace(wait=lambda: waited.append(idx))

        w._pp_send_work = [_handle(i) for i in range(2)]
        w.model_runner = _RunnerDouble()
        pp = _PPGroupStub()
        monkeypatch.setattr(worker, "get_pp_group", lambda: pp)
        monkeypatch.setattr(worker, "get_tp_group", lambda: SimpleNamespace())
        so = SimpleNamespace(total_num_scheduled_tokens=0)
        w.execute_model(so)
        # Both stale handles waited exactly once, in order.
        assert waited == [0, 1]
        assert w._pp_send_work == []

    def test_first_rank_skips_irecv(self, worker_env, monkeypatch):
        """A first-rank worker runs the forward without receiving tensors."""
        worker, tree = worker_env()
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        w.model_runner = _RunnerDouble()
        pp = _PPGroupStub(is_first_rank=True, is_last_rank=True)
        monkeypatch.setattr(worker, "get_pp_group", lambda: pp)
        monkeypatch.setattr(worker, "get_tp_group", lambda: SimpleNamespace())
        so = SimpleNamespace(total_num_scheduled_tokens=4)
        w.execute_model(so)
        assert pp.irecv_calls == []
        # Runner got the scheduler output and no intermediate tensors.
        assert ("execute_model", (so, None), {}) in w.model_runner.calls

    def test_non_first_rank_irecv_builds_async_intermediates(
        self, worker_env, monkeypatch
    ):
        """A middle-rank worker receives intermediates lazily."""
        worker, tree = worker_env()
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        w.model_runner = _RunnerDouble()
        pp = _PPGroupStub(is_first_rank=False, is_last_rank=True)
        monkeypatch.setattr(worker, "get_pp_group", lambda: pp)
        monkeypatch.setattr(worker, "get_tp_group", lambda: SimpleNamespace())
        so = SimpleNamespace(total_num_scheduled_tokens=4)
        w.execute_model(so)
        assert len(pp.irecv_calls) == 1
        # The runner received an AsyncIntermediateTensors wrapping the dict.
        run_call = next(c for c in w.model_runner.calls if c[0] == "execute_model")
        intermediates = run_call[1][1]
        assert isinstance(intermediates, worker.AsyncIntermediateTensors)
        assert intermediates.tensors == {"hidden": torch.zeros(1)}

    def test_zero_tokens_skips_irecv_only(self, worker_env, monkeypatch):
        """A zero-token scheduler output still forwards but receives nothing."""
        worker, tree = worker_env()
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        w.model_runner = _RunnerDouble()
        pp = _PPGroupStub(is_first_rank=False, is_last_rank=True)
        monkeypatch.setattr(worker, "get_pp_group", lambda: pp)
        monkeypatch.setattr(worker, "get_tp_group", lambda: SimpleNamespace())
        so = SimpleNamespace(total_num_scheduled_tokens=0)
        w.execute_model(so)
        # No irecv happened, but the runner still ran with no intermediates.
        assert pp.irecv_calls == []
        assert ("execute_model", (so, None), {}) in w.model_runner.calls
        assert pp.isend_calls == []


class TestAnnotateProfile:
    """Trace annotation strings for context/generation phases."""

    def _make(self, worker_env):
        worker, tree = worker_env()
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        return w, worker

    @staticmethod
    def _scheduler_output():
        """Scheduler output double: one new ctx req, one cached gen req."""
        new_req = SimpleNamespace(req_id="r1", num_computed_tokens=0)
        cached = SimpleNamespace(
            req_ids=["r2"],
            num_computed_tokens=[3],
            is_context_phase=lambda req_id: False,
        )
        return SimpleNamespace(
            scheduled_new_reqs=[new_req],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens={"r1": 5, "r2": 2},
        )

    def test_no_profiler_returns_nullcontext(self, worker_env):
        """Without a profiler the annotation is a no-op nullcontext."""
        w, worker = self._make(worker_env)
        ctx = w.annotate_profile(self._scheduler_output())
        assert isinstance(ctx, contextlib.nullcontext)

    def test_profiler_not_running_returns_nullcontext(self, worker_env):
        """A stopped profiler still short-circuits to nullcontext."""
        w, worker = self._make(worker_env)
        w.profiler = SimpleNamespace(
            step=lambda: None, is_running=False, annotate_context_manager=lambda a: a
        )
        ctx = w.annotate_profile(self._scheduler_output())
        assert isinstance(ctx, contextlib.nullcontext)

    def test_simple_annotation(self, worker_env):
        """Non-detailed traces annotate only request/token counts."""
        w, worker = self._make(worker_env)
        w.profiler = SimpleNamespace(
            step=lambda: None, is_running=True, annotate_context_manager=lambda a: a
        )
        so = self._scheduler_output()
        w.vllm_config.profiler_config = SimpleNamespace(detailed_trace_annotation=False)
        worker.compute_iteration_details = lambda so: SimpleNamespace(
            num_ctx_requests=1,
            num_ctx_tokens=5,
            num_generation_requests=1,
            num_generation_tokens=2,
        )
        ann = w.annotate_profile(so)
        assert "execute_context_1(5)" in ann
        assert "_generation_1(2)" in ann
        assert "sqsq" not in ann

    def test_detailed_annotation_arithmetic(self, worker_env):
        """Detailed traces compute qq/qk sums per phase."""
        w, worker = self._make(worker_env)
        w.profiler = SimpleNamespace(
            step=lambda: None, is_running=True, annotate_context_manager=lambda a: a
        )
        so = self._scheduler_output()
        w.vllm_config.profiler_config = SimpleNamespace(detailed_trace_annotation=True)
        worker.compute_iteration_details = lambda so: SimpleNamespace(
            num_ctx_requests=1,
            num_ctx_tokens=5,
            num_generation_requests=1,
            num_generation_tokens=2,
        )
        ann = w.annotate_profile(so)
        # Independent oracle: r1 is new (ctx): query 5, seq 0+5=5 ->
        # qq=25, qk=25; r2 is gen: query 2, seq 3+2=5 -> qq=4, qk=10.
        assert "sq5" in ann
        assert "sqsq25" in ann
        assert "sqsk25" in ann
        assert "sk5" in ann
        assert "sqsq4" in ann
        assert "sqsk10" in ann

    def test_detailed_annotation_cached_ctx_request(self, worker_env):
        """A cached request in context phase counts toward ctx metrics."""
        w, worker = self._make(worker_env)
        w.profiler = SimpleNamespace(
            step=lambda: None, is_running=True, annotate_context_manager=lambda a: a
        )
        new_req = SimpleNamespace(req_id="r1", num_computed_tokens=0)
        cached = SimpleNamespace(
            req_ids=["r2"],
            num_computed_tokens=[2],
            is_context_phase=lambda req_id: req_id == "r2",
        )
        so = SimpleNamespace(
            scheduled_new_reqs=[new_req],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens={"r1": 1, "r2": 3},
        )
        w.vllm_config.profiler_config = SimpleNamespace(detailed_trace_annotation=True)
        worker.compute_iteration_details = lambda so: SimpleNamespace(
            num_ctx_requests=2,
            num_ctx_tokens=4,
            num_generation_requests=0,
            num_generation_tokens=0,
        )
        ann = w.annotate_profile(so)
        # Both requests are context phase: total tokens 4, seq sums
        # r1: 0+1=1, r2: 2+3=5 -> sk6; qq 1+9=10; qk 1+15=16.
        assert "sk6" in ann
        assert "sqsq10" in ann
        assert "sqsk16" in ann


class TestProfile:
    """Profiler lifecycle start/stop through the wrapper classes."""

    def _make(self, worker_env, profiler_type="torch"):
        worker, tree = worker_env()
        cfg = make_vllm_config(profiler_config=SimpleNamespace(profiler=profiler_type))
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        return w, worker

    def test_profiling_disabled_raises(self, worker_env):
        """No profiler_config refuses profile() with guidance."""
        w, worker = self._make(worker_env, profiler_type=None)
        w.profiler_config = None
        with pytest.raises(RuntimeError, match="Profiling is not enabled"):
            w.profile(True)

    def test_none_profiler_raises(self, worker_env):
        """A config without a profiler entry also refuses."""
        w, worker = self._make(worker_env)
        w.profiler_config = SimpleNamespace(profiler=None)
        with pytest.raises(RuntimeError, match="Profiling is not enabled"):
            w.profile(True)

    def test_torch_profiler_start_stop(self, worker_env, monkeypatch):
        """torch type builds a TorchProfilerWrapper, start and stop it."""
        w, worker = self._make(worker_env, profiler_type="torch")
        built = []
        started = []
        stopped = []

        class _Wrap:
            def __init__(self, cfg, worker_name, local_rank, activities):
                built.append((cfg, worker_name, local_rank, activities))

            def start(self):
                started.append(True)

            def stop(self):
                stopped.append(True)

        # worker.py binds the wrapper classes by name at import time.
        monkeypatch.setattr(worker, "TorchProfilerWrapper", _Wrap)
        rank_utils = ModuleType("vllm.distributed.utils")
        rank_utils.get_worker_rank_suffix = lambda global_rank: f"rank_{global_rank}"
        monkeypatch.setitem(sys.modules, "vllm.distributed.utils", rank_utils)
        w.profile(True, profile_prefix="trace")
        assert w.profiler is not None
        assert started == [True]
        w.profile(False)
        assert stopped == [True]

    def test_torch_profiler_trace_name_combines_prefix(self, worker_env, monkeypatch):
        """The trace name joins the prefix with the worker rank suffix."""
        w, worker = self._make(worker_env, profiler_type="torch")
        built = []

        class _Wrap:
            def __init__(self, cfg, worker_name, local_rank, activities):
                built.append(worker_name)

            def start(self):
                pass

            def stop(self):
                pass

        monkeypatch.setattr(worker, "TorchProfilerWrapper", _Wrap)
        rank_utils = ModuleType("vllm.distributed.utils")
        rank_utils.get_worker_rank_suffix = lambda global_rank: f"rank_{global_rank}"
        monkeypatch.setitem(sys.modules, "vllm.distributed.utils", rank_utils)
        w.profile(True, profile_prefix="trace")
        # Independent oracle: prefix + suffix for rank 0.
        assert built == ["trace_rank_0"]
        # Without a prefix the suffix alone names the trace.
        w.profiler = None
        w.profile(True)
        assert built == ["trace_rank_0", "rank_0"]

    def test_cuda_profiler_start_stop(self, worker_env, monkeypatch):
        """cuda type builds the lightweight CudaProfilerWrapper."""
        w, worker = self._make(worker_env, profiler_type="cuda")
        built = []
        rank_utils = ModuleType("vllm.distributed.utils")
        rank_utils.get_worker_rank_suffix = lambda global_rank: "r"
        monkeypatch.setitem(sys.modules, "vllm.distributed.utils", rank_utils)

        class _Wrap:
            def __init__(self, cfg):
                built.append(cfg)

            def start(self):
                pass

            def stop(self):
                pass

        monkeypatch.setattr(worker, "CudaProfilerWrapper", _Wrap)
        w.profile(True)
        assert built == [w.profiler_config]
        w.profile(False)
        # cuda wrappers survive a stop: only proton is recreated.
        assert w.profiler is not None

    def test_invalid_profiler_type_raises(self, worker_env, monkeypatch):
        """An unknown profiler string is rejected at wrapper construction."""
        w, worker = self._make(worker_env, profiler_type="bogus")
        rank_utils = ModuleType("vllm.distributed.utils")
        rank_utils.get_worker_rank_suffix = lambda global_rank: "r"
        monkeypatch.setitem(sys.modules, "vllm.distributed.utils", rank_utils)
        with pytest.raises(ValueError, match="Invalid profiler"):
            w.profile(True)

    def test_stop_without_start_warns(self, worker_env, caplog):
        """Stopping a never-started profiler logs and returns."""
        w, worker = self._make(worker_env, profiler_type="torch")
        with caplog.at_level(logging.WARNING):
            w.profile(False)
        assert any("nothing to stop" in rec.message for rec in caplog.records)

    def test_proton_profiler_recreated_after_stop(self, worker_env, monkeypatch):
        """proton wrappers drop themselves so the next prefix is honored."""
        w, worker = self._make(worker_env, profiler_type="proton")
        rank_utils = ModuleType("vllm.distributed.utils")
        rank_utils.get_worker_rank_suffix = lambda global_rank: "r"
        monkeypatch.setitem(sys.modules, "vllm.distributed.utils", rank_utils)

        class _Wrap:
            def __init__(self, cfg, worker_name):
                pass

            def start(self):
                pass

            def stop(self):
                pass

        monkeypatch.setattr(worker, "ProtonProfilerWrapper", _Wrap)
        w.profile(True)
        first = w.profiler
        w.profile(False)
        assert w.profiler is None
        w.profile(True)
        assert w.profiler is not first


# ---------------------------------------------------------------------------
# memory pool contexts / allocator max split
# ---------------------------------------------------------------------------


class TestScopedAllocatorMaxSplit:
    """The PYTORCH_CUDA_ALLOC_CONF max-split guard."""

    def _make(self, worker_env):
        worker, tree = worker_env()
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        return w, worker

    def test_non_cuda_platform_is_passthrough(self, worker_env, monkeypatch):
        """Non-CUDA platforms never touch the allocator settings."""
        w, worker = self._make(worker_env)
        settings = []
        monkeypatch.setattr(
            worker.current_platform,
            "is_cuda",
            lambda: False,
            raising=False,
        )
        # torch._C._accelerator_setAllocatorSettings must not be called.
        with w._scoped_allocator_max_split(20):
            settings.append("inside")
        assert settings == ["inside"]

    def test_cuda_platform_sets_and_restores(self, worker_env, monkeypatch):
        """CUDA sets max_split_size_mb and restores SIZE_MAX on exit."""
        w, worker = self._make(worker_env)
        calls = []
        monkeypatch.setattr(
            worker.current_platform, "is_cuda", lambda: True, raising=False
        )
        monkeypatch.setattr(
            worker.torch._C,
            "_accelerator_setAllocatorSettings",
            lambda setting: calls.append(setting),
            raising=False,
        )
        with w._scoped_allocator_max_split(20):
            assert calls == ["max_split_size_mb:20"]
        # Restore: PyTorch default is SIZE_MAX (no limit).
        _SIZE_MAX_MB = (2**64 - 1) // (1024 * 1024)
        assert calls[-1] == f"max_split_size_mb:{_SIZE_MAX_MB}"

    def test_cuda_platform_restores_env_value(self, worker_env, monkeypatch):
        """A pre-set conf value is restored verbatim on exit."""
        w, worker = self._make(worker_env)
        calls = []
        monkeypatch.setattr(
            worker.current_platform, "is_cuda", lambda: True, raising=False
        )
        monkeypatch.setattr(
            worker.torch._C,
            "_accelerator_setAllocatorSettings",
            lambda setting: calls.append(setting),
            raising=False,
        )
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        with w._scoped_allocator_max_split(20):
            pass
        assert calls[-1] == "max_split_size_mb:128"


# ---------------------------------------------------------------------------
# determine_available_memory
# ---------------------------------------------------------------------------


class TestDetermineAvailableMemory:
    """TXDA budget parsing and the profiled memory accounting path."""

    def _make(self, worker_env, vendor="nvidia", device="cuda", **cfg_overrides):
        worker, tree = worker_env(vendor_name=vendor, device_type=device)
        cfg = make_vllm_config(**cfg_overrides)
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        return w, worker, tree

    def test_txda_returns_configured_budget(self, worker_env, monkeypatch):
        """kv_cache_memory_bytes set on the cache config is returned as-is."""
        w, worker, tree = self._make(worker_env, vendor="sunrise", device="txda")
        w.cache_config.kv_cache_memory_bytes = 5 * (1 << 30)
        assert w.determine_available_memory() == 5 * (1 << 30)

    def test_txda_env_var_default(self, worker_env, monkeypatch):
        """A missing budget falls back to VLLM_TXDA_KV_CACHE_SIZE."""
        w, worker, tree = self._make(worker_env, vendor="sunrise", device="txda")
        w.cache_config.kv_cache_memory_bytes = None
        monkeypatch.setenv("VLLM_TXDA_KV_CACHE_SIZE", str(3 * (1 << 20)))
        assert w.determine_available_memory() == 3 * (1 << 20)

    def test_txda_env_var_invalid_rejected(self, worker_env, monkeypatch):
        """A non-integer env value raises the validation error."""
        w, worker, tree = self._make(worker_env, vendor="sunrise", device="txda")
        w.cache_config.kv_cache_memory_bytes = None
        monkeypatch.setenv("VLLM_TXDA_KV_CACHE_SIZE", "abc")
        with pytest.raises(ValueError, match="positive integer in bytes"):
            w.determine_available_memory()

    def test_txda_zero_budget_rejected(self, worker_env, monkeypatch):
        """A non-positive budget is rejected with the combined message."""
        w, worker, tree = self._make(worker_env, vendor="sunrise", device="txda")
        w.cache_config.kv_cache_memory_bytes = 0
        with pytest.raises(ValueError, match="must be positive"):
            w.determine_available_memory()

    def test_profile_run_with_explicit_budget(self, worker_env, monkeypatch):
        """An explicit kv budget still triggers the compile-only profile run."""
        w, worker, tree = self._make(worker_env)
        w.cache_config.kv_cache_memory_bytes = 2 * (1 << 30)
        w.model_runner = _RunnerDouble()
        w.init_snapshot = SimpleNamespace(free_memory=200, total_memory=256)
        monkeypatch.setattr(
            worker, "reserve_mm_ipc_gpu_memory", lambda budget, mm, api: budget + 7
        )
        assert w.determine_available_memory() == 2 * (1 << 30) + 7
        assert any(name == "profile_run" for name, _, _ in w.model_runner.calls)

    def test_profiled_path_memory_accounting(self, worker_env, monkeypatch):
        """The free-memory path subtracts non-KV usage from the request."""
        w, worker, tree = self._make(worker_env)
        w.model_runner = _RunnerDouble()
        w.model_runner.profile_cudagraph_memory = lambda: 0
        w.model_runner.model_memory_usage = 10
        w.init_snapshot = SimpleNamespace(free_memory=200, total_memory=256)
        w.requested_memory = 150

        # Startup plan + memory_profiling boundaries.
        monkeypatch.setattr(worker, "maybe_apply_startup_plan", lambda worker: None)
        monkeypatch.setattr(
            worker,
            "reserve_mm_ipc_gpu_memory",
            lambda budget, mm, api: budget,
        )

        class _ProfileResult:
            total_consumed = 30
            transient_peak_headroom = 20
            non_kv_cache_memory = 60
            after_profile = SimpleNamespace(free_memory=190)

        @contextlib.contextmanager
        def _memory_profiling(snapshot, weights_memory):
            yield _ProfileResult()

        monkeypatch.setattr(worker, "memory_profiling", _memory_profiling)
        # Independent oracle: 150 - 60 - 0 = 90.
        assert w.determine_available_memory() == 90
        assert w.total_consumed == 30
        assert w.peak_activation_memory == 20
        assert any(name == "profile_run" for name, _, _ in w.model_runner.calls)

    def test_profiled_path_subtracts_cudagraph_estimate(self, worker_env, monkeypatch):
        """The opt-in env flag applies the cudagraph estimate to the budget."""
        w, worker, tree = self._make(worker_env)
        w.vllm_config.compilation_config.cudagraph_mode = _CUDAGraphMode.FULL
        w.model_runner = _RunnerDouble()
        w.model_runner.profile_cudagraph_memory = lambda: 25
        w.model_runner.model_memory_usage = 10
        w.init_snapshot = SimpleNamespace(free_memory=200, total_memory=256)
        w.requested_memory = 150
        monkeypatch.setattr(worker, "maybe_apply_startup_plan", lambda worker: None)
        monkeypatch.setattr(
            worker, "reserve_mm_ipc_gpu_memory", lambda budget, mm, api: budget
        )

        class _ProfileResult:
            total_consumed = 30
            transient_peak_headroom = 20
            non_kv_cache_memory = 60
            after_profile = SimpleNamespace(free_memory=190)

        @contextlib.contextmanager
        def _memory_profiling(snapshot, weights_memory):
            yield _ProfileResult()

        monkeypatch.setattr(worker, "memory_profiling", _memory_profiling)
        monkeypatch.setattr(
            worker.envs, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", True
        )
        # Independent oracle: 150 - 60 - 25 = 65; peak adds the estimate too.
        assert w.determine_available_memory() == 65
        assert w.peak_activation_memory == 45

    def test_musa_logs_and_reports_exhausted_budget(
        self, worker_env, monkeypatch, caplog
    ):
        """A musa device with a drained budget logs the diagnostic."""
        w, worker, tree = self._make(worker_env, vendor="musa", device="musa")
        w.device = SimpleNamespace(type="musa")
        w.model_runner = _RunnerDouble()
        w.model_runner.profile_cudagraph_memory = lambda: 0
        w.model_runner.model_memory_usage = 10
        w.init_snapshot = SimpleNamespace(free_memory=200, total_memory=256)
        w.requested_memory = 150
        monkeypatch.setattr(worker, "maybe_apply_startup_plan", lambda worker: None)
        monkeypatch.setattr(
            worker, "reserve_mm_ipc_gpu_memory", lambda budget, mm, api: budget
        )

        class _ProfileResult:
            total_consumed = 30
            transient_peak_headroom = 20
            non_kv_cache_memory = 160
            after_profile = SimpleNamespace(free_memory=190)

        @contextlib.contextmanager
        def _memory_profiling(snapshot, weights_memory):
            yield _ProfileResult()

        monkeypatch.setattr(worker, "memory_profiling", _memory_profiling)
        monkeypatch.setattr(
            worker.current_platform,
            "torch_device_fn",
            SimpleNamespace(mem_get_info=lambda device: (100, 256)),
            raising=False,
        )
        with caplog.at_level(logging.ERROR):
            result = w.determine_available_memory()
        # Independent oracle: 150 - 160 - 0 = -10 (reported as-is).
        assert result == -10
        assert any("budget exhausted" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# initialize_from_config / compile_or_warm_up_model
# ---------------------------------------------------------------------------


class TestInitializeFromConfig:
    """KV cache allocation with connector init and optional extras."""

    def _make(self, worker_env):
        worker, tree = worker_env()
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        w.model_runner = _RunnerDouble()
        return w, worker, tree

    @staticmethod
    def _kv_config(**overrides):
        kv = SimpleNamespace(num_blocks=17, needs_kv_cache_zeroing=False)
        for key, value in overrides.items():
            setattr(kv, key, value)
        return kv

    def test_full_flow(self, worker_env):
        """Block count lands on the cache config before runner allocation."""
        w, worker, tree = self._make(worker_env)
        kv = self._kv_config()
        w.initialize_from_config(kv)
        assert w.cache_config.num_gpu_blocks == 17
        assert ("ensure_kv_transfer_initialized", (w.vllm_config, kv), {}) in tree.calls
        assert ("initialize_kv_cache", (kv,), {}) in w.model_runner.calls

    def test_routed_experts_capturer_initialized(self, worker_env):
        """enable_return_routed_experts triggers the capturer init."""
        w, worker, tree = self._make(worker_env)
        w.model_config.enable_return_routed_experts = True
        w.initialize_from_config(self._kv_config())
        assert any(
            name == "init_routed_experts_capturer"
            for name, _, _ in w.model_runner.calls
        )

    def test_kv_zero_meta_initialized(self, worker_env):
        """needs_kv_cache_zeroing drives the runner's zero-metadata init."""
        w, worker, tree = self._make(worker_env)
        # The runner double raises AttributeError on _init_kv_zero_meta, so
        # give it a real one.
        w.model_runner._init_kv_zero_meta = lambda: w.model_runner.calls.append(
            ("_init_kv_zero_meta", (), {})
        )
        w.initialize_from_config(self._kv_config(needs_kv_cache_zeroing=True))
        assert ("_init_kv_zero_meta", (), {}) in w.model_runner.calls


class TestCompileOrWarmUpModel:
    """Warmup sizes, kernel warmup gates and V1/V2 sampler priming."""

    def _make(self, worker_env, **cfg_overrides):
        worker, tree = worker_env()
        cfg = make_vllm_config(**cfg_overrides)
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        w.model_runner = _RunnerDouble()
        w.device = SimpleNamespace(type="cuda")
        w.scheduler_config = cfg.scheduler_config
        w.observability_config = SimpleNamespace(
            jit_monitor_mode="off", jit_monitor_verbose=False
        )
        return w, worker, tree

    def _patch_jit_boundaries(self, worker, monkeypatch, activated):
        """Patch the jit monitor and inductor lazy-init leaves."""
        jit_mod = ModuleType("vllm.utils.jit_monitor")
        jit_mod.activate = lambda **kw: activated.append(kw)
        monkeypatch.setitem(sys.modules, "vllm.utils.jit_monitor", jit_mod)
        comp_iface = ModuleType("vllm.compilation.compiler_interface")
        comp_iface.trigger_inductor_lazy_init = lambda device: activated.append(
            ("inductor", device)
        )
        monkeypatch.setitem(
            sys.modules, "vllm.compilation.compiler_interface", comp_iface
        )

    def test_v2_flow_runs_warmup_kernels(self, worker_env, monkeypatch):
        """The V2 path primes triton kernels through warmup_kernels."""
        w, worker, tree = self._make(worker_env)
        activated = []
        self._patch_jit_boundaries(worker, monkeypatch, activated)
        result = w.compile_or_warm_up_model()
        assert ("capture_model", (), {}) in w.model_runner.calls
        assert any(name == "warmup_kernels" for name, _, _ in tree.calls)
        assert ("set_random_seed", (42,), {}) in tree.calls
        # CompilationTimes carries the config's recorded durations.
        assert result.language_model == 1.5
        assert result.encoder == 0.5

    def test_warmup_sizes_exclude_cudagraph_capture_sizes(
        self, worker_env, monkeypatch
    ):
        """compile_sizes minus cudagraph capture sizes are dummy-run."""
        w, worker, tree = self._make(worker_env)
        w.vllm_config.compilation_config.mode = worker.CompilationMode.VLLM_COMPILE
        w.vllm_config.compilation_config.compile_sizes = [1, 2, 8]
        w.vllm_config.compilation_config.cudagraph_mode = worker.CUDAGraphMode.FULL
        w.vllm_config.compilation_config.cudagraph_capture_sizes = [8]
        activated = []
        self._patch_jit_boundaries(worker, monkeypatch, activated)
        w.compile_or_warm_up_model()
        dummy_runs = [c for c in w.model_runner.calls if c[0] == "_dummy_run"]
        # Independent oracle: warmup sizes {1,2} (8 is captured by graphs),
        # run in descending order with skip_eplb and loras intact.
        assert [c[1][0] for c in dummy_runs] == [2, 1]
        for _, args, kwargs in dummy_runs:
            assert kwargs == {"skip_eplb": True, "remove_lora": False}

    def test_warmup_sizes_add_compile_range_ends(self, worker_env, monkeypatch):
        """An uncovered compile range contributes its end size."""
        w, worker, tree = self._make(worker_env)
        w.vllm_config.compilation_config.mode = worker.CompilationMode.VLLM_COMPILE
        w.vllm_config.compilation_config.compile_sizes = [2]
        w.vllm_config.compilation_config.cudagraph_mode = worker.CUDAGraphMode.NONE

        class _CompileRange:
            """Range double: membership via a real range plus an .end."""

            def __init__(self, start, end):
                self._range = range(start, end)
                self.end = end

            def __contains__(self, value):
                return value in self._range

        w.vllm_config.compilation_config.get_compile_ranges = lambda: [
            _CompileRange(1, 5),
            _CompileRange(6, 9),
        ]
        activated = []
        self._patch_jit_boundaries(worker, monkeypatch, activated)
        w.compile_or_warm_up_model()
        dummy_runs = [c for c in w.model_runner.calls if c[0] == "_dummy_run"]
        # Independent oracle: 2 covers range(1,5); range(6,9) is uncovered so
        # its end (9) is warmed. Descending order: [9, 2].
        assert [c[1][0] for c in dummy_runs] == [9, 2]

    def test_txda_skips_generic_kernel_warmup(self, worker_env, monkeypatch, caplog):
        """txda devices replace the generic kernel warmup with a warning."""
        worker, tree = worker_env(vendor_name="sunrise", device_type="txda")
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        w.model_runner = _RunnerDouble()
        w.device = SimpleNamespace(type="txda")
        w.observability_config = SimpleNamespace(
            jit_monitor_mode="off", jit_monitor_verbose=False
        )
        activated = []
        self._patch_jit_boundaries(worker, monkeypatch, activated)
        with caplog.at_level(logging.WARNING):
            w.compile_or_warm_up_model()
        assert not any(name == "kernel_warmup" for name, _, _ in tree.calls)
        assert any(
            "skipping generic kernel_warmup" in rec.message for rec in caplog.records
        )

    def test_kunlunxin_skips_generic_kernel_warmup(self, worker_env, monkeypatch):
        """kunlunxin platforms skip the generic warmup like txda."""
        worker, tree = worker_env(vendor_name="kunlunxin", device_type="xpu")
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        w.model_runner = _RunnerDouble()
        w.device = SimpleNamespace(type="xpu")
        w.observability_config = SimpleNamespace(
            jit_monitor_mode="off", jit_monitor_verbose=False
        )
        activated = []
        self._patch_jit_boundaries(worker, monkeypatch, activated)
        w.compile_or_warm_up_model()
        assert not any(name == "kernel_warmup" for name, _, _ in tree.calls)

    def test_kernel_warmup_import_error_is_swallowed(self, worker_env, monkeypatch):
        """A missing optional warmup dep degrades to a warning."""
        w, worker, tree = self._make(worker_env)
        activated = []
        self._patch_jit_boundaries(worker, monkeypatch, activated)

        def _boom(worker):
            raise ImportError("torchvision missing")

        monkeypatch.setattr(worker, "kernel_warmup", _boom)
        w.compile_or_warm_up_model()
        # The rest of the flow still ran.
        assert ("capture_model", (), {}) in w.model_runner.calls

    def test_v1_sampler_warmup_on_last_rank(self, worker_env, monkeypatch):
        """V1 last-rank workers prime the sampler with a dummy batch."""
        w, worker, tree = self._make(worker_env, use_v2_model_runner=False)
        pp = _PPGroupStub(is_first_rank=True, is_last_rank=True)
        monkeypatch.setattr(worker, "get_pp_group", lambda: pp)
        activated = []
        self._patch_jit_boundaries(worker, monkeypatch, activated)
        w.model_runner.is_pooling_model = False
        w.model_runner.capture_model = lambda: 0
        w.compile_or_warm_up_model()
        # Independent oracle: min(max_num_seqs=4, max_num_batched_tokens=8) = 4.
        dummy_runs = [c for c in w.model_runner.calls if c[0] == "_dummy_run"]
        assert dummy_runs[-1][2]["num_tokens"] == 4
        assert dummy_runs[-1][2]["skip_eplb"] is True
        assert any(name == "_dummy_sampler_run" for name, _, _ in w.model_runner.calls)
        assert not any(
            name == "_dummy_pooler_run" for name, _, _ in w.model_runner.calls
        )

    def test_eager_mode_skips_capture(self, worker_env, monkeypatch):
        """enforce_eager skips graph capture entirely."""
        w, worker, tree = self._make(worker_env)
        w.model_config.enforce_eager = True
        activated = []
        self._patch_jit_boundaries(worker, monkeypatch, activated)
        result = w.compile_or_warm_up_model()
        assert not any(name == "capture_model" for name, _, _ in w.model_runner.calls)
        assert result.language_model == 1.5

    def test_npu_device_keeps_thread_config(self, worker_env, monkeypatch):
        """npu devices skip the set_torch_threads_for_runtime call."""
        w, worker, tree = self._make(worker_env)
        w.device = SimpleNamespace(type="npu")
        activated = []
        self._patch_jit_boundaries(worker, monkeypatch, activated)
        w.compile_or_warm_up_model()
        assert not any(
            name == "set_torch_threads_for_runtime" for name, _, _ in tree.calls
        )

    def test_jit_monitor_gated_on_jit_warmup_flag(self, worker_env, monkeypatch):
        """Platforms allowing runtime JIT only monitor when warmup is on."""
        worker, tree = worker_env(vendor_name="kunlunxin", device_type="xpu")
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        w.model_runner = _RunnerDouble()
        w.device = SimpleNamespace(type="xpu")
        w.observability_config = SimpleNamespace(
            jit_monitor_mode="off", jit_monitor_verbose=False
        )
        activated = []
        self._patch_jit_boundaries(worker, monkeypatch, activated)
        w.vllm_config.kernel_config.enable_jit_warmup = False
        w.compile_or_warm_up_model()
        assert activated == []


# ---------------------------------------------------------------------------
# shutdown / elastic ep / ft command / delegation methods
# ---------------------------------------------------------------------------


class TestShutdown:
    """Teardown ordering across gc, transfer engines and the allocator."""

    def _make(self, worker_env):
        worker, tree = worker_env()
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        w.model_runner = _RunnerDouble()
        return w, worker, tree

    def test_shutdown_order(self, worker_env, monkeypatch):
        """gc unfreezes first, then engines, profiler and runner stop."""
        w, worker, tree = self._make(worker_env)
        order = []
        monkeypatch.setattr(worker.gc, "unfreeze", lambda: order.append("gc"))
        monkeypatch.setattr(
            worker, "ensure_kv_transfer_shutdown", lambda: order.append("kv")
        )
        monkeypatch.setattr(
            worker, "ensure_ec_transfer_shutdown", lambda: order.append("ec")
        )
        w.profiler = SimpleNamespace(shutdown=lambda: order.append("profiler"))
        w.weight_transfer_engine = SimpleNamespace(
            shutdown=lambda: order.append("weight_engine")
        )
        w.model_runner.shutdown = lambda: order.append("runner")
        # CuMem release runs after the runner on cuda-like platforms.
        released = []
        cumem_mod = ModuleType("vllm.device_allocator.cumem")

        class _CuMemAllocator:
            instance = SimpleNamespace(release_pools=lambda: released.append(True))

        cumem_mod.CuMemAllocator = _CuMemAllocator
        monkeypatch.setitem(sys.modules, "vllm.device_allocator.cumem", cumem_mod)
        w.shutdown()
        assert order == ["gc", "kv", "ec", "profiler", "weight_engine", "runner"]
        assert released == [True]

    def test_shutdown_without_optionals(self, worker_env, monkeypatch):
        """Missing profiler/engine/runner still shut down cleanly."""
        w, worker, tree = self._make(worker_env)
        monkeypatch.setattr(worker.gc, "unfreeze", lambda: None)
        monkeypatch.setattr(worker, "ensure_kv_transfer_shutdown", lambda: None)
        monkeypatch.setattr(worker, "ensure_ec_transfer_shutdown", lambda: None)
        w.profiler = None
        w.weight_transfer_engine = None
        w.model_runner = None
        w.shutdown()  # must not raise

    def test_shutdown_releases_cumem_pools_on_cuda(self, worker_env, monkeypatch):
        """CUDA-like platforms release the CuMem allocator pools."""
        w, worker, tree = self._make(worker_env)
        monkeypatch.setattr(worker.gc, "unfreeze", lambda: None)
        monkeypatch.setattr(worker, "ensure_kv_transfer_shutdown", lambda: None)
        monkeypatch.setattr(worker, "ensure_ec_transfer_shutdown", lambda: None)
        released = []
        cumem_mod = ModuleType("vllm.device_allocator.cumem")

        class _CuMemAllocator:
            instance = SimpleNamespace(release_pools=lambda: released.append(True))

        cumem_mod.CuMemAllocator = _CuMemAllocator
        monkeypatch.setitem(sys.modules, "vllm.device_allocator.cumem", cumem_mod)
        w.shutdown()
        assert released == [True]

    def test_shutdown_skips_cumem_on_non_cuda(self, worker_env, monkeypatch):
        """Non-CUDA platforms leave the CuMem allocator alone."""
        worker, tree = worker_env(vendor_name="ascend", device_type="npu")
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        w.model_runner = _RunnerDouble()
        monkeypatch.setattr(worker.gc, "unfreeze", lambda: None)
        monkeypatch.setattr(worker, "ensure_kv_transfer_shutdown", lambda: None)
        monkeypatch.setattr(worker, "ensure_ec_transfer_shutdown", lambda: None)
        real_import = builtins.__import__

        def _fail_import(name, *args, **kwargs):
            if "cumem" in str(name):
                raise AssertionError("cumem import must not happen")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _fail_import)
        w.shutdown()

    def test_shutdown_tolerates_none_shutdown_handles(self, worker_env, monkeypatch):
        """Interpreter-shutdown None handles are skipped, not called."""
        w, worker, tree = self._make(worker_env)
        monkeypatch.setattr(worker.gc, "unfreeze", lambda: None)
        monkeypatch.setattr(worker, "ensure_kv_transfer_shutdown", None)
        monkeypatch.setattr(worker, "ensure_ec_transfer_shutdown", None)
        w.shutdown()  # must not raise


class TestElasticEpExecute:
    """elastic_ep_execute delegates to the scaling executor."""

    def test_delegates_method_name_and_args(self, worker_env):
        """The method string, args and kwargs reach the executor."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        seen = []
        w.elastic_ep_executor = SimpleNamespace(
            execute=lambda method, *a, **k: (seen.append((method, a, k)), "result")[1]
        )
        out = w.elastic_ep_execute("forward", 1, 2, key="v")
        assert out == "result"
        assert seen == [("forward", (1, 2), {"key": "v"})]


class TestHandleFtCommand:
    """Fault-tolerance command routing through the worker sentinel."""

    def test_delegates_to_sentinel(self, worker_env):
        """The command reaches the sentinel and its answer returns."""
        worker, tree = worker_env()
        cfg = make_vllm_config()
        w = worker.WorkerFL(
            vllm_config=cfg, local_rank=0, rank=0, distributed_init_method="tcp://x:1"
        )
        handled = []
        w.worker_sentinel = SimpleNamespace(
            handle_command=lambda req: handled.append(req) or "ok"
        )
        assert w.handle_ft_command("REQ") == "ok"
        assert handled == ["REQ"]


class TestDelegationMethods:
    """One-line forwarders onto the model runner."""

    def _make(self, worker_env):
        worker, tree = worker_env()
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        w.model_runner = _RunnerDouble()
        return w, worker

    def test_update_config(self, worker_env):
        """update_config forwards the overrides dict."""
        w, _ = self._make(worker_env)
        w.update_config({"max_num_seqs": 3})
        assert ("update_config", ({"max_num_seqs": 3},), {}) in w.model_runner.calls

    def test_reload_weights(self, worker_env):
        """reload_weights forwards args under the current config context."""
        w, _ = self._make(worker_env)
        w.reload_weights("path")
        assert ("reload_weights", ("path",), {}) in w.model_runner.calls

    def test_get_model_returns_runner_model(self, worker_env):
        """get_model returns the runner's model object."""
        w, _ = self._make(worker_env)
        assert w.get_model() is w.model_runner.model

    def test_get_draft_model_returns_runner_draft(self, worker_env):
        """get_draft_model returns the runner's draft model."""
        w, _ = self._make(worker_env)
        draft = SimpleNamespace(draft=True)
        w.model_runner.draft_model = draft
        assert w.get_draft_model() is draft

    def test_get_kv_cache_spec(self, worker_env):
        """get_kv_cache_spec forwards to the runner."""
        w, _ = self._make(worker_env)
        w.get_kv_cache_spec()
        assert ("get_kv_cache_spec", (), {}) in w.model_runner.calls

    def test_update_max_model_len(self, worker_env, caplog):
        """max_model_len updates both the config and the runner."""
        w, worker = self._make(worker_env)
        with caplog.at_level(logging.DEBUG):
            w.update_max_model_len(512)
        assert w.model_config.max_model_len == 512
        assert ("update_max_model_len", (512,), {}) in w.model_runner.calls
        assert any("512" in rec.message for rec in caplog.records)

    def test_update_max_model_len_without_runner(self, worker_env):
        """A missing runner only updates the cached config."""
        w, worker = self._make(worker_env)
        w.model_runner = None
        w.update_max_model_len(256)
        assert w.model_config.max_model_len == 256

    def test_get_supported_tasks(self, worker_env):
        """get_supported_tasks forwards to the runner."""
        w, _ = self._make(worker_env)
        w.get_supported_tasks()
        assert ("get_supported_tasks", (), {}) in w.model_runner.calls

    def test_get_encoder_timing_stats(self, worker_env):
        """get_encoder_timing_stats forwards to the runner."""
        w, _ = self._make(worker_env)
        w.get_encoder_timing_stats()
        assert ("get_encoder_timing_stats", (), {}) in w.model_runner.calls

    def test_reset_mm_cache(self, worker_env):
        """reset_mm_cache forwards to the runner."""
        w, _ = self._make(worker_env)
        w.reset_mm_cache()
        assert ("reset_mm_cache", (), {}) in w.model_runner.calls

    def test_reset_encoder_cache(self, worker_env):
        """reset_encoder_cache forwards to the runner."""
        w, _ = self._make(worker_env)
        w.reset_encoder_cache()
        assert ("reset_encoder_cache", (), {}) in w.model_runner.calls

    def test_take_draft_token_ids(self, worker_env):
        """take_draft_token_ids forwards to the runner."""
        w, _ = self._make(worker_env)
        w.take_draft_token_ids()
        assert ("take_draft_token_ids", (), {}) in w.model_runner.calls

    def test_execute_dummy_batch(self, worker_env):
        """execute_dummy_batch dummy-runs the uniform decode length."""
        w, _ = self._make(worker_env)
        w.execute_dummy_batch()
        runs = [c for c in w.model_runner.calls if c[0] == "_dummy_run"]
        # Independent oracle: uniform_decode_query_len defaults to 3.
        assert runs == [("_dummy_run", (3,), {"uniform_decode": True})]

    def test_execute_dummy_batch_without_query_len(self, worker_env):
        """A runner without uniform_decode_query_len uses one token."""
        w, _ = self._make(worker_env)
        recorded = []

        class _BareRunner:
            """Runner double lacking the query-length attribute."""

            def _dummy_run(self, *args, **kwargs):
                recorded.append((args, kwargs))

        w.model_runner = _BareRunner()
        w.execute_dummy_batch()
        # Independent oracle: the getattr default is 1.
        assert recorded == [((1,), {"uniform_decode": True})]

    def test_lora_methods_forward(self, worker_env):
        """add/remove/list/pin lora all reach the runner."""
        w, _ = self._make(worker_env)
        w.add_lora("req")
        w.remove_lora(7)
        w.list_loras()
        w.pin_lora(9)
        names = [c[0] for c in w.model_runner.calls]
        assert names == ["add_lora", "remove_lora", "list_loras", "pin_lora"]

    def test_check_health_is_noop(self, worker_env):
        """check_health always succeeds while the worker is alive."""
        w, _ = self._make(worker_env)
        assert w.check_health() is None

    def test_save_sharded_state(self, worker_env, monkeypatch):
        """save_sharded_state runs the ShardedStateLoader boundary."""
        w, worker = self._make(worker_env)
        saved = []

        class _Loader:
            @staticmethod
            def save_model(model, path, pattern=None, max_size=None):
                saved.append((model, path, pattern, max_size))

        loader_mod = sys.modules["vllm.model_executor.model_loader"]
        monkeypatch.setattr(loader_mod, "ShardedStateLoader", _Loader)
        w.save_sharded_state("/tmp/x", pattern="p", max_size=5)
        assert saved == [(w.model_runner.model, "/tmp/x", "p", 5)]

    def test_save_tensorized_model(self, worker_env, monkeypatch):
        """save_tensorized_model runs the TensorizerLoader boundary."""
        w, worker = self._make(worker_env)
        saved = []

        class _Loader:
            @staticmethod
            def save_model(model, tensorizer_config=None, model_config=None):
                saved.append((model, tensorizer_config, model_config))

        # TensorizerLoader is bound by name at import time.
        monkeypatch.setattr(worker, "TensorizerLoader", _Loader)
        w.save_tensorized_model("tconf")
        assert saved == [(w.model_runner.model, "tconf", w.model_config)]

    def test_get_compilation_match_table(self, worker_env, monkeypatch):
        """get_compilation_match_table reads the inductor pass registry."""
        w, worker = self._make(worker_env)
        pass_mod = ModuleType("vllm.compilation.passes.vllm_inductor_pass")
        pass_mod.get_match_table = lambda: {"a": 1}
        monkeypatch.setitem(
            sys.modules, "vllm.compilation.passes.vllm_inductor_pass", pass_mod
        )
        assert w.get_compilation_match_table() == {"a": 1}


class TestGetKvConnectorHandshakeMetadata:
    """Optional KV-connector handshake metadata keyed by (pp, tp) ranks."""

    def _make(self, worker_env):
        worker, tree = worker_env()
        w = worker.WorkerFL(
            vllm_config=make_vllm_config(),
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://x:1",
        )
        return w, worker, tree

    def test_no_group_returns_none(self, worker_env, monkeypatch):
        """Without a KV transfer group there is no metadata."""
        w, worker, tree = self._make(worker_env)
        monkeypatch.setattr(worker, "has_kv_transfer_group", lambda: False)
        assert w.get_kv_connector_handshake_metadata() is None

    def test_connector_without_metadata_returns_none(self, worker_env, monkeypatch):
        """Connectors opting out of handshakes return None."""
        w, worker, tree = self._make(worker_env)
        monkeypatch.setattr(worker, "has_kv_transfer_group", lambda: True)
        monkeypatch.setattr(
            worker,
            "get_kv_transfer_group",
            lambda: SimpleNamespace(get_handshake_metadata=lambda: None),
        )
        assert w.get_kv_connector_handshake_metadata() is None

    def test_metadata_keyed_by_ranks(self, worker_env, monkeypatch):
        """The dict key pairs the PP and TP group ranks."""
        w, worker, tree = self._make(worker_env)
        monkeypatch.setattr(worker, "has_kv_transfer_group", lambda: True)
        metadata = SimpleNamespace(meta=True)
        monkeypatch.setattr(
            worker,
            "get_kv_transfer_group",
            lambda: SimpleNamespace(get_handshake_metadata=lambda: metadata),
        )
        monkeypatch.setattr(
            worker,
            "get_pp_group",
            lambda: SimpleNamespace(rank_in_group=2),
        )
        monkeypatch.setattr(
            worker,
            "get_tp_group",
            lambda: SimpleNamespace(rank_in_group=1),
        )
        assert w.get_kv_connector_handshake_metadata() == {(2, 1): metadata}
