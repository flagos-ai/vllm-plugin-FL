# SPDX-License-Identifier: Apache-2.0
"""Residual real-import contracts for vllm_fl.worker.worker branch arms.

Complements w4a/w4b/w16/w17.  The real ``vllm_fl.worker.worker`` module is
imported (no AST splicing); vLLM/distributed helpers that need an accelerator or
a process group are replaced with explicit doubles.  Instances are built with
``object.__new__`` so no base ``__init__`` and no device are touched.
"""

import contextlib
import sys
import types
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm_fl.worker.worker as w


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _ensure_pkg(monkeypatch, name):
    existing = sys.modules.get(name)
    if isinstance(existing, types.ModuleType):
        return existing
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure_pkg(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    return module


def _install(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure_pkg(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def make_worker(**attrs):
    obj = object.__new__(w.WorkerFL)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


def _platform(**over):
    base = dict(
        device_type="cuda",
        vendor_name="",
        dist_backend="nccl",
        is_rocm=lambda: False,
        is_cuda_alike=lambda: True,
        is_cuda=lambda: True,
        is_xpu=lambda: False,
        is_cpu=lambda: False,
        is_sleep_mode_available=lambda: True,
        logical_device_id_to_visible_device_id=lambda index: index,
        set_device=lambda device: None,
        check_if_supports_dtype=lambda dtype: None,
    )
    base.update(over)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# module-level helpers
# --------------------------------------------------------------------------- #


def test_is_hybrid_kv_model_falls_back_when_is_hybrid_raises():
    class _Raising:
        @property
        def is_hybrid(self):
            raise RuntimeError("not resolvable yet")

    model_config = _Raising()
    model_config.hf_text_config = SimpleNamespace(
        layer_types=["linear_attention", "full_attention"]
    )
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        model_config=model_config,
    )
    assert w._is_hybrid_kv_model(cfg) is True


def test_is_hybrid_kv_model_only_full_attention_is_not_hybrid():
    class _Raising:
        @property
        def is_hybrid(self):
            raise RuntimeError("boom")

    model_config = _Raising()
    model_config.hf_text_config = SimpleNamespace(layer_types=["full_attention"])
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        model_config=model_config,
    )
    assert w._is_hybrid_kv_model(cfg) is False


def test_install_v2_runner_compat_falls_through_when_is_available_raises(monkeypatch):
    rec = []
    fake_cuda = SimpleNamespace(
        is_available=Mock(side_effect=RuntimeError("cuda stub")),
    )
    fake_torch = SimpleNamespace(
        cuda=fake_cuda,
        musa=SimpleNamespace(Stream="MUSA_STREAM"),
    )
    monkeypatch.setattr(w, "torch", fake_torch)
    monkeypatch.setattr(w, "_install_v2_dispatch_warmup", lambda: rec.append("warmup"))
    buffer_utils = _install(monkeypatch, "vllm.v1.worker.gpu.buffer_utils")

    w._install_v2_runner_compat()

    assert fake_cuda.Stream == "MUSA_STREAM"
    assert buffer_utils.is_uva_available() is False
    assert rec == ["warmup"]


def test_install_v2_runner_compat_returns_when_cuda_available(monkeypatch):
    rec = []
    fake_cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(
        w, "torch", SimpleNamespace(cuda=fake_cuda, musa=SimpleNamespace(Stream="MUSA"))
    )
    monkeypatch.setattr(w, "_install_v2_dispatch_warmup", lambda: rec.append("warmup"))

    w._install_v2_runner_compat()

    assert rec == []


# --------------------------------------------------------------------------- #
# init_device residual branches
# --------------------------------------------------------------------------- #


class _Dev:
    def __init__(self, spec):
        self.spec = spec
        self.type = spec.split(":")[0]
        self.index = int(spec.split(":")[1])


def _parallel(**over):
    base = dict(
        assigned_physical_gpu_ids=None,
        local_world_size=1,
        distributed_executor_backend="mp",
        data_parallel_backend=None,
        nnodes_within_dp=1,
        data_parallel_rank_local=None,
        data_parallel_index=0,
        enable_dbo=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def init_env(monkeypatch):
    rec = []
    fake_torch = SimpleNamespace(
        device=lambda spec: _Dev(spec),
        accelerator=SimpleNamespace(
            device_count=lambda: 8, empty_cache=lambda: rec.append("empty")
        ),
    )
    monkeypatch.setattr(w, "torch", fake_torch)
    monkeypatch.setattr(w, "_apply_worker_vendor_patches", lambda: rec.append("vendor"))
    monkeypatch.setattr(
        w, "init_worker_distributed_environment", lambda *a: rec.append(("dist", a))
    )
    monkeypatch.setattr(w, "set_random_seed", lambda seed: rec.append(("seed", seed)))
    monkeypatch.setattr(w, "request_memory", lambda snapshot, cache: 111)
    monkeypatch.setattr(w, "report_usage_stats", lambda cfg: rec.append("usage"))
    monkeypatch.setattr(w, "MemorySnapshot", lambda device: SimpleNamespace(device=device))
    monkeypatch.setattr(
        w,
        "init_workspace_manager",
        lambda device, ub, lanes: rec.append(("workspace", ub, lanes)),
    )
    monkeypatch.setattr(w, "_num_workspace_lanes", lambda cfg, v2: 3)
    monkeypatch.setattr(w, "gc", SimpleNamespace(collect=lambda: rec.append("collect")))
    return rec, fake_torch


def _worker(parallel, **over):
    vllm_config = SimpleNamespace(parallel_config=parallel)
    return make_worker(
        device_config=SimpleNamespace(device_type=over.pop("config_device_type", "cuda")),
        parallel_config=parallel,
        local_rank=over.pop("local_rank", 0),
        rank=over.pop("rank", 0),
        distributed_init_method="tcp://127.0.0.1:1",
        model_config=SimpleNamespace(dtype="float16", seed=7),
        cache_config=SimpleNamespace(),
        vllm_config=vllm_config,
        use_v2_model_runner=over.pop("use_v2", False),
    )


def test_init_device_ray_backend_skips_local_world_size_assert(monkeypatch, init_env):
    rec, _fake = init_env
    monkeypatch.setattr(w, "current_platform", _platform())
    published = []
    _install(
        monkeypatch,
        "vllm.platforms.interface",
        set_assigned_physical_gpu_ids=lambda ids: published.append(list(ids)),
    )
    _install(monkeypatch, "vllm_fl.worker.model_runner", ModelRunnerFL=lambda cfg, dev: object())

    # local_world_size (3) exceeds the assigned list (2); the ray backend must
    # skip that bound check while still publishing the mapping.
    parallel = _parallel(
        assigned_physical_gpu_ids=[5, 6],
        local_world_size=3,
        distributed_executor_backend="ray",
    )
    worker = _worker(parallel, local_rank=0)
    worker.init_device()
    assert published == [[5, 6]]
    assert worker.local_rank == 0


def test_init_device_unassigned_rank_out_of_bounds_raises(monkeypatch, init_env):
    rec, fake_torch = init_env
    fake_torch.accelerator = SimpleNamespace(
        device_count=lambda: 1, empty_cache=lambda: rec.append("empty")
    )
    monkeypatch.setattr(w, "current_platform", _platform())
    parallel = _parallel(distributed_executor_backend="ray")
    worker = _worker(parallel, local_rank=5)
    with pytest.raises(AssertionError, match="out of bounds"):
        worker.init_device()


def test_init_device_v2_musa_installs_compat(monkeypatch, init_env):
    rec, _fake = init_env
    called = []
    monkeypatch.setattr(w, "_install_v2_runner_compat", lambda: called.append("compat"))
    _install(
        monkeypatch,
        "vllm.v1.worker.gpu.model_runner",
        GPUModelRunner=lambda cfg, dev: ("musa-runner", dev),
    )
    monkeypatch.setattr(w, "current_platform", _platform(device_type="musa", vendor_name="mthreads"))

    worker = _worker(_parallel(), use_v2=True, config_device_type="musa")
    worker.init_device()
    assert called == ["compat"]
    assert worker.model_runner[0] == "musa-runner"


def test_init_device_v2_generic_vendor_uses_upstream_runner(monkeypatch, init_env):
    rec, _fake = init_env
    _install(
        monkeypatch,
        "vllm.v1.worker.gpu.model_runner",
        GPUModelRunner=lambda cfg, dev: ("upstream", dev),
    )
    monkeypatch.setattr(w, "current_platform", _platform(device_type="cuda", vendor_name="nvidia"))

    worker = _worker(_parallel(), use_v2=True)
    worker.init_device()
    assert worker.model_runner[0] == "upstream"


# --------------------------------------------------------------------------- #
# load_model / memory profiling residual branches
# --------------------------------------------------------------------------- #


def test_load_model_npu_sets_torch_threads(monkeypatch):
    rec = []
    calls = []
    monkeypatch.setattr(w, "set_current_vllm_config", lambda cfg: contextlib.nullcontext())
    monkeypatch.setattr(w, "set_torch_threads_for_runtime", lambda: rec.append("threads"))
    model_runner = SimpleNamespace(load_model=lambda load_dummy_weights: calls.append(load_dummy_weights))
    worker = make_worker(
        model_runner=model_runner,
        device=SimpleNamespace(type="npu"),
        vllm_config=SimpleNamespace(weight_transfer_config=None, model_config=SimpleNamespace()),
    )
    worker._maybe_get_memory_pool_context = lambda tag: contextlib.nullcontext()
    worker._scoped_allocator_max_split = lambda max_split_size_mb: contextlib.nullcontext()
    worker.load_model()
    assert calls == [False]
    assert rec == ["threads"]


def test_determine_available_memory_amd_defaults_api_process_count(monkeypatch):
    from vllm.config import CUDAGraphMode

    rec = []

    @contextlib.contextmanager
    def _profiling(*args, **kwargs):
        rec.append("profiling")
        yield SimpleNamespace(
            total_consumed=5,
            transient_peak_headroom=3,
            non_kv_cache_memory=10,
            after_profile=SimpleNamespace(free_memory=100),
        )

    import vllm_fl.dispatch as dispatch_pkg

    monkeypatch.setattr(
        dispatch_pkg, "warmup_cached_ops", lambda **kwargs: rec.append(("warm", kwargs)), raising=False
    )
    monkeypatch.setattr(w, "memory_profiling", _profiling)
    monkeypatch.setattr(w, "maybe_apply_startup_plan", lambda worker: rec.append("startup_plan"))
    monkeypatch.setattr(
        w,
        "reserve_mm_ipc_gpu_memory",
        lambda memory, mm, api: rec.append(("reserve", memory, mm, api)) or "kv",
    )
    monkeypatch.setattr(w, "current_platform", _platform(device_type="cuda", vendor_name="amd"))
    monkeypatch.setattr(w.envs, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", True, raising=False)

    worker = make_worker(
        cache_config=SimpleNamespace(kv_cache_memory_bytes=None, gpu_memory_utilization=0.9),
        model_config=SimpleNamespace(multimodal_config="mm"),
        parallel_config=SimpleNamespace(),
        init_snapshot=SimpleNamespace(free_memory=200, total_memory=400),
        requested_memory=150,
        use_v2_model_runner=False,
        model_runner=SimpleNamespace(
            profile_run=lambda: rec.append("profile_run"),
            model_memory_usage=10,
            profile_cudagraph_memory=lambda: 10,
        ),
        device=SimpleNamespace(type="cuda"),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
        ),
    )
    assert worker.determine_available_memory() == "kv"
    assert ("warm", {"amd_compile": True}) in rec
    # requested 150 - non_kv 10 - graph 10 = 130, api count falls back to 1.
    assert ("reserve", 130, "mm", 1) in rec


# --------------------------------------------------------------------------- #
# execute_model residual branches
# --------------------------------------------------------------------------- #


def test_execute_model_zero_tokens_skips_pp_receive(monkeypatch):
    calls = []

    def _recv(**kwargs):
        raise AssertionError("irecv must not run without a forward pass")

    monkeypatch.setattr(
        w, "get_pp_group", lambda: SimpleNamespace(is_first_rank=False, irecv_tensor_dict=_recv)
    )
    worker = make_worker(
        model_runner=SimpleNamespace(execute_model=lambda so, it: calls.append(it)),
        _pp_send_work=[],
        use_v2_model_runner=False,
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace()),
    )
    worker.annotate_profile = lambda so: contextlib.nullcontext()
    assert worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=0)) is None
    assert calls == [None]


def test_execute_model_external_launcher_asserts_on_intermediate(monkeypatch):
    class _IT:
        def __init__(self):
            self.tensors = {}

    monkeypatch.setattr(w, "IntermediateTensors", _IT)
    monkeypatch.setattr(
        w,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
    )
    worker = make_worker(
        model_runner=SimpleNamespace(execute_model=lambda so, it: _IT()),
        _pp_send_work=[],
        use_v2_model_runner=False,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(distributed_executor_backend="external_launcher")
        ),
    )
    worker.annotate_profile = lambda so: contextlib.nullcontext()
    with pytest.raises(AssertionError):
        worker.execute_model(SimpleNamespace(total_num_scheduled_tokens=4))


# --------------------------------------------------------------------------- #
# distributed environment timeout branch
# --------------------------------------------------------------------------- #


def test_init_worker_distributed_environment_applies_timeout(monkeypatch):
    rec = []
    # The worker body imports init_batch_invariance locally, so the module that
    # backs that import must be the double.
    _install(
        monkeypatch,
        "vllm.model_executor.determinism.batch_invariant",
        init_batch_invariance=lambda: rec.append("batch"),
    )
    monkeypatch.setattr(
        w, "override_envs_for_eplb", lambda pc, moe_backend=None: rec.append(("eplb", moe_backend))
    )
    monkeypatch.setattr(w, "set_custom_all_reduce", lambda enabled: rec.append(("car", enabled)))
    monkeypatch.setattr(
        w, "init_distributed_environment", lambda *args: rec.append(("init", args))
    )
    monkeypatch.setattr(
        w, "ensure_model_parallel_initialized", lambda *args: rec.append(("mp", args))
    )
    monkeypatch.setattr(w, "ensure_ec_transfer_initialized", lambda cfg: rec.append("ec"))
    monkeypatch.setattr(w, "_normalize_process_group_backend", lambda backend: backend)

    parallel = SimpleNamespace(
        world_size=2,
        disable_custom_all_reduce=False,
        distributed_timeout_seconds=30,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    cfg = SimpleNamespace(
        parallel_config=parallel,
        kernel_config=SimpleNamespace(moe_backend="moe"),
    )
    w.init_worker_distributed_environment(cfg, 1, "tcp://127.0.0.1:2", 0, "nccl")

    init_calls = [entry[1] for entry in rec if isinstance(entry, tuple) and entry[0] == "init"]
    assert init_calls == [(2, 1, "tcp://127.0.0.1:2", 0, "nccl", timedelta(seconds=30))]
    assert ("eplb", "moe") in rec
    assert ("car", True) in rec
    assert "ec" in rec
