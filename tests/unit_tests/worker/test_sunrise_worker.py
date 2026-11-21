# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for the official-0.32 Sunrise worker bridge."""

import functools
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def worker_module(monkeypatch):
    events = []

    def install(name, **attrs):
        parent_name, _, child = name.rpartition(".")
        parent = None
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is None:
                parent = install(parent_name)
        module = ModuleType(name)
        module.__path__ = []
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        if parent is not None:
            monkeypatch.setattr(parent, child, module, raising=False)
        return module

    def checked(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            events.append("checked")
            return fn(*args, **kwargs)

        return wrapper

    class NativeWorker:
        def __init__(self, cfg, local_rank=0, rank=0, **kwargs):
            events.append("native_init")
            vars(self).update(vars(cfg))
            self.vllm_config = cfg
            self.local_rank = local_rank
            self.rank = rank
            self.distributed_init_method = "tcp://127.0.0.1:12345"
            self.use_v2_model_runner = getattr(cfg, "use_v2_model_runner", False)

        def _init_workspace_and_model_runner(self):
            events.append("runner")

        def get_kv_cache_spec(self):
            return "native_032_spec"

        @torch.inference_mode()
        def determine_available_memory(self):
            return torch.is_inference_mode_enabled(), torch.is_grad_enabled()

        @torch.inference_mode()
        @checked
        def execute_model(self, output):
            return torch.is_inference_mode_enabled(), torch.is_grad_enabled()

        @torch.inference_mode()
        @checked
        def sample_tokens(self, output):
            return torch.is_inference_mode_enabled(), torch.is_grad_enabled()

    device_fn = NS(
        device_count=lambda: 8,
        empty_cache=lambda: events.append("empty"),
        synchronize=Mock(),
    )
    platform = NS(
        device_type="ptpu",
        torch_device_fn=device_fn,
        set_device=lambda device: events.append(("device", device)),
        check_if_supports_dtype=lambda dtype: events.append("dtype"),
        logical_device_id_to_visible_device_id=lambda index: index,
    )
    install("vllm")
    install("vllm.platforms", current_platform=platform)
    install("vllm.platforms.interface", set_assigned_physical_gpu_ids=Mock())
    install("vllm.utils.gpu_sync_debug", with_gpu_sync_check=checked)
    upstream = install(
        "vllm.v1.worker.gpu_worker",
        Worker=NativeWorker,
        init_worker_distributed_environment=Mock(
            side_effect=lambda *args: events.append(("pg", args[-1]))
        ),
        set_random_seed=lambda seed: events.append("seed"),
        get_model_loader=lambda config: NS(get_external_weight_memory=lambda cfg: 128),
        MemorySnapshot=lambda **kwargs: events.append("snapshot") or "snapshot",
        request_memory=lambda snapshot, cache, external: (
            events.append(("request", external)) or 4096
        ),
        logger=Mock(),
    )
    install("vllm_fl")
    install(
        "vllm_fl.dispatch.backends.vendor.sunrise.patch",
        apply_sunrise_patches=lambda: events.append("patch"),
    )
    install(
        "vllm_fl.worker.runtime",
        _initialize_fl_runtime=lambda rank: events.append("flruntime"),
    )
    root = Path(__file__).resolve().parents[3]

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    io = load("vllm_fl.dispatch.io_common", root / "vllm_fl/dispatch/io_common.py")
    module = load("_sunrise_worker_test", root / "vllm_fl/worker/sunrise_worker.py")
    # CPU PyTorch does not register a PTPU torch.device parser.
    monkeypatch.setattr(module, "torch", NS(device=lambda value: value))
    config = NS(
        device_config=NS(device_type="ptpu"),
        parallel_config=NS(
            distributed_executor_backend="mp",
            data_parallel_backend="mp",
            nnodes_within_dp=1,
            data_parallel_rank_local=0,
            data_parallel_index=0,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=2,
            assigned_physical_gpu_ids=None,
            local_world_size=2,
        ),
        model_config=NS(
            dtype=torch.bfloat16,
            seed=42,
            enable_sleep_mode=False,
            enable_cumem_allocator=False,
        ),
        load_config=NS(),
        cache_config=NS(),
    )
    return NS(
        module=module,
        upstream=upstream,
        platform=platform,
        config=config,
        events=events,
        io=io,
    )


def test_runtime_is_initialized_before_device_snapshot(worker_module):
    state = worker_module
    worker = state.module.SunriseWorker(state.config)
    assert state.events == ["native_init", "patch", "flruntime"]
    worker.init_device()
    assert state.events.index("patch") < state.events.index("snapshot")
    assert state.events.index(("pg", "pccl")) < state.events.index("snapshot")
    assert state.events[-1] == "runner"
    assert worker.device == "ptpu:0"
    assert worker.requested_memory == 4096
    assert worker.get_kv_cache_spec() == "native_032_spec"


def test_dp_rank_mapping_keeps_pcp_dimension(worker_module):
    state = worker_module
    state.config.parallel_config.data_parallel_rank_local = 1
    state.config.parallel_config.prefill_context_parallel_size = 2
    worker = state.module.SunriseWorker(state.config)
    worker.init_device()
    assert worker.local_rank == 4
    assert worker.device == "ptpu:4"


def test_ray_worker_keeps_assigned_local_rank(worker_module):
    state = worker_module
    state.config.parallel_config.distributed_executor_backend = "ray"
    state.config.parallel_config.data_parallel_rank_local = 1
    worker = state.module.SunriseWorker(state.config, local_rank=1)
    worker.init_device()
    assert worker.local_rank == 1
    assert worker.device == "ptpu:1"


@pytest.mark.parametrize(
    "method", ["execute_model", "sample_tokens", "determine_available_memory"]
)
@pytest.mark.parametrize("io_active", [False, True])
def test_managed_context_replaces_inner_native_inference_mode(
    worker_module, method, io_active
):
    state = worker_module
    worker = state.module.SunriseWorker(state.config)
    state.io.set_io_active(io_active)
    try:
        args = () if method == "determine_available_memory" else (NS(),)
        inference_enabled, grad_enabled = getattr(worker, method)(*args)
        assert inference_enabled is (not io_active)
        assert not grad_enabled
        assert state.events.count("checked") == (
            0 if method == "determine_available_memory" else 1
        )
    finally:
        state.io.set_io_active(False)


def test_memory_pool_uses_regular_ptpu_allocator(worker_module):
    worker = worker_module.module.SunriseWorker(worker_module.config)
    with worker._maybe_get_memory_pool_context("weights"):
        pass
    worker.model_config.enable_sleep_mode = True
    with pytest.raises(NotImplementedError, match="CuMem"):
        worker._maybe_get_memory_pool_context("weights")


def test_other_platforms_cannot_use_sunrise_worker(worker_module):
    worker_module.platform.device_type = "cuda"
    with pytest.raises(RuntimeError, match="Sunrise/PTPU"):
        worker_module.module.SunriseWorker(worker_module.config)
    assert worker_module.events == []


@pytest.mark.parametrize("use_v2", [False, True])
def test_runner_selection_preserves_the_native_config(
    worker_module, monkeypatch, use_v2
):
    state = worker_module
    state.config.use_v2_model_runner = use_v2
    native_runner = Mock(return_value="v1_runner")
    v2_runner = Mock(return_value="v2_runner")
    for name, attr, runner in (
        ("sunrise_model_runner", "SunriseModelRunner", native_runner),
        ("sunrise_v2_model_runner", "SunriseV2ModelRunner", v2_runner),
    ):
        module = ModuleType(f"vllm_fl.worker.{name}")
        setattr(module, attr, runner)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    worker = state.module.SunriseWorker(state.config)
    worker.device = "ptpu:0"
    assert worker.use_v2_model_runner is use_v2
    assert worker._make_model_runner() == ("v2_runner" if use_v2 else "v1_runner")
    selected, unselected = (
        (v2_runner, native_runner) if use_v2 else (native_runner, v2_runner)
    )
    selected.assert_called_once_with(state.config, worker.device)
    unselected.assert_not_called()
