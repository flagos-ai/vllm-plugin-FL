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

"""Coverage tests for WorkerFL device binding, load, warmup and MoE reconfig.

``init_device`` translates the rank-local device through the platform and
then constructs the runner; ``load_model`` patches and loads; the compile
hook expands warmup sizes/capture memory; ``execute_model`` plumbs the SP
gather metadata; and ``_reconfigure_moe`` resizes model and drafter modules.
Per CONTRACT R2 the vllm engine objects, distributed groups, memory probes
and the model-runner class are inputs replaced by recorders.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

# The shim import is load-bearing: it must precede vllm_fl/vllm imports.
# isort: split
from . import _imports  # noqa: F401  (installs local import shims first)

# isort: split
import vllm.distributed.parallel_state as parallel_state
from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationMode
from vllm.sequence import IntermediateTensors

import vllm_fl.worker.model_runner as model_runner_module
import vllm_fl.worker.worker as worker_module
from vllm_fl.worker.worker import WorkerFL


class _DeviceFn:
    def __init__(self, count=8):
        self.count = count

    def device_count(self):
        return self.count

    def is_available(self):
        return True


class _FakeDevice:
    """Device token usable where torch lacks the vendor backend."""

    def __init__(self, device_type, index=0):
        self.type = device_type
        self.index = index

    def __eq__(self, other):
        return isinstance(other, _FakeDevice) and (self.type, self.index) == (
            other.type,
            other.index,
        )

    def __hash__(self):
        return hash((self.type, self.index))


class _Platform:
    def __init__(self, *, device_type="cuda", vendor_name="nvidia", count=8):
        self.device_type = device_type
        self.vendor_name = vendor_name
        self.dist_backend = "nccl"
        self.torch_device_fn = _DeviceFn(count)
        self.events = []

    def logical_device_id_to_visible_device_id(self, local_rank):
        return local_rank

    def set_device(self, device):
        self.events.append(("set_device", device))

    def empty_cache(self):
        self.events.append("empty_cache")

    def check_if_supports_dtype(self, dtype):
        self.events.append(("dtype", dtype))


def _parallel_config(**overrides):
    values = {
        "distributed_executor_backend": "mp",
        "data_parallel_backend": "mp",
        "nnodes_within_dp": 1,
        "data_parallel_rank_local": 0,
        "data_parallel_index": 0,
        "data_parallel_rank": 0,
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 1,
        "local_world_size": 1,
        "assigned_physical_gpu_ids": None,
        "enable_dbo": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _worker_for_init(**overrides):
    parallel = overrides.pop("parallel_config", None) or _parallel_config()
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=parallel),
        parallel_config=parallel,
        model_config=SimpleNamespace(dtype=torch.float16, seed=421),
        cache_config=SimpleNamespace(gpu_memory_utilization=0.5),
        device_config=SimpleNamespace(device=None),
        local_rank=overrides.pop("local_rank", 0),
        rank=overrides.pop("rank", 0),
        distributed_init_method="env://",
        **overrides,
    )
    return worker


class _DeviceToken:
    """Vendor device stand-in for platforms absent from this torch build."""

    def __init__(self, spec):
        self.spec = spec
        self.type, _, index = spec.partition(":")
        self.index = int(index) if index else 0


class _TorchProxy:
    """Only ``torch.device`` is needed by ``init_device``."""

    device = staticmethod(_DeviceToken)


def _patch_init_env(monkeypatch, platform, snapshot_free=50, snapshot_total=100):
    monkeypatch.setattr(worker_module, "current_platform", platform)
    if platform.device_type != "cuda":
        # Standard NVIDIA torch rejects vendor names such as musa/npu/ptpu at
        # torch.device(); the branch under test only needs an opaque handle.
        monkeypatch.setattr(worker_module, "torch", _TorchProxy())
    monkeypatch.setattr(
        worker_module,
        "MemorySnapshot",
        lambda: SimpleNamespace(free_memory=snapshot_free, total_memory=snapshot_total),
    )
    calls = {}

    def init_dist(config, rank, method, local_rank, backend):
        calls["dist"] = (config, rank, method, local_rank, backend)

    def init_workspace(device, ubatches):
        calls["workspace"] = (device, ubatches)

    class _Runner:
        def __init__(self, config, device):
            calls["runner"] = (config, device)

    monkeypatch.setattr(worker_module, "init_worker_distributed_environment", init_dist)
    monkeypatch.setattr(worker_module, "init_workspace_manager", init_workspace)
    monkeypatch.setattr(
        worker_module, "set_random_seed", lambda s: calls.update(seed=s)
    )
    monkeypatch.setattr(
        worker_module, "report_usage_stats", lambda cfg: calls.update(usage=cfg)
    )
    monkeypatch.setattr(model_runner_module, "ModelRunnerFL", _Runner)
    return calls


def test_init_device_binds_visible_device_and_builds_runner(monkeypatch):
    """The rank-local device is translated, distributed init runs, runner built."""
    platform = _Platform()
    calls = _patch_init_env(monkeypatch, platform)
    worker = _worker_for_init(local_rank=2)
    WorkerFL.init_device(worker)

    device = torch.device("cuda:2")
    assert worker.device == device
    assert worker.device_config.device is None  # only MUSA mirrors it
    assert platform.events[0] == ("set_device", device)
    assert platform.events[1] == ("dtype", torch.float16)
    assert "empty_cache" in platform.events
    assert calls["dist"] == (
        worker.vllm_config,
        0,
        "env://",
        2,
        "nccl",
    )
    assert calls["seed"] == 421
    assert calls["workspace"] == (device, 1)
    assert calls["runner"] == (worker.vllm_config, device)
    assert calls["usage"] is worker.vllm_config
    assert worker.init_snapshot.total_memory == 100
    assert worker.requested_memory == 50


def test_init_device_applies_data_parallel_rank_offset(monkeypatch):
    """DP_LOCAL_RANK * TP_PP_WORLD_SIZE offsets the local rank."""
    platform = _Platform()
    _patch_init_env(monkeypatch, platform)
    parallel = _parallel_config(
        data_parallel_rank_local=None,
        data_parallel_index=2,
        pipeline_parallel_size=2,
        tensor_parallel_size=1,
    )
    worker = _worker_for_init(local_rank=0, parallel_config=parallel)
    WorkerFL.init_device(worker)
    assert worker.local_rank == 4
    assert worker.device == torch.device("cuda:4")


@pytest.mark.parametrize(
    "overrides",
    [
        {"distributed_executor_backend": "ray"},
        {"distributed_executor_backend": "external_launcher"},
        {"data_parallel_backend": "ray"},
        {"nnodes_within_dp": 2},
    ],
)
def test_init_device_skips_rank_offset_for_ray_or_multinode(monkeypatch, overrides):
    """Ray/external-launcher and multi-node DP leave the local rank alone."""
    platform = _Platform()
    _patch_init_env(monkeypatch, platform)
    parallel = _parallel_config()
    for name, value in overrides.items():
        setattr(parallel, name, value)
    parallel.data_parallel_rank_local = 3
    parallel.data_parallel_index = 3
    parallel.pipeline_parallel_size = 2
    worker = _worker_for_init(local_rank=1, parallel_config=parallel)
    WorkerFL.init_device(worker)
    assert worker.local_rank == 1
    assert worker.device == torch.device("cuda:1")


def test_init_device_rejects_out_of_bounds_local_rank(monkeypatch):
    """The DP-adjusted local rank must fit the visible devices."""
    platform = _Platform(count=8)
    _patch_init_env(monkeypatch, platform)
    parallel = _parallel_config(
        data_parallel_rank_local=None,
        data_parallel_index=1,
        pipeline_parallel_size=2,
        tensor_parallel_size=1,
    )
    worker = _worker_for_init(local_rank=7, parallel_config=parallel)
    with pytest.raises(AssertionError, match="out of bounds"):
        WorkerFL.init_device(worker)


def test_init_device_rejects_oversized_local_world(monkeypatch):
    """local_world_size must not exceed the visible device count."""
    platform = _Platform(count=4)
    _patch_init_env(monkeypatch, platform)
    parallel = _parallel_config(local_world_size=5)
    worker = _worker_for_init(local_rank=0, parallel_config=parallel)
    with pytest.raises(AssertionError, match="local_world_size"):
        WorkerFL.init_device(worker)


def test_init_device_registers_assigned_physical_gpu_ids(monkeypatch):
    """Assigned physical ids are registered and bound the local rank."""
    import vllm.platforms.interface as interface

    platform = _Platform()
    _patch_init_env(monkeypatch, platform)
    registered = []
    monkeypatch.setattr(
        interface,
        "set_assigned_physical_gpu_ids",
        lambda ids: registered.append(ids),
    )
    parallel = _parallel_config(assigned_physical_gpu_ids=[3, 4])
    worker = _worker_for_init(local_rank=1, parallel_config=parallel)
    WorkerFL.init_device(worker)
    assert registered == [[3, 4]]
    assert worker.device == torch.device("cuda:1")

    parallel.assigned_physical_gpu_ids = [3, 4]
    worker = _worker_for_init(local_rank=2, parallel_config=parallel)
    with pytest.raises(AssertionError, match="out of bounds for"):
        WorkerFL.init_device(worker)


def test_init_device_rejects_insufficient_free_memory(monkeypatch):
    """Free memory below the requested utilization raises the startup error."""
    platform = _Platform()
    _patch_init_env(monkeypatch, platform, snapshot_free=10, snapshot_total=100)
    worker = _worker_for_init(local_rank=0)
    with pytest.raises(ValueError, match="is less than desired GPU memory"):
        WorkerFL.init_device(worker)


def test_init_device_mirrors_device_on_musa(monkeypatch):
    """MUSA keeps DeviceConfig aligned with the worker's rank-local device."""
    platform = _Platform(device_type="musa")
    _patch_init_env(monkeypatch, platform)
    worker = _worker_for_init(local_rank=1)
    WorkerFL.init_device(worker)
    assert worker.device.spec == "musa:1"
    assert worker.device_config.device is worker.device


def test_init_device_initializes_npu_triton_properties(monkeypatch):
    """NPU workers initialize the triton device properties after dist init."""
    import sys
    import types

    platform = _Platform(device_type="npu", vendor_name="ascend")
    _patch_init_env(monkeypatch, platform)
    called = []
    fake = types.ModuleType("vllm_fl.dispatch.backends.vendor.ascend.impl.triton_utils")
    fake.init_device_properties_triton = lambda: called.append(1)
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    worker = _worker_for_init(local_rank=0)
    WorkerFL.init_device(worker)
    assert called == [1]


def test_init_device_applies_iluvatar_patches(monkeypatch):
    """Iluvatar remaps the inductor target and sampler before any forward."""
    import sys
    import types

    platform = _Platform(vendor_name="iluvatar")
    _patch_init_env(monkeypatch, platform)
    calls = []
    fake = types.ModuleType("vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar")
    fake.patch_torch_inductor_for_iluvatar = lambda: calls.append("inductor")
    fake.patch_sampler_compile_for_iluvatar = lambda: calls.append("sampler")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    worker = _worker_for_init(local_rank=0)
    WorkerFL.init_device(worker)
    assert calls == ["inductor", "sampler"]


def test_init_device_registers_ptpu_inductor_backend(monkeypatch):
    """PTPU registers its inductor wiring in every worker process."""
    import sys
    import types

    platform = _Platform(device_type="ptpu", vendor_name="sunrise")
    _patch_init_env(monkeypatch, platform)
    calls = []
    fake = types.ModuleType("vllm_fl.compilation.inductor_backend")
    fake.register_ptpu_inductor_backend = lambda: calls.append(1)
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    worker = _worker_for_init(local_rank=0)
    WorkerFL.init_device(worker)
    assert calls == [1]


def test_init_device_nonzero_rank_skips_usage_stats(monkeypatch):
    """Only rank 0 reports usage statistics."""
    platform = _Platform()
    calls = _patch_init_env(monkeypatch, platform)
    worker = _worker_for_init(local_rank=0, rank=1)
    WorkerFL.init_device(worker)
    assert "usage" not in calls


def test_init_device_enables_dbo_double_workspace(monkeypatch):
    """enable_dbo doubles the ubatch count handed to the workspace manager."""
    platform = _Platform()
    calls = _patch_init_env(monkeypatch, platform)
    parallel = _parallel_config(enable_dbo=True)
    worker = _worker_for_init(local_rank=0, parallel_config=parallel)
    WorkerFL.init_device(worker)
    assert calls["workspace"][1] == 2


class _RunnerLoad:
    def __init__(self, model=None):
        self.model = model or nn.Module()
        self.loads = []

    def load_model(self, *, load_dummy_weights=False):
        self.loads.append(load_dummy_weights)


def _patch_load_env(monkeypatch, device_type="cuda"):
    import vllm_fl.patches.moe_sum as moe_sum

    events = []
    monkeypatch.setattr(moe_sum, "patch_vllm_moe_sum", lambda: events.append("patch"))
    monkeypatch.setattr(
        worker_module,
        "set_current_vllm_config",
        lambda cfg: events.append(("config", cfg)) or nullcontext(),
    )
    monkeypatch.setattr(
        worker_module,
        "current_platform",
        SimpleNamespace(device_type=device_type),
    )
    monkeypatch.setattr(
        worker_module, "logger", SimpleNamespace(info=lambda *a, **k: None)
    )
    return events


def test_load_model_patches_before_loading_under_the_config(monkeypatch):
    """The moe-sum patch runs first, then the runner loads inside the config."""
    events = _patch_load_env(monkeypatch)
    runner = _RunnerLoad()
    worker = SimpleNamespace(vllm_config="CFG", model_runner=runner, rank=0)
    WorkerFL.load_model(worker, load_dummy_weights=True)
    assert events[0] == "patch"
    assert events[1] == ("config", "CFG")
    assert runner.loads == [True]


def test_load_model_musa_accepts_parameters_on_the_bound_device(monkeypatch):
    """MUSA load accepts parameters whose device matches the worker."""
    _patch_load_env(monkeypatch, device_type="musa")

    class _Model(nn.Module):
        def named_parameters(self, *args, **kwargs):
            return [("w", SimpleNamespace(device=_FakeDevice("musa", 1)))]

    runner = _RunnerLoad(model=_Model())
    worker = SimpleNamespace(
        vllm_config="CFG",
        model_runner=runner,
        rank=0,
        device=_FakeDevice("musa", 1),
        device_config=SimpleNamespace(device=_FakeDevice("musa", 1)),
    )
    WorkerFL.load_model(worker)
    assert runner.loads == [False]


def test_load_model_musa_rejects_parameters_on_another_device(monkeypatch):
    """A parameter on a different MUSA device fails the rank check."""
    _patch_load_env(monkeypatch, device_type="musa")

    class _Model(nn.Module):
        def named_parameters(self, *args, **kwargs):
            return [("w", SimpleNamespace(device=_FakeDevice("musa", 2)))]

    runner = _RunnerLoad(model=_Model())
    worker = SimpleNamespace(
        vllm_config="CFG",
        model_runner=runner,
        rank=3,
        device=_FakeDevice("musa", 1),
        device_config=SimpleNamespace(device=_FakeDevice("musa", 1)),
    )
    with pytest.raises(RuntimeError, match="Rank 3 expected model parameters"):
        WorkerFL.load_model(worker)


class _Range:
    def __init__(self, start, end):
        self.start, self.end = start, end

    def __contains__(self, value):
        return self.start <= value <= self.end


class _WarmupRunner:
    def __init__(self, dummy=("hidden", "last"), cudagraph_bytes=0):
        self.dummy = dummy
        self.cudagraph_bytes = cudagraph_bytes
        self.dummy_calls = []
        self.pooler_calls = []
        self.sampler_calls = []
        self.capture_calls = 0
        self.lora_removals = []
        self.model_memory_usage = 1000
        self.is_pooling_model = False
        self.lora_config = "LORA"

    def _dummy_run(self, num_tokens, **kwargs):
        self.dummy_calls.append((num_tokens, kwargs))
        return self.dummy

    def _dummy_pooler_run(self, hidden):
        self.pooler_calls.append(hidden)

    def _dummy_sampler_run(self, *, hidden_states):
        self.sampler_calls.append(hidden_states)

    def maybe_remove_all_loras(self, config):
        self.lora_removals.append(config)

    def capture_model(self):
        self.capture_calls += 1
        return self.cudagraph_bytes


def _warmup_worker(runner, compilation, *, last_rank=True, vendor="nvidia"):
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(compilation_config=compilation),
        compilation_config=compilation,
        model_runner=runner,
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.9
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=32),
        model_config=SimpleNamespace(seed=99),
        init_snapshot=SimpleNamespace(free_memory=100, total_memory=200),
        requested_memory=80,
        peak_activation_memory=7,
        non_torch_memory=3,
        available_kv_cache_memory_bytes=11,
    )
    return worker


def test_compile_or_warm_up_without_compile_skips_size_expansion(monkeypatch):
    """CompilationMode.NONE warms only the sampler/pooler path."""
    runner = _WarmupRunner()
    runner.is_pooling_model = True
    compilation = SimpleNamespace(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        compilation_time=0.5,
        encoder_compilation_time=0.25,
    )
    worker = _warmup_worker(runner, compilation)
    warmed = []
    monkeypatch.setattr(worker_module, "kernel_warmup", lambda w: warmed.append(w))
    monkeypatch.setattr(
        worker_module, "current_platform", SimpleNamespace(device_type="cuda")
    )
    monkeypatch.setattr(
        worker_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    seeds = []
    monkeypatch.setattr(worker_module, "set_random_seed", seeds.append)

    times = WorkerFL.compile_or_warm_up_model(worker)

    assert warmed == [worker]
    assert runner.lora_removals == ["LORA"]
    assert runner.dummy_calls == [
        (
            4,
            {
                "skip_eplb": True,
                "cudagraph_runtime_mode": CUDAGraphMode.NONE,
            },
        )
    ]
    assert runner.pooler_calls == ["hidden"]
    assert runner.sampler_calls == []
    assert runner.capture_calls == 0
    assert seeds == [99]
    assert (times.language_model, times.encoder) == (0.5, 0.25)


def test_compile_or_warm_up_expands_sizes_and_accounts_graph_memory(monkeypatch):
    """cg sizes are excluded, empty ranges completed, memory logs emitted."""
    runner = _WarmupRunner(cudagraph_bytes=123)
    runner.is_pooling_model = False
    compilation = SimpleNamespace(
        mode=CompilationMode.VLLM_COMPILE,
        compile_sizes=[8, 32],
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        cudagraph_capture_sizes=[8, 16],
        get_compile_ranges=lambda: [_Range(1, 16), _Range(20, 24)],
        compilation_time=1.0,
        encoder_compilation_time=0.5,
    )
    worker = _warmup_worker(runner, compilation)
    worker.cudagraph_memory_estimate = 500
    monkeypatch.setattr(worker_module, "kernel_warmup", lambda w: None)
    monkeypatch.setattr(
        worker_module, "current_platform", SimpleNamespace(device_type="cuda")
    )
    monkeypatch.setattr(
        worker_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    monkeypatch.setattr(worker_module, "set_random_seed", lambda s: None)
    monkeypatch.setattr(
        worker_module.envs, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", True
    )

    WorkerFL.compile_or_warm_up_model(worker)

    sizes = [call[0] for call in runner.dummy_calls]
    assert sizes == [32, 24]  # 8 removed (cg), 32 kept, 24 appended, descending
    assert runner.capture_calls == 1
    assert runner.sampler_calls == []


def test_compile_or_warm_up_skips_kernel_warmup_for_vendor_runtime(monkeypatch):
    """kunlunxin/thead/txda skip the generic kernel warmup."""
    runner = _WarmupRunner()
    compilation = SimpleNamespace(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        compilation_time=0.0,
        encoder_compilation_time=0.0,
    )
    worker = _warmup_worker(runner, compilation)
    warmed = []
    monkeypatch.setattr(worker_module, "kernel_warmup", lambda w: warmed.append(w))
    monkeypatch.setattr(
        worker_module,
        "current_platform",
        SimpleNamespace(device_type="cuda", vendor_name="kunlunxin"),
    )
    monkeypatch.setattr(
        worker_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    monkeypatch.setattr(worker_module, "set_random_seed", lambda s: None)
    WorkerFL.compile_or_warm_up_model(worker)
    assert warmed == []


def test_compile_or_warm_up_swallows_warmup_import_failure(monkeypatch):
    """A missing MiniMax/torchvision chain during warmup is logged, not fatal."""
    runner = _WarmupRunner()
    compilation = SimpleNamespace(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        compilation_time=0.0,
        encoder_compilation_time=0.0,
    )
    worker = _warmup_worker(runner, compilation)

    def failing(worker_arg):
        raise ImportError("torchvision missing")

    monkeypatch.setattr(worker_module, "kernel_warmup", failing)
    monkeypatch.setattr(
        worker_module, "current_platform", SimpleNamespace(device_type="cuda")
    )
    monkeypatch.setattr(
        worker_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    monkeypatch.setattr(worker_module, "set_random_seed", lambda s: None)
    WorkerFL.compile_or_warm_up_model(worker)


def test_compile_or_warm_up_sampler_run_uses_last_hidden_states(monkeypatch):
    """The last PP rank warms the sampler with the model's last hidden states."""
    runner = _WarmupRunner(dummy=("hidden-x", "last-x"))
    compilation = SimpleNamespace(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        compilation_time=0.0,
        encoder_compilation_time=0.0,
    )
    worker = _warmup_worker(runner, compilation)
    monkeypatch.setattr(worker_module, "kernel_warmup", lambda w: None)
    monkeypatch.setattr(
        worker_module, "current_platform", SimpleNamespace(device_type="cuda")
    )
    monkeypatch.setattr(
        worker_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    monkeypatch.setattr(worker_module, "set_random_seed", lambda s: None)
    WorkerFL.compile_or_warm_up_model(worker)
    assert runner.sampler_calls == ["last-x"]


class _SPRunner:
    def __init__(self):
        self.batch_calls = []
        self.exec_calls = []

    def _determine_batch_execution_and_padding(self, **kwargs):
        self.batch_calls.append(kwargs)
        return None, SimpleNamespace(num_tokens=7), None, None, None

    def execute_model(self, scheduler_output, intermediate_tensors):
        self.exec_calls.append(intermediate_tensors)
        return IntermediateTensors({"out": torch.ones(1)})


def test_execute_model_sp_gather_plumbing(monkeypatch):
    """PP+SP forwards the computed residual gather map to recv and send."""
    runner = _SPRunner()
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(enable_sp=True)
            ),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=2,
                distributed_executor_backend="mp",
            ),
        ),
        model_runner=runner,
        profiler=None,
    )
    worker.annotate_profile = lambda sched: WorkerFL.annotate_profile(worker, sched)
    sent = []
    received = {}

    class _PP:
        is_first_rank = False
        is_last_rank = False

        def recv_tensor_dict(self, all_gather_group=None, all_gather_tensors=None):
            received.update(group=all_gather_group, tensors=dict(all_gather_tensors))
            return {"hidden": torch.zeros(1)}

        def send_tensor_dict(
            self, tensors, all_gather_group=None, all_gather_tensors=None
        ):
            sent.append((tensors, dict(all_gather_tensors)))

    tp_group = object()
    monkeypatch.setattr(worker_module, "get_pp_group", lambda: _PP())
    monkeypatch.setattr(worker_module, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        worker_module, "is_residual_scattered_for_sp", lambda config, tokens: False
    )
    sched = SimpleNamespace(
        total_num_scheduled_tokens=5,
        num_scheduled_tokens={"a": 2, "b": 3},
    )
    result = WorkerFL.execute_model(worker, sched)

    assert result is None
    call = runner.batch_calls[0]
    assert call["num_tokens"] == 5
    assert call["num_reqs"] == 2
    assert call["num_scheduled_tokens_np"].tolist() == [2, 3]
    assert call["max_num_scheduled_tokens"] == 3
    assert call["use_cascade_attn"] is False
    assert received == {"group": tp_group, "tensors": {"residual": True}}
    assert len(sent) == 1
    assert sent[0][1] == {"residual": True}
    assert isinstance(runner.exec_calls[0], IntermediateTensors)


class FusedMoE(nn.Module):
    def __init__(self, num_local, num_experts):
        super().__init__()
        self.moe_config = SimpleNamespace(
            num_local_experts=num_local,
            num_experts=num_experts,
            moe_parallel_config=None,
        )
        self.global_num_experts = num_experts
        self.moe_parallel_config = None


class _MoEModel(nn.Module):
    def __init__(self, moe=None):
        super().__init__()
        self.moe = moe or FusedMoE(4, 4)
        self.linear = nn.Module()

    def update_physical_experts_metadata(self, **kwargs):
        self.updates = kwargs


class _Groups:
    tp = SimpleNamespace(world_size=2)
    pcp = SimpleNamespace(world_size=1)
    dp = SimpleNamespace(world_size=1, rank_in_group=0)
    ep = SimpleNamespace(world_size=2, cpu_group="cpu-group")


def test_reconfigure_moe_updates_drafter_and_model_together(monkeypatch):
    """A drafter MoE model with matching expert counts is resized and prepared."""
    import vllm.model_executor.layers.fused_moe.config as moe_config

    model = _MoEModel(FusedMoE(4, 4))
    drafter_model = _MoEModel(FusedMoE(4, 4))
    prepared = []
    parallel = SimpleNamespace(
        enable_expert_parallel=True,
        all2all_backend="none",
        enable_eplb=True,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=parallel),
        model_runner=SimpleNamespace(
            model=model,
            drafter=SimpleNamespace(model=drafter_model),
            eplb_state=SimpleNamespace(
                rearrange=lambda **kw: None,
                physical_to_logical_map=torch.zeros(2, 10),
                logical_replica_count=torch.zeros(2, 6),
            ),
        ),
    )
    monkeypatch.setattr(worker_module, "get_tp_group", lambda: _Groups.tp)
    monkeypatch.setattr(worker_module, "get_pcp_group", lambda: _Groups.pcp)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: _Groups.tp)
    monkeypatch.setattr(parallel_state, "get_pcp_group", lambda: _Groups.pcp)
    monkeypatch.setattr(parallel_state, "get_dp_group", lambda: _Groups.dp)
    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: _Groups.ep)
    monkeypatch.setattr(moe_config, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(moe_config, "get_dp_group", lambda: _Groups.dp)
    monkeypatch.setattr(moe_config, "get_pcp_group", lambda: _Groups.pcp)
    monkeypatch.setattr(
        parallel_state,
        "prepare_communication_buffer_for_model",
        lambda m: prepared.append(m),
    )
    monkeypatch.setattr(
        worker_module, "is_mixture_of_experts", lambda m: m is drafter_model
    )

    result = WorkerFL._reconfigure_moe(worker, old_ep_size=4, new_ep_size=2)

    assert result is None
    assert prepared == [model, drafter_model]
    for candidate in (model, drafter_model):
        assert candidate.moe.moe_config.num_experts == 8
        assert candidate.moe.global_num_experts == 8
    assert model.updates == {
        "num_physical_experts": 10,
        "num_local_physical_experts": 4,
    }
    assert parallel.eplb_config.num_redundant_experts == 4


def test_reconfigure_moe_rejects_drafter_with_different_expert_count(monkeypatch):
    """Drafter and model must agree on the local expert count."""
    import vllm.model_executor.layers.fused_moe.config as moe_config

    model = _MoEModel(FusedMoE(4, 4))
    drafter_model = _MoEModel(FusedMoE(3, 3))
    parallel = SimpleNamespace(
        enable_expert_parallel=True,
        all2all_backend="none",
        enable_eplb=True,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(parallel_config=parallel),
        model_runner=SimpleNamespace(
            model=model,
            drafter=SimpleNamespace(model=drafter_model),
            eplb_state=SimpleNamespace(
                rearrange=lambda **kw: None,
                physical_to_logical_map=torch.zeros(2, 10),
                logical_replica_count=torch.zeros(2, 6),
            ),
        ),
    )
    monkeypatch.setattr(worker_module, "get_tp_group", lambda: _Groups.tp)
    monkeypatch.setattr(worker_module, "get_pcp_group", lambda: _Groups.pcp)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: _Groups.tp)
    monkeypatch.setattr(parallel_state, "get_pcp_group", lambda: _Groups.pcp)
    monkeypatch.setattr(parallel_state, "get_dp_group", lambda: _Groups.dp)
    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: _Groups.ep)
    monkeypatch.setattr(moe_config, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(moe_config, "get_dp_group", lambda: _Groups.dp)
    monkeypatch.setattr(moe_config, "get_pcp_group", lambda: _Groups.pcp)
    monkeypatch.setattr(
        parallel_state, "prepare_communication_buffer_for_model", lambda m: None
    )
    monkeypatch.setattr(worker_module, "is_mixture_of_experts", lambda m: True)
    with pytest.raises(AssertionError, match="Drafter and model configs"):
        WorkerFL._reconfigure_moe(worker, old_ep_size=4, new_ep_size=2)


class _MemoryPlatform:
    """Device-memory surface used by determine_available_memory."""

    def __init__(self):
        self.events = []

    @staticmethod
    def is_cuda():
        return True

    @staticmethod
    def is_rocm():
        return False

    vendor_name = "nvidia"
    device_type = "cuda"
    torch_device_fn = SimpleNamespace(
        reset_peak_memory_stats=lambda: None,
        memory_stats=lambda: {"allocated_bytes.all.peak": 60},
    )

    def empty_cache(self):
        self.events.append("empty_cache")


@pytest.mark.parametrize(("estimate_env", "expected"), [(True, 125), (False, 145)])
def test_determine_available_memory_profiles_and_accounts_graphs(
    monkeypatch, estimate_env, expected
):
    """Peak activation, non-torch memory and the graph estimate size the cache."""
    platform = _MemoryPlatform()
    monkeypatch.setattr(worker_module, "current_platform", platform)
    monkeypatch.setattr(
        worker_module.envs, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", estimate_env
    )
    profile_calls = []
    runner = SimpleNamespace(
        profile_run=lambda: profile_calls.append("run"),
        profile_cudagraph_memory=lambda: profile_calls.append("graph") or 20,
        model_memory_usage=5,
    )

    @contextmanager
    def fake_profiling(baseline, weights_memory):
        result = SimpleNamespace(
            before_profile=SimpleNamespace(torch_peak=10),
            after_profile=SimpleNamespace(free_memory=40),
            non_torch_increase=100,
            weights_memory=weights_memory,
        )
        yield result

    monkeypatch.setattr(worker_module, "memory_profiling_fl", fake_profiling)
    worker = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.5
        ),
        model_runner=runner,
        init_snapshot=SimpleNamespace(free_memory=50, total_memory=100),
        requested_memory=300,
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE)
        ),
    )

    result = WorkerFL.determine_available_memory(worker)

    assert result == expected
    assert profile_calls == ["run", "graph"]
    assert worker.non_torch_memory == 100
    assert worker.peak_activation_memory == 50
    assert worker.cudagraph_memory_estimate == 20
    assert worker.available_kv_cache_memory_bytes == expected
    assert platform.events == ["empty_cache"]
