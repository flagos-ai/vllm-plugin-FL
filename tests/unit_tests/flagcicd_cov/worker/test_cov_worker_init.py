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

"""Coverage tests for vllm_fl/worker/worker.py construction and memory paths.

``WorkerFL.__init__`` runs its platform/env validation before delegating
to vllm's WorkerBase: the GCU per-rank triton cache dir, the MTP=1
async-scheduling scheduler rewrite, and the profiler type whitelist.
The vllm WorkerBase/engine objects are inputs (CONTRACT R2) and are
driven through the same SimpleNamespace config the dev tests use;
``register_oot_ops`` mutates a process-global registry, so each
construction clears it and restores state afterwards.

``MemorySnapshot``/``memory_profiling_fl`` read memory counters through
``current_platform.torch_device_fn`` — a platform input the tests replace
with recording fakes (the boundary the dev tests also patch). The txda
and ``kv_cache_memory_bytes`` branches of
``determine_available_memory`` and the DP-rank math of ``init_device``
are likewise driven by platform/parallel-config inputs.
"""

from . import _imports  # noqa: F401  (installs local import shims first)

import os
from contextlib import nullcontext
from types import SimpleNamespace

import torch

import vllm_fl.worker.worker as worker_module
from vllm_fl.worker.worker import (
    WorkerFL,
    memory_profiling_fl,
    MemorySnapshot,
)


def _make_vllm_config(**overrides):
    """Build the SimpleNamespace vllm config WorkerBase.__init__ reads."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=128, dtype=torch.float16, seed=1234
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks=None,
            num_cpu_blocks=None,
            kv_cache_memory_bytes=None,
            gpu_memory_utilization=0.9,
        ),
        lora_config=None,
        load_config=None,
        parallel_config=SimpleNamespace(rank=None, enable_expert_parallel=False),
        scheduler_config=SimpleNamespace(
            scheduler_cls="vllm.v1.core.sched.scheduler.Scheduler",
            async_scheduling=False,
            max_num_seqs=4,
        ),
        device_config=None,
        speculative_config=None,
        observability_config=None,
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=None, ir_enable_torch_wrap=False
        ),
        profiler_config=SimpleNamespace(profiler=None),
        kernel_config=SimpleNamespace(
            ir_op_priority=SimpleNamespace(set_default=lambda: None)
        ),
        num_speculative_tokens=None,
    )
    for key, value in overrides.items():
        section, sep, field = key.partition("__")
        if not sep:
            setattr(config, key, value)
        else:
            setattr(getattr(config, section), field, value)
    return config


class _RegistryGuard:
    """Clear the process-global OOT op registry around a WorkerFL build.

    ``register_oot_ops`` (invoked at the end of every WorkerFL.__init__)
    asserts on duplicate op names in vllm's global registry; tests build
    more than one worker per process and must reset it between builds.
    """

    def __enter__(self):
        from vllm.model_executor import custom_op

        self._custom_op = custom_op
        self._registry = custom_op.op_registry_oot
        self._saved = dict(custom_op.op_registry_oot)
        custom_op.op_registry_oot = {}
        return self

    def __exit__(self, *exc):
        self._custom_op.op_registry_oot = self._saved
        return False


def _build_worker(**overrides):
    """Construct a WorkerFL with USE_FLAGGEMS off and registry isolated."""
    os.environ["USE_FLAGGEMS"] = "0"
    config = _make_vllm_config(**overrides)
    with _RegistryGuard():
        worker = WorkerFL(
            vllm_config=config,
            local_rank=0,
            rank=0,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
    return worker, config


class _FakeDeviceFn:
    """Fake torch_device_fn recording memory counter values."""

    def __init__(self, stats=777, free=100, total=500, reserved=50):
        self.stats = stats
        self.free = free
        self.total = total
        self.reserved = reserved

    def memory_stats(self):
        return {"allocated_bytes.all.peak": self.stats}

    def mem_get_info(self):
        return (self.free, self.total)

    def memory_reserved(self):
        return self.reserved

    def empty_cache(self):
        pass

    def reset_peak_memory_stats(self):
        pass


def _patched_platform(worker_module, fake):
    """Replace worker_module.current_platform, returning a restore token."""
    original = worker_module.current_platform
    worker_module.current_platform = fake
    return original


def test_snapshot_measures_all_counters_from_platform():
    """measure() derives every field from the torch_device_fn counters."""
    original = _patched_platform(
        worker_module,
        SimpleNamespace(torch_device_fn=_FakeDeviceFn()),
    )
    try:
        snap = MemorySnapshot()
    finally:
        worker_module.current_platform = original
    assert snap.torch_peak == 777
    assert snap.free_memory == 100
    assert snap.total_memory == 500
    assert snap.cuda_memory == 400
    assert snap.torch_memory == 50
    assert snap.non_torch_memory == 350
    assert snap.timestamp > 0


def test_snapshot_falls_back_to_zero_when_counters_unavailable():
    """AttributeError/RuntimeError from counters leaves zeros, not errors."""

    class _Raising:
        def memory_stats(self):
            raise AttributeError("no stats")

        def mem_get_info(self):
            return (10, 20)

        def memory_reserved(self):
            raise RuntimeError("no reserved")

    original = _patched_platform(
        worker_module, SimpleNamespace(torch_device_fn=_Raising())
    )
    try:
        snap = MemorySnapshot()
    finally:
        worker_module.current_platform = original
    assert snap.torch_peak == 0
    assert snap.torch_memory == 0
    assert snap.free_memory == 10
    assert snap.cuda_memory == 10
    assert snap.non_torch_memory == 10


def test_snapshot_subtraction_keeps_total_and_deltas():
    """__sub__ reports per-field deltas and keeps the minuend total."""
    d = MemorySnapshot(auto_measure=False)
    d.torch_peak, d.free_memory, d.total_memory = 10, 3, 100
    d.cuda_memory, d.torch_memory, d.non_torch_memory, d.timestamp = 90, 20, 70, 5.0
    e = MemorySnapshot(auto_measure=False)
    e.torch_peak, e.free_memory, e.total_memory = 4, 1, 100
    e.cuda_memory, e.torch_memory, e.non_torch_memory, e.timestamp = 60, 8, 52, 2.0
    diff = d - e
    assert diff.torch_peak == 6
    assert diff.free_memory == 2
    assert diff.total_memory == 100
    assert diff.cuda_memory == 30
    assert diff.torch_memory == 12
    assert diff.non_torch_memory == 18
    assert diff.timestamp == 3.0


def test_memory_profiling_fl_populates_result_fields():
    """The context manager fills increases from before/after measurements."""
    platform = SimpleNamespace(torch_device_fn=_FakeDeviceFn(stats=10, reserved=2))
    original = _patched_platform(worker_module, platform)
    try:
        base = MemorySnapshot(auto_measure=False)
        base.torch_peak, base.free_memory, base.total_memory = 1, 2, 3
        base.cuda_memory, base.torch_memory, base.non_torch_memory = 4, 5, 6
        base.timestamp = 7.0
        with memory_profiling_fl(base, weights_memory=64) as result:
            assert result.before_create is base
            assert result.weights_memory == 64
            platform.torch_device_fn = _FakeDeviceFn(stats=74, reserved=9)
        # before_profile measured stats=10 at entry; after_profile 74.
        assert result.torch_peak_increase == 64
        # after_profile: free=100/total=500 -> cuda=400, reserved=9 ->
        # non_torch=391; before_create was hand-built with non_torch=6.
        assert result.non_torch_increase == 391 - 6
        assert result.non_kv_cache_memory == (391 - 6) + 64 + 64
        assert result.profile_time >= 0.0
    finally:
        worker_module.current_platform = original


def test_init_sets_gcu_triton_cache_dir_per_rank():
    """GCU workers get a per-rank TRITON_CACHE_DIR when none is set."""
    os.environ.pop("TRITON_CACHE_DIR", None)
    original = _patched_platform(
        worker_module, SimpleNamespace(device_type="gcu")
    )
    try:
        _, _ = _build_worker()
        assert os.environ["TRITON_CACHE_DIR"] == "/tmp/triton-cache-fl-rank-0"
    finally:
        os.environ.pop("TRITON_CACHE_DIR", None)
        worker_module.current_platform = original


def test_init_keeps_preset_triton_cache_dir_on_gcu():
    """An operator-provided TRITON_CACHE_DIR is never overwritten."""
    os.environ["TRITON_CACHE_DIR"] = "/custom-cache"
    original = _patched_platform(
        worker_module, SimpleNamespace(device_type="gcu")
    )
    try:
        _build_worker()
        assert os.environ["TRITON_CACHE_DIR"] == "/custom-cache"
    finally:
        os.environ.pop("TRITON_CACHE_DIR", None)
        worker_module.current_platform = original


def test_init_does_not_set_triton_cache_dir_on_cuda():
    """Non-GCU platforms leave the TRITON_CACHE_DIR env untouched."""
    original = _patched_platform(
        worker_module, SimpleNamespace(device_type="cuda")
    )
    try:
        _build_worker()
        assert "TRITON_CACHE_DIR" not in os.environ
    finally:
        os.environ.pop("TRITON_CACHE_DIR", None)
        worker_module.current_platform = original


def test_init_rejects_unknown_profiler_type():
    """A profiler outside torch/cuda/None trips the whitelist ValueError."""
    try:
        _build_worker(profiler_config__profiler="bogus")
    except ValueError as exc:
        assert "Unknown profiler type: bogus" in str(exc)
        return
    raise AssertionError("unknown profiler type did not raise ValueError")


def test_init_rewrites_scheduler_cls_for_mtp1_async_scheduling():
    """MTP=1 + async scheduling swaps in the plugin AsyncSchedulerFL."""
    worker, config = _build_worker(
        num_speculative_tokens=1, scheduler_config__async_scheduling=True
    )
    assert (
        config.scheduler_config.scheduler_cls
        == "vllm_fl.worker.scheduler_fl.AsyncSchedulerFL"
    )
    assert worker._sleep_saved_buffers == {}
    assert worker.profiler is None


def test_init_keeps_scheduler_cls_without_async_scheduling():
    """MTP=1 without async scheduling keeps the stock scheduler class."""
    _, config = _build_worker(num_speculative_tokens=1)
    assert (
        config.scheduler_config.scheduler_cls
        == "vllm.v1.core.sched.scheduler.Scheduler"
    )


def test_init_keeps_scheduler_cls_for_mtp_two():
    """MTP != 1 keeps the stock scheduler class even with async scheduling."""
    _, config = _build_worker(
        num_speculative_tokens=2, scheduler_config__async_scheduling=True
    )
    assert (
        config.scheduler_config.scheduler_cls
        == "vllm.v1.core.sched.scheduler.Scheduler"
    )


def test_initialize_cache_records_block_counts():
    """initialize_cache stores both block counts on the cache config."""
    worker, config = _build_worker()
    worker.initialize_cache(11, 3)
    assert config.cache_config.num_gpu_blocks == 11
    assert config.cache_config.num_cpu_blocks == 3


def test_update_max_model_len_updates_model_and_runner():
    """update_max_model_len mirrors the new len into model_config and runner."""
    worker, config = _build_worker()
    worker.model_runner = SimpleNamespace(max_model_len=128)
    worker.update_max_model_len(256)
    assert config.model_config.max_model_len == 256
    assert worker.model_runner.max_model_len == 256


def test_determine_available_memory_txda_env_override():
    """txda reads VLLM_TXDA_KV_CACHE_SIZE when the env var is set."""
    os.environ["VLLM_TXDA_KV_CACHE_SIZE"] = str(5 * 1024**3)
    original = _patched_platform(
        worker_module, SimpleNamespace(device_type="txda")
    )
    try:
        worker = SimpleNamespace()
        assert WorkerFL.determine_available_memory(worker) == 5368709120
    finally:
        os.environ.pop("VLLM_TXDA_KV_CACHE_SIZE", None)
        worker_module.current_platform = original


def test_determine_available_memory_txda_default():
    """txda without the env var returns the 20 GiB fallback."""
    os.environ.pop("VLLM_TXDA_KV_CACHE_SIZE", None)
    original = _patched_platform(
        worker_module, SimpleNamespace(device_type="txda")
    )
    try:
        worker = SimpleNamespace()
        assert WorkerFL.determine_available_memory(worker) == 20 * 1024**3
    finally:
        worker_module.current_platform = original


def test_determine_available_memory_returns_kv_cache_memory_bytes():
    """A configured kv_cache_memory_bytes is returned after a profile run."""
    calls = []

    class _Runner:
        def profile_run(self):
            calls.append("profile")

    worker = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_memory_bytes=12345),
        model_runner=_Runner(),
        init_snapshot=SimpleNamespace(free_memory=1),
    )
    original = _patched_platform(
        worker_module, SimpleNamespace(device_type="cuda")
    )
    try:
        result = WorkerFL.determine_available_memory(worker)
    finally:
        worker_module.current_platform = original
    assert result == 12345
    assert calls == ["profile"]
