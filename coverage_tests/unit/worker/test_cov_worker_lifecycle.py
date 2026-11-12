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

"""Coverage tests for vllm_fl/worker/worker.py profiler and teardown paths.

``profile`` (all six branches), the KV-connector handshake metadata,
``init_worker_distributed_environment`` argument forwarding,
``save_sharded_state``, ``shutdown``, and the compile-range completion
half of ``compile_or_warm_up_model`` are driven here through unbound
WorkerFL method calls on SimpleNamespace workers — the same pattern the
dev tests (tests/unit_tests/worker/test_worker.py) use. Per CONTRACT R2
the vllm engine objects, distributed helpers, profiler wrappers, and
module-level collaborators are inputs replaced with recorders; nothing
on WorkerFL itself is patched to select a branch.
"""

from contextlib import contextmanager
from types import SimpleNamespace

# The shim import is load-bearing: it must precede vllm_fl/vllm
# imports on hosts without triton. Do not alphabetize.
# isort: split
from . import _imports  # noqa: F401  (installs local import shims first)

# isort: split
import vllm_fl.worker.worker as worker_module
from vllm_fl.worker.worker import WorkerFL


class _Patch:
    """Deterministic set/restore of a module attribute (R2 input boundary)."""

    def __init__(self, module, name, value):
        self.module = module
        self.name = name
        self.value = value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def test_profile_raises_when_profiling_disabled():
    """A missing profiler config or profiler value refuses to start."""
    raised = 0
    for config in (None, SimpleNamespace(profiler=None)):
        try:
            WorkerFL.profile(SimpleNamespace(profiler_config=config))
        except RuntimeError:
            raised += 1
    assert raised == 2, "disabled profiler did not raise RuntimeError"


def test_profile_start_torch_builds_torch_profiler_wrapper():
    """start with type torch wraps TorchProfilerWrapper and starts it."""
    started = []

    class _TorchWrapper:
        def __init__(self, cfg, worker_name=None, local_rank=None, activities=None):
            self.cfg, self.worker_name = cfg, worker_name
            self.local_rank, self.activities = local_rank, activities

        def start(self):
            started.append("torch")

    import vllm.distributed.utils as dist_utils

    dist_patch = _Patch(
        dist_utils, "get_worker_rank_suffix", lambda global_rank: f"rank{global_rank}"
    )
    wrapper_patch = _Patch(worker_module, "TorchProfilerWrapper", _TorchWrapper)
    with dist_patch, wrapper_patch:
        worker = SimpleNamespace(
            profiler_config=SimpleNamespace(profiler="torch"),
            profiler=None,
            rank=3,
            local_rank=1,
        )
        WorkerFL.profile(worker, is_start=True, profile_prefix="pre")

    assert started == ["torch"]
    assert worker.profiler.worker_name == "pre_rank3"
    assert worker.profiler.local_rank == 1
    assert worker.profiler.activities == ["CPU", "CUDA"]


def test_profile_start_cuda_builds_cuda_profiler_wrapper():
    """start with type cuda instantiates CudaProfilerWrapper with the config."""
    started = []

    class _CudaWrapper:
        def __init__(self, cfg):
            self.cfg = cfg

        def start(self):
            started.append("cuda")

    with _Patch(worker_module, "CudaProfilerWrapper", _CudaWrapper):
        worker = SimpleNamespace(
            profiler_config=SimpleNamespace(profiler="cuda"),
            profiler=None,
            rank=0,
            local_rank=0,
        )
        WorkerFL.profile(worker, is_start=True)

    assert started == ["cuda"]
    assert worker.profiler.cfg is worker.profiler_config


def test_profile_start_rejects_unknown_type():
    """An unknown profiler type raises ValueError naming the value."""
    worker = SimpleNamespace(
        profiler_config=SimpleNamespace(profiler="badtype"),
        profiler=None,
        rank=0,
        local_rank=0,
    )
    try:
        WorkerFL.profile(worker, is_start=True)
    except ValueError as exc:
        assert "Invalid profiler value of badtype" in str(exc)
        return
    raise AssertionError("unknown profiler type did not raise ValueError")


def test_profile_stop_without_start_is_a_noop():
    """stop with no active profiler only warns and returns."""
    WorkerFL.profile(
        SimpleNamespace(
            profiler_config=SimpleNamespace(profiler="torch"), profiler=None
        ),
        is_start=False,
    )
    # Reaching this point without an exception proves the no-op branch;
    # stop must also leave the worker's profiler untouched (still None).
    worker = SimpleNamespace(
        profiler_config=SimpleNamespace(profiler="torch"), profiler=None
    )
    WorkerFL.profile(worker, is_start=False)
    assert worker.profiler is None


def test_profile_stop_stops_the_active_profiler():
    """stop forwards to the active profiler wrapper."""
    stopped = []
    worker = SimpleNamespace(
        profiler_config=SimpleNamespace(profiler="torch"),
        profiler=SimpleNamespace(stop=lambda: stopped.append(1)),
    )
    WorkerFL.profile(worker, is_start=False)
    assert stopped == [1]


def test_get_kv_connector_handshake_metadata_paths():
    """No group / None metadata / present metadata cover all three paths."""

    def patch(name, fn):
        return _Patch(worker_module, name, fn)

    worker = SimpleNamespace()
    with patch("has_kv_transfer_group", lambda: False):
        assert WorkerFL.get_kv_connector_handshake_metadata(worker) is None

    metadata = {"k": "v"}
    with (
        patch("has_kv_transfer_group", lambda: True),
        patch(
            "get_kv_transfer_group",
            lambda: SimpleNamespace(get_handshake_metadata=lambda: None),
        ),
    ):
        assert WorkerFL.get_kv_connector_handshake_metadata(worker) is None

    with (
        patch("has_kv_transfer_group", lambda: True),
        patch(
            "get_kv_transfer_group",
            lambda: SimpleNamespace(get_handshake_metadata=lambda: metadata),
        ),
        patch("get_tp_group", lambda: SimpleNamespace(rank_in_group=5)),
    ):
        result = WorkerFL.get_kv_connector_handshake_metadata(worker)
    assert result == {5: metadata}


def test_init_worker_distributed_environment_forwards_all_arguments():
    """Every parallel/size argument reaches the vllm distributed helpers."""
    import vllm.model_executor.layers.batch_invariant as batch_invariant

    fwd = {}

    with (
        _Patch(
            worker_module,
            "set_custom_all_reduce",
            lambda v: fwd.update(custom_all_reduce=v),
        ),
        _Patch(
            worker_module,
            "init_distributed_environment",
            lambda ws, rank, init_method, local_rank, backend: fwd.update(
                world_size=ws,
                rank=rank,
                init_method=init_method,
                local_rank=local_rank,
                backend=backend,
            ),
        ),
        _Patch(
            worker_module,
            "ensure_model_parallel_initialized",
            lambda tp, pp, pcp, dcp: fwd.update(tp=tp, pp=pp, pcp=pcp, dcp=dcp),
        ),
        _Patch(
            worker_module,
            "ensure_ec_transfer_initialized",
            lambda cfg: fwd.update(ec=cfg),
        ),
        _Patch(
            batch_invariant,
            "init_batch_invariance",
            lambda: fwd.update(batch_inv=True),
        ),
    ):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                world_size=16,
                disable_custom_all_reduce=True,
                tensor_parallel_size=2,
                pipeline_parallel_size=3,
                prefill_context_parallel_size=4,
                decode_context_parallel_size=5,
            )
        )
        worker_module.init_worker_distributed_environment(
            config, 7, "tcp://host:1234", 2, "gloo"
        )

    assert fwd["custom_all_reduce"] is False
    assert fwd["world_size"] == 16
    assert fwd["rank"] == 7
    assert fwd["init_method"] == "tcp://host:1234"
    assert fwd["local_rank"] == 2
    assert fwd["backend"] == "gloo"
    assert (fwd["tp"], fwd["pp"], fwd["pcp"], fwd["dcp"]) == (2, 3, 4, 5)
    assert fwd["ec"] is config
    assert fwd["batch_inv"] is True


def test_init_worker_distributed_environment_defaults():
    """Omitted init method and backend default to env:// and nccl."""
    import vllm.model_executor.layers.batch_invariant as batch_invariant

    fwd = {}
    with (
        _Patch(
            worker_module,
            "set_custom_all_reduce",
            lambda v: None,
        ),
        _Patch(
            worker_module,
            "init_distributed_environment",
            lambda ws, rank, init_method, local_rank, backend: fwd.update(
                init_method=init_method,
                backend=backend,
            ),
        ),
        _Patch(
            worker_module,
            "ensure_model_parallel_initialized",
            lambda *a: None,
        ),
        _Patch(
            worker_module,
            "ensure_ec_transfer_initialized",
            lambda cfg: None,
        ),
        _Patch(
            batch_invariant,
            "init_batch_invariance",
            lambda: None,
        ),
    ):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                world_size=1,
                disable_custom_all_reduce=False,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
            )
        )
        worker_module.init_worker_distributed_environment(config, 1)

    assert fwd["init_method"] == "env://"
    assert fwd["backend"] == "nccl"


def test_save_sharded_state_forwards_path_pattern_and_max_size():
    """save_sharded_state passes every kwarg to ShardedStateLoader."""
    import vllm.model_executor.model_loader as model_loader

    saved = {}

    class _Loader:
        @staticmethod
        def save_model(model, path, pattern=None, max_size=None):
            saved.update(model=model, path=path, pattern=pattern, max_size=max_size)

    with _Patch(model_loader, "ShardedStateLoader", _Loader):
        worker = SimpleNamespace(model_runner=SimpleNamespace(model="MODELX"))
        WorkerFL.save_sharded_state(worker, "/tmp/out", pattern="p*", max_size=5)

    assert saved == {
        "model": "MODELX",
        "path": "/tmp/out",
        "pattern": "p*",
        "max_size": 5,
    }


@contextmanager
def _kv_shutdown_recorder():
    """Record ensure_kv_transfer_shutdown calls and restore afterwards."""
    from vllm.distributed.kv_transfer import ensure_kv_transfer_shutdown

    order = []
    worker_module.ensure_kv_transfer_shutdown = lambda: order.append("kv")
    try:
        yield order
    finally:
        worker_module.ensure_kv_transfer_shutdown = ensure_kv_transfer_shutdown


def test_shutdown_runs_kv_profiler_and_runner_in_order():
    """shutdown tears down the kv connector, profiler, and model runner."""
    with _kv_shutdown_recorder() as order:
        worker = SimpleNamespace(
            profiler=SimpleNamespace(shutdown=lambda: order.append("profiler")),
            model_runner=SimpleNamespace(shutdown=lambda: order.append("runner")),
        )
        WorkerFL.shutdown(worker)
        assert order == ["kv", "profiler", "runner"]


def test_shutdown_without_profiler_or_runner_only_tears_down_kv():
    """A minimal worker with no profiler/model_runner still shuts down."""
    with _kv_shutdown_recorder() as order:
        WorkerFL.shutdown(SimpleNamespace(profiler=None))
        assert order == ["kv"]


class _Range:
    """Closed interval matching compile_range membership tests."""

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __contains__(self, value):
        return self.start <= value <= self.end


def test_compile_or_warm_up_model_appends_range_ends_descending():
    """Compile ranges without any warmup size are completed by range end.

    compile_sizes=[7] with ranges [20,32] and [40,48] contains no overlap,
    so both range ends are appended and every dummy run happens in
    descending order: 48, 32, 7.
    """
    from vllm.config import CUDAGraphMode
    from vllm.config.compilation import CompilationMode

    runner = SimpleNamespace(
        _dummy_run=lambda n, **kw: ("h", "lh"),
        maybe_remove_all_loras=lambda cfg: None,
        lora_config=None,
        is_pooling_model=False,
        _dummy_sampler_run=lambda **kw: None,
        capture_model=lambda: 0,
        model_memory_usage=0,
    )
    compilation = SimpleNamespace(
        mode=CompilationMode.VLLM_COMPILE,
        compile_sizes=[7],
        cudagraph_mode=CUDAGraphMode.NONE,
        cudagraph_capture_sizes=None,
        get_compile_ranges=lambda: [_Range(20, 32), _Range(40, 48)],
        compilation_time=1.5,
        encoder_compilation_time=0.25,
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(compilation_config=compilation),
        compilation_config=compilation,
        model_runner=runner,
        cache_config=SimpleNamespace(kv_cache_memory_bytes=1),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=32),
        model_config=SimpleNamespace(seed=11),
    )
    seen_sizes = []
    runner._dummy_run = lambda num_tokens, **kw: (
        seen_sizes.append(num_tokens) or ("h", "lh")
    )

    with (
        _Patch(
            worker_module,
            "current_platform",
            SimpleNamespace(device_type="cuda", vendor_name="nvidia"),
        ),
        _Patch(worker_module, "kernel_warmup", lambda w: None),
        _Patch(
            worker_module, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
        ),
        _Patch(
            worker_module,
            "set_random_seed",
            lambda seed: seen_sizes.append(("seed", seed)),
        ),
    ):
        times = WorkerFL.compile_or_warm_up_model(worker)

    assert seen_sizes[:3] == [48, 32, 7]
    assert seen_sizes[-1] == ("seed", 11)
    assert times.language_model == 1.5
    assert times.encoder == 0.25
