# Copyright (c) 2026 BAAI. All rights reserved.
"""Execute production memory methods with deterministic CPU memory doubles.

These checks cover accounting, backend isolation and cleanup. They do not
validate native graph capture, the GPU allocator, or model memory capacity.
"""

import __future__

import ast
import gc
import sys
import weakref
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest

SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/worker"
MIB = 1 << 20
CAPTURE_SIZES = list(reversed([1, 2, 4, *range(8, 257, 8), *range(272, 513, 16)]))


class Mode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


def _method(filename, class_name, method_name, namespace):
    path = SOURCE / filename
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(n for n in module.body if getattr(n, "name", None) == class_name)
    method = next(n for n in cls.body if getattr(n, "name", None) == method_name)
    method.decorator_list = []
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        namespace,
    )
    return namespace[method_name]


@pytest.fixture
def profile_factory():
    def make(device="musa", groups=(), *, encoder_mib=0, fail_at=None):
        state = NS(free=10000 * MIB, seen=[], retained=[], enabled=False)
        state.profile_logger_factory = Mock(return_value=Mock())
        counter = NS(num_cudagraph_captured=13)
        wrappers = [NS(graph_pool=object()) for _ in range(3)]
        original_pools = [wrapper.graph_pool for wrapper in wrappers]
        clears = []
        for wrapper in wrappers:
            clear = Mock(side_effect=state.retained.clear)
            clears.append(NS(_all_instances=[wrapper], clear_all_graphs=clear))
        capture_groups = []
        increments = {}
        for mode, samples in groups:
            descs = [
                NS(num_tokens=size, uniform=False)
                for size in CAPTURE_SIZES[: len(samples)]
            ]
            capture_groups.append((mode, descs))
            increments.update(
                {id(d): sample * MIB for d, sample in zip(descs, samples)}
            )
        keys = {mode: {"existing"} for mode, _ in groups}
        encoder = None
        if encoder_mib:

            def capture_encoder(**kwargs):
                assert kwargs["graph_pool"] == "encoder pool"
                assert state.enabled
                if fail_at == "encoder":
                    raise RuntimeError("capture sentinel")
                state.free -= encoder_mib * MIB

            encoder = NS(
                get_num_graphs_to_capture=lambda: 2,
                token_budgets=[8, 16],
                capture=capture_encoder,
                clear=Mock(),
            )

        def capture(desc, **kwargs):
            assert state.enabled
            assert all(w.graph_pool == "decoder pool" for w in wrappers)
            state.seen.append((desc, kwargs, len(state.retained)))
            if len(state.seen) == fail_at:
                raise RuntimeError("capture sentinel")
            state.free -= increments[id(desc)]
            state.retained.append(desc)
            counter.num_cudagraph_captured += 1

        runner = NS(
            device=device + ":0",
            vllm_config=object(),
            _init_minimal_kv_cache_for_profiling=Mock(),
            _create_encoder_cudagraph_manager=lambda: encoder,
            cudagraph_dispatcher=NS(
                get_capture_descs=lambda: capture_groups,
                cudagraph_keys=keys,
                keys_initialized=True,
            ),
            _cleanup_profiling_kv_cache=Mock(),
            _freeze_gc=nullcontext,
            _warmup_and_capture=capture,
            max_model_len=8192,
            max_num_tokens=16384,
            lora_config=object(),
            maybe_remove_all_loras=Mock(),
        )
        pools = iter(
            ["encoder pool"] if device == "musa" else ["decoder pool", "encoder pool"]
        )
        graph_capture = Mock(return_value=nullcontext())
        stream = object()
        namespace = {
            "set_current_vllm_config": lambda _: nullcontext(),
            "compilation_counter": counter,
            "logger": Mock(),
            "init_logger": state.profile_logger_factory,
            "current_platform": NS(
                device_type=device,
                graph_pool_handle=lambda: next(pools),
                get_global_graph_pool=lambda: "decoder pool",
                torch_device_fn=NS(Stream=Mock(return_value=stream)),
                is_rocm=lambda: device == "rocm",
            ),
            "CUDAGraphWrapper": clears[0],
            "GraphWrapper": clears[1],
            "BreakableCUDAGraphWrapper": clears[2],
            "CUDAGraphMode": Mode,
            "set_cudagraph_capturing_enabled": lambda value: setattr(
                state, "enabled", value
            ),
            "graph_capture": graph_capture,
            "GraphCaptureContext": lambda value: NS(stream=value),
            "current_stream": lambda: stream,
            "torch": NS(
                accelerator=NS(
                    synchronize=Mock(),
                    empty_cache=Mock(),
                    get_memory_info=lambda: (state.free, 10000 * MIB),
                )
            ),
        }
        get_context = _method(
            "model_runner.py",
            "ModelRunnerFL",
            "_get_musa_graph_capture_context",
            namespace,
        )
        runner._get_musa_graph_capture_context = lambda: get_context(runner)
        method = _method(
            "model_runner.py", "ModelRunnerFL", "profile_cudagraph_memory", namespace
        )

        def invoke():
            try:
                return method(runner)
            finally:
                runner._init_minimal_kv_cache_for_profiling.assert_called_once()
                runner._cleanup_profiling_kv_cache.assert_called_once()
                assert not state.enabled
                assert not state.retained
                assert [w.graph_pool for w in wrappers] == original_pools
                assert counter.num_cudagraph_captured == 13
                if groups or encoder is not None:
                    for item in clears:
                        item.clear_all_graphs.assert_called_once()
                    assert all(not values for values in keys.values())
                    assert not runner.cudagraph_dispatcher.keys_initialized
                    runner.maybe_remove_all_loras.assert_called_once_with(
                        runner.lora_config
                    )
                    if encoder is not None:
                        encoder.clear.assert_called_once()
                    kwargs = graph_capture.call_args.kwargs
                    assert kwargs["device"] == runner.device
                    context = kwargs["graph_capture_context"]
                    if device in ("rocm", "musa"):
                        assert context.stream is stream
                    else:
                        assert context is None

        return invoke, state

    return make


@pytest.mark.parametrize(
    "device, expected_mib, count",
    [("musa", 135, 4), ("cuda", 160, 2), ("rocm", 160, 2)],
)
def test_shape_dependent_allocations(profile_factory, device, expected_mib, count):
    run, state = profile_factory(device, [(Mode.PIECEWISE, [100, 20, 10, 5])])
    assert run() == expected_mib * MIB
    assert len(state.seen) == count
    assert [retained for _, _, retained in state.seen] == list(range(count))
    assert all(kwargs["profile_seq_lens"] is None for _, kwargs, _ in state.seen)


def test_all_51_original_shapes_are_measured(profile_factory):
    run, state = profile_factory(groups=[(Mode.PIECEWISE, [100, 20] + [1] * 49)])
    assert run() == 169 * MIB
    assert len(state.seen) == 51
    assert [desc.num_tokens for desc, _, _ in state.seen] == CAPTURE_SIZES
    assert state.seen[-1][2] == 50


@pytest.mark.parametrize(
    "device, expected_mib, count", [("musa", 182, 6), ("cuda", 197, 4)]
)
def test_modes_keep_shared_pool_overlap_and_add_encoder(
    profile_factory, device, expected_mib, count
):
    run, state = profile_factory(
        device,
        [(Mode.FULL, [100, 20, 10]), (Mode.PIECEWISE, [120, 10, 5])],
        encoder_mib=17,
    )
    assert run() == expected_mib * MIB
    assert len(state.seen) == count
    assert state.seen[0][1]["profile_seq_lens"] == 32
    assert all(kwargs["profile_seq_lens"] is None for _, kwargs, _ in state.seen[1:])


@pytest.mark.parametrize(
    "samples, expected_mib", [([100], 100), ([100, 0, -3], 102), ([-3, 0], 1)]
)
def test_musa_minimum_driver_cost_and_nonnegative_first(
    profile_factory, samples, expected_mib
):
    run, state = profile_factory(groups=[(Mode.PIECEWISE, samples)])
    assert run() == expected_mib * MIB
    assert len(state.seen) == len(samples)


@pytest.mark.parametrize("encoder_mib", [0, 25])
def test_no_decoder_graphs(profile_factory, encoder_mib):
    run, state = profile_factory(encoder_mib=encoder_mib)
    assert run() == encoder_mib * MIB
    assert state.seen == []


@pytest.mark.parametrize("fail_at", [1, 3, "encoder"])
def test_capture_errors_propagate_and_cleanup_all_state(profile_factory, fail_at):
    run, _ = profile_factory(
        groups=[(Mode.PIECEWISE, [100, 20, 10])], encoder_mib=17, fail_at=fail_at
    )
    with pytest.raises(RuntimeError, match="capture sentinel"):
        run()


@pytest.mark.parametrize(
    "device, non_kv_mib, estimate_enabled, expected_mib, error_count",
    [
        ("musa", 60, True, -10, 1),
        ("musa", 50, True, 0, 1),
        ("musa", 40, True, 10, 0),
        ("cuda", 60, True, -10, 0),
        ("musa", 60, False, 20, 0),
    ],
)
def test_exhausted_budget_diagnostics_preserve_accounting(
    device, non_kv_mib, estimate_enabled, expected_mib, error_count
):
    profile = NS(
        total_consumed=30 * MIB,
        transient_peak_headroom=(non_kv_mib - 30) * MIB,
        non_kv_cache_memory=non_kv_mib * MIB,
        after_profile=NS(free_memory=70 * MIB),
    )
    runner = NS(
        device=device + ":0",
        cache_config=NS(kv_cache_memory_bytes=None, gpu_memory_utilization=0.8),
        init_snapshot=NS(total_memory=100 * MIB, free_memory=100 * MIB),
        requested_memory=80 * MIB,
        model_runner=NS(
            model_memory_usage=30 * MIB,
            profile_run=Mock(),
            profile_cudagraph_memory=Mock(return_value=30 * MIB),
        ),
        vllm_config=NS(compilation_config=NS(cudagraph_mode=Mode.PIECEWISE)),
        model_config=NS(multimodal_config=object()),
        parallel_config=NS(_api_process_count=2),
    )
    logger = Mock()
    reserve = Mock(side_effect=lambda amount, *_: amount)
    method = _method(
        "worker.py",
        "WorkerFL",
        "determine_available_memory",
        {
            "current_platform": NS(
                device_type=device,
                is_cuda_alike=lambda: True,
                torch_device_fn=NS(
                    mem_get_info=Mock(return_value=(70 * MIB, 100 * MIB))
                ),
            ),
            "maybe_apply_startup_plan": Mock(),
            "memory_profiling": lambda *args, **kwargs: nullcontext(profile),
            "CUDAGraphMode": Mode,
            "envs": NS(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=estimate_enabled),
            "logger": logger,
            "init_logger": Mock(return_value=Mock()),
            "format_gib": str,
            "reserve_mm_ipc_gpu_memory": reserve,
        },
    )
    assert method(runner) == expected_mib * MIB
    runner.model_runner.profile_run.assert_called_once()
    runner.model_runner.profile_cudagraph_memory.assert_called_once()
    reserve.assert_called_once_with(
        expected_mib * MIB, runner.model_config.multimodal_config, 2
    )
    assert logger.error.call_count == error_count
    if error_count:
        message, *values = logger.error.call_args.args
        assert values == [
            runner.device,
            80 * MIB,
            non_kv_mib * MIB,
            30 * MIB,
            (non_kv_mib - 30) * MIB,
            30 * MIB,
            30 * MIB,
            expected_mib * MIB,
            100 * MIB,
            70 * MIB,
            100 * MIB,
        ]
        assert "MUSA KV cache memory budget exhausted" in message % tuple(values)


@pytest.fixture
def pool_lifecycle(monkeypatch):
    """Model pools whose blocks survive graph reset, with stream-local reuse."""

    def make(device="musa", late_wrapper=False, fail_at=None, stream_error=False):
        total = 1000 * MIB
        global_pool = object()
        streams = []
        pools = []
        state = NS(
            stream=None,
            enabled=False,
            phase="profile",
            seen=[],
            capacities={},
            used={},
            pool_ids=[],
            contexts=[],
            wrapper=None,
            cleanup_count=0,
        )
        descs = [NS(num_tokens=n, uniform=False) for n in (4096, 1024, 256)]
        sizes = {id(desc): value * MIB for desc, value in zip(descs, (100, 20, 10))}
        counter = NS(num_cudagraph_captured=7, num_gpu_runner_capture_triggers=0)
        sentinel = RuntimeError("capture sentinel")

        def new_stream(**kwargs):
            assert kwargs == {"device": device + ":0"}
            if stream_error:
                raise sentinel
            stream = object()
            streams.append(stream)
            return stream

        def new_pool():
            pool = object()
            pools.append(pool)
            return pool

        platform = NS(
            device_type=device,
            graph_pool_handle=new_pool,
            get_global_graph_pool=lambda: global_pool,
            is_rocm=lambda: device == "rocm",
            torch_device_fn=NS(Stream=new_stream),
        )

        @contextmanager
        def capture_context(*, device, graph_capture_context=None):
            context = graph_capture_context
            if context is None:
                context = NS(stream=new_stream(device=device))
            state.contexts.append(context)
            previous, state.stream = state.stream, context.stream
            try:
                yield context
            finally:
                state.stream = previous

        # Use the production wrapper constructor and clearing method: wrappers
        # born during first tracing must start in the same actual global pool.
        class Wrapper:
            _all_instances = weakref.WeakSet()

            @classmethod
            def clear_all_graphs(cls):
                for wrapper in list(cls._all_instances):
                    wrapper.clear_graphs()

        wrapper_namespace = {
            "current_platform": platform,
            "envs": NS(VLLM_LOGGING_LEVEL="INFO"),
            "CUDAGraphMode": Mode,
            "GraphWrapper": Wrapper,
            "GraphOptions": NS,
        }
        Wrapper.__init__ = _method(
            "../compilation/graph.py", "GraphWrapper", "__init__", wrapper_namespace
        )
        Wrapper.clear_graphs = _method(
            "../compilation/graph.py", "GraphWrapper", "clear_graphs", {}
        )
        config = NS(
            compilation_config=NS(cudagraph_mode=Mode.PIECEWISE),
            profiler_config=NS(capture_torch_profiler=False),
        )

        def wrapper():
            if state.wrapper is None:
                state.wrapper = Wrapper(lambda: None, config, Mode.PIECEWISE)
            return state.wrapper

        if not late_wrapper:
            wrapper()

        class Allocation:
            def __init__(self, key, amount):
                self.key, self.amount = key, amount
                state.used[key] = state.used.get(key, 0) + amount
                state.capacities[key] = max(
                    state.capacities.get(key, 0), state.used[key]
                )

            def __del__(self):
                state.used[self.key] -= self.amount

        def capture(desc, **kwargs):
            assert state.enabled
            item = wrapper()
            key = (item.graph_pool, state.stream)
            state.pool_ids.append(item.graph_pool)
            state.seen.append((state.phase, desc, kwargs))
            if state.phase == "profile" and len(state.seen) == fail_at:
                raise sentinel
            item.concrete_graph_entries[id(desc)] = Allocation(key, sizes[id(desc)])
            counter.num_cudagraph_captured += 1

        def cleanup():
            state.cleanup_count += 1
            gc.collect()
            # CUDA releases discarded private pools; the MUSA 2.9 reset path
            # omits releasePool, so this double deliberately retains blocks.
            if device != "musa":
                for key in list(state.capacities):
                    if state.used[key] == 0:
                        del state.capacities[key]

        empty_wrappers = NS(_all_instances=(), clear_all_graphs=Mock())
        namespace = {
            "current_platform": platform,
            "compilation_counter": counter,
            "set_current_vllm_config": lambda _: nullcontext(),
            "logger": Mock(),
            "init_logger": Mock(return_value=Mock()),
            "GraphWrapper": Wrapper,
            "CUDAGraphWrapper": empty_wrappers,
            "BreakableCUDAGraphWrapper": empty_wrappers,
            "CUDAGraphMode": Mode,
            "GraphCaptureContext": lambda s: NS(stream=s),
            "graph_capture": capture_context,
            "current_stream": lambda: "rocm stream",
            "set_cudagraph_capturing_enabled": lambda v: setattr(state, "enabled", v),
            "torch": NS(
                accelerator=NS(
                    synchronize=Mock(),
                    empty_cache=Mock(),
                    get_memory_info=lambda: (
                        total - sum(state.capacities.values()),
                        total,
                    ),
                )
            ),
            "nullcontext": nullcontext,
            "lock_workspace": Mock(),
            "time": NS(perf_counter=Mock(side_effect=range(10))),
        }
        runner = NS(
            device=device + ":0",
            vllm_config=config,
            compilation_config=config.compilation_config,
            _init_minimal_kv_cache_for_profiling=Mock(),
            _create_encoder_cudagraph_manager=lambda: None,
            _maybe_init_encoder_cudagraph_manager=Mock(),
            encoder_cudagraph_manager=None,
            _cleanup_profiling_kv_cache=cleanup,
            _freeze_gc=nullcontext,
            _warmup_and_capture=capture,
            max_model_len=8192,
            max_num_tokens=16384,
            lora_config=None,
            maybe_remove_all_loras=Mock(),
            cudagraph_dispatcher=NS(
                get_capture_descs=lambda: [(Mode.PIECEWISE, descs)],
                cudagraph_keys={Mode.PIECEWISE: {"before"}},
                keys_initialized=True,
            ),
        )
        runner._capture_cudagraphs = lambda **kw: [
            capture(d, cudagraph_runtime_mode=kw["cudagraph_runtime_mode"])
            for d in kw["batch_descriptors"]
        ]
        get_context = _method(
            "model_runner.py",
            "ModelRunnerFL",
            "_get_musa_graph_capture_context",
            namespace,
        )
        runner._get_musa_graph_capture_context = lambda: get_context(runner)
        profile = _method(
            "model_runner.py", "ModelRunnerFL", "profile_cudagraph_memory", namespace
        )
        formal = _method("model_runner.py", "ModelRunnerFL", "capture_model", namespace)
        parallel = ModuleType("vllm.distributed.parallel_state")
        parallel.get_world_group = lambda: NS(local_rank=1)
        monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", parallel)

        def run_profile():
            try:
                return profile(runner)
            finally:
                assert not state.enabled
                assert state.stream is None
                assert state.cleanup_count == 1
                assert counter.num_cudagraph_captured == 7
                assert not runner.cudagraph_dispatcher.keys_initialized
                assert not runner.cudagraph_dispatcher.cudagraph_keys[Mode.PIECEWISE]
                assert wrapper().graph_pool is global_pool
                assert not wrapper().concrete_graph_entries
                assert not any(state.used.values())

        def run_formal():
            state.phase = "formal"
            return formal(runner)

        return NS(
            profile=run_profile,
            formal=run_formal,
            state=state,
            sentinel=sentinel,
            global_pool=global_pool,
            streams=streams,
            pools=pools,
        )

    return make


@pytest.mark.parametrize("late_wrapper", [False, True])
def test_musa_profile_and_formal_capture_reuse_pool_and_stream(
    pool_lifecycle, late_wrapper
):
    case = pool_lifecycle(late_wrapper=late_wrapper)
    # Independent byte oracle: all three original retained shape allocations.
    assert case.profile() == 130 * MIB
    assert sum(case.state.capacities.values()) == 130 * MIB
    assert case.formal() == 0
    assert sum(case.state.capacities.values()) == 130 * MIB
    assert len(case.streams) == 1
    assert case.state.contexts[0] is case.state.contexts[1]
    assert all(pool is case.global_pool for pool in case.state.pool_ids)
    assert len(case.state.seen) == 6
    assert not case.state.enabled
    assert case.state.stream is None


@pytest.mark.parametrize("late_wrapper", [False, True])
def test_musa_pool_cleanup_preserves_original_error(pool_lifecycle, late_wrapper):
    case = pool_lifecycle(late_wrapper=late_wrapper, fail_at=2)
    with pytest.raises(RuntimeError) as error:
        case.profile()
    assert error.value is case.sentinel
    # Cached blocks survive graph cleanup but can be reused by formal capture.
    assert sum(case.state.capacities.values()) == 100 * MIB
    assert case.formal() == 30 * MIB
    assert sum(case.state.capacities.values()) == 130 * MIB
    assert case.state.contexts[0] is case.state.contexts[1]


@pytest.mark.parametrize("device", ["cuda", "rocm", "npu"])
def test_other_backends_keep_temporary_profile_pool(pool_lifecycle, device):
    case = pool_lifecycle(device=device)
    assert case.profile() == 140 * MIB  # Two samples and original extrapolation.
    assert not case.state.capacities
    assert all(pool is not case.global_pool for pool in case.state.pool_ids)
    assert case.formal() == 130 * MIB
    assert all(pool is case.global_pool for pool in case.state.pool_ids[2:])
    assert case.state.contexts[0] is not case.state.contexts[1]


@pytest.mark.parametrize("late_wrapper", [False, True])
def test_profile_stream_creation_failure_cleans_up(pool_lifecycle, late_wrapper):
    case = pool_lifecycle(late_wrapper=late_wrapper, stream_error=True)
    with pytest.raises(RuntimeError) as error:
        case.profile()
    assert error.value is case.sentinel
    assert not case.state.contexts
    assert not case.state.capacities
    assert not case.state.seen


def test_formal_stream_creation_failure_does_not_enable_capture(pool_lifecycle):
    case = pool_lifecycle(stream_error=True)
    with pytest.raises(RuntimeError) as error:
        case.formal()
    assert error.value is case.sentinel
    assert not case.state.enabled
    assert not case.state.contexts
    assert not case.state.capacities
