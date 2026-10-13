# Copyright (c) 2026 BAAI. All rights reserved.
"""Execute production memory methods with deterministic CPU memory doubles.

These checks cover accounting, backend isolation and cleanup. They do not
validate native graph capture, the GPU allocator, or model memory capacity.
"""

import __future__

import ast
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
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
            descs = [NS(num_tokens=size) for size in CAPTURE_SIZES[: len(samples)]]
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
        pools = iter(["decoder pool", "encoder pool"])
        graph_capture = Mock(return_value=nullcontext())
        stream = object()
        namespace = {
            "set_current_vllm_config": lambda _: nullcontext(),
            "compilation_counter": counter,
            "logger": Mock(),
            "current_platform": NS(
                device_type=device,
                graph_pool_handle=lambda: next(pools),
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
                    if device == "rocm":
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
            "current_platform": NS(device_type=device, is_cuda_alike=lambda: True),
            "maybe_apply_startup_plan": Mock(),
            "memory_profiling": lambda *args, **kwargs: nullcontext(profile),
            "CUDAGraphMode": Mode,
            "envs": NS(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=estimate_enabled),
            "logger": logger,
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
