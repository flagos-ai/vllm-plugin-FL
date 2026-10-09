# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.config import CUDAGraphMode
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda()
    and getattr(current_platform, "vendor_name", None) != "kunlunxin",
    reason="Graph memory profiling is enabled only on CUDA and Kunlunxin",
)
@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("fail_capture", [False, True])
def test_graph_memory_preserves_pool_policy_and_cleans_up(
    monkeypatch, private, fail_capture
):
    import vllm_fl.worker.model_runner as module

    mib = 1 << 20
    shared_pool = object()
    profiling_pool = object()
    shared = SimpleNamespace(graph_pool=shared_pool)
    full = SimpleNamespace(graph_pool=None if private else shared_pool)
    graph_wrapper = SimpleNamespace(
        _all_instances=[shared], clear_all_graphs=MagicMock()
    )
    breakable = SimpleNamespace(_all_instances=[full], clear_all_graphs=MagicMock())
    monkeypatch.setattr(module, "GraphWrapper", graph_wrapper)
    monkeypatch.setattr(module, "BreakableCUDAGraphWrapper", breakable)
    free = [1000 * mib]
    platform = SimpleNamespace(
        graph_pool_handle=lambda: profiling_pool,
        torch_device_fn=SimpleNamespace(mem_get_info=lambda: (free[0], 1000 * mib)),
    )
    monkeypatch.setattr(module, "current_platform", platform)
    monkeypatch.setattr(module, "_accelerator_synchronize", lambda: None)
    monkeypatch.setattr(module.torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(module, "graph_capture", lambda **kwargs: nullcontext())
    monkeypatch.setattr(module, "set_current_vllm_config", lambda config: nullcontext())
    capture_enabled = MagicMock()
    monkeypatch.setattr(module, "set_cudagraph_capturing_enabled", capture_enabled)
    counter = SimpleNamespace(num_cudagraph_captured=7)
    monkeypatch.setattr(module, "compilation_counter", counter)
    descs = [SimpleNamespace(num_tokens=n) for n in (8, 4, 2, 1)]
    keys = {CUDAGraphMode.FULL: {1}, CUDAGraphMode.PIECEWISE: {2}}
    dispatcher = SimpleNamespace(
        get_capture_descs=lambda: [
            (CUDAGraphMode.PIECEWISE, descs),
            (CUDAGraphMode.FULL, descs),
        ],
        cudagraph_keys=keys,
        keys_initialized=True,
    )
    calls = []

    def capture(desc, *, cudagraph_runtime_mode, profile_seq_lens):
        assert shared.graph_pool is profiling_pool
        assert full.graph_pool is (None if private else profiling_pool)
        calls.append((cudagraph_runtime_mode, desc.num_tokens))
        counter.num_cudagraph_captured += 1
        if fail_capture:
            raise RuntimeError("capture failed")
        # Each FULL descriptor includes all independently allocated segments.
        cost = desc.num_tokens * (
            3 if cudagraph_runtime_mode == CUDAGraphMode.FULL else 1
        )
        free[0] -= cost * mib

    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(),
        _init_minimal_kv_cache_for_profiling=MagicMock(),
        _cleanup_profiling_kv_cache=MagicMock(),
        cudagraph_dispatcher=dispatcher,
        _create_encoder_cudagraph_manager=lambda: None,
        _freeze_gc=nullcontext,
        device="cpu",
        _warmup_and_capture=capture,
        max_model_len=128,
        max_num_tokens=256,
        maybe_remove_all_loras=MagicMock(),
        lora_config=None,
    )
    if fail_capture:
        with pytest.raises(RuntimeError, match="capture failed"):
            module.ModelRunnerFL.profile_cudagraph_memory(runner)
    else:
        result = module.ModelRunnerFL.profile_cudagraph_memory(runner)
        # Shared: max(8,24) + 3*4 + 3*12. Private: shared(8+3*4) + 3*(8+4+2+1).
        assert result == (65 if private else 72) * mib
        full_sizes = [size for mode, size in calls if mode == CUDAGraphMode.FULL]
        assert full_sizes == ([8, 4, 2, 1] if private else [8, 4])
    assert shared.graph_pool is shared_pool
    assert full.graph_pool is (None if private else shared_pool)
    assert counter.num_cudagraph_captured == 7
    assert not dispatcher.keys_initialized
    assert all(not values for values in keys.values())
    runner._cleanup_profiling_kv_cache.assert_called_once()
    graph_wrapper.clear_all_graphs.assert_called_once()
    breakable.clear_all_graphs.assert_called_once()
    assert [call.args[0] for call in capture_enabled.call_args_list] == [True, False]


@pytest.mark.parametrize(
    "vendor,mode,enabled,manual,expected_profile",
    [
        ("kunlunxin", CUDAGraphMode.FULL_AND_PIECEWISE, True, 0, True),
        ("kunlunxin", CUDAGraphMode.FULL_DECODE_ONLY, True, 0, True),
        ("kunlunxin", CUDAGraphMode.NONE, True, 0, False),
        ("kunlunxin", CUDAGraphMode.FULL_AND_PIECEWISE, False, 0, True),
        ("kunlunxin", CUDAGraphMode.FULL_AND_PIECEWISE, True, 123, False),
        ("nvidia", CUDAGraphMode.FULL_AND_PIECEWISE, True, 0, True),
        ("musa", CUDAGraphMode.FULL_AND_PIECEWISE, True, 0, False),
    ],
)
def test_worker_reserves_graph_budget(
    monkeypatch, vendor, mode, enabled, manual, expected_profile
):
    import vllm_fl.worker.worker as module

    platform = SimpleNamespace(
        vendor_name=vendor,
        device_type="cuda" if vendor != "musa" else "musa",
        is_cuda=lambda: vendor == "nvidia",
        empty_cache=lambda: None,
        torch_device_fn=SimpleNamespace(
            reset_peak_memory_stats=lambda: None,
            memory_stats=lambda: {"allocated_bytes.all.peak": 200},
        ),
    )
    monkeypatch.setattr(module, "current_platform", platform)
    monkeypatch.setattr(
        module.envs, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", enabled
    )
    result = SimpleNamespace(
        before_profile=SimpleNamespace(torch_peak=50),
        after_profile=SimpleNamespace(free_memory=600),
        non_torch_increase=20,
        weights_memory=100,
    )
    monkeypatch.setattr(
        module, "memory_profiling_fl", lambda *a, **kw: nullcontext(result)
    )
    runner = SimpleNamespace(
        profile_run=MagicMock(),
        profile_cudagraph_memory=MagicMock(return_value=80),
        model_memory_usage=100,
    )
    worker = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=manual, gpu_memory_utilization=0.9
        ),
        init_snapshot=SimpleNamespace(free_memory=1000, total_memory=1000),
        requested_memory=900,
        model_runner=runner,
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=mode)
        ),
    )
    budget = module.WorkerFL.determine_available_memory(worker)
    assert runner.profile_cudagraph_memory.call_count == int(expected_profile)
    runner.profile_run.assert_called_once()
    assert budget == (manual or 630 - (80 if expected_profile and enabled else 0))
