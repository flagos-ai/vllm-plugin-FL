# Copyright (c) 2025 BAAI. All rights reserved.

"""Tests for worker output handling on non-generate runners."""

from types import SimpleNamespace


class _PoolingOutput:
    """Stand-in for a pooling/embedding runner output."""


def _worker_with_output(worker_module, output):
    worker = object.__new__(worker_module.WorkerFL)
    worker.model_runner = SimpleNamespace(
        execute_model=lambda scheduler_output, intermediate_tensors: output
    )
    worker.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False)
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    worker.profiler = None
    return worker


def _scheduler_output():
    return SimpleNamespace(
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"req": 1},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
    )


def test_execute_model_returns_pooling_output(monkeypatch):
    import vllm_fl.worker.worker as worker_module

    group = SimpleNamespace(is_first_rank=True, is_last_rank=True, rank_in_group=0)
    monkeypatch.setattr(worker_module, "get_pp_group", lambda: group)
    monkeypatch.setattr(worker_module, "get_tp_group", lambda: group)

    expected = _PoolingOutput()
    worker = _worker_with_output(worker_module, expected)

    assert worker.execute_model(_scheduler_output()) is expected
