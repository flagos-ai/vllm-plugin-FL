# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope and lifecycle regressions; numerical/Graph checks run on real hardware."""

from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_fl.distributed import kimi_k3_pp as feedback
from vllm_fl.worker import kimi_k3 as runner_mod


def make_handler():
    handler = feedback.HostPackedPPHandler.__new__(feedback.HostPackedPPHandler)
    handler.feedback_executor = Mock()
    handler.feedback_cpu_group = object()
    handler.feedback_send_futures = deque()
    handler.feedback_recv_futures = deque()
    handler.queue = deque([object()])
    handler.broadcast_stream = Mock()
    return handler


def test_feedback_failure_still_drains_and_destroys_owned_resources(monkeypatch):
    handler = make_handler()
    pool, group = handler.feedback_executor, handler.feedback_cpu_group
    failed = Future()
    failed.set_exception(RuntimeError("send failed"))
    remaining = Mock()
    handler.feedback_send_futures.append(failed)
    handler.feedback_recv_futures.append(remaining)
    destroy = Mock()
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy)
    with pytest.raises(RuntimeError, match="send failed"):
        handler.close()
    remaining.result.assert_called_once()
    handler.broadcast_stream.synchronize.assert_called_once()
    pool.shutdown.assert_called_once_with(wait=True)
    destroy.assert_called_once_with(group)
    assert handler.feedback_executor is None and handler.feedback_cpu_group is None
    assert not handler.queue and not handler.feedback_recv_futures
    handler.close()
    destroy.assert_called_once()


def test_feedback_pool_creation_failure_releases_the_new_group(monkeypatch):
    handler = make_handler()
    group = object()
    monkeypatch.setattr(feedback, "_make_feedback_cpu_group", lambda: group)
    monkeypatch.setattr(
        feedback, "ThreadPoolExecutor", Mock(side_effect=RuntimeError("pool failed"))
    )
    destroy = Mock()
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy)
    with pytest.raises(RuntimeError, match="pool failed"):
        handler._init_broadcast_transport()
    destroy.assert_called_once_with(group)
    assert handler.feedback_cpu_group is None


def test_group_creation_failure_releases_only_the_owned_group(monkeypatch):
    group = object()
    monkeypatch.setattr(
        feedback,
        "get_pp_group",
        lambda: SimpleNamespace(rank=0, group_ranks=[[0, 1], [2, 3]]),
    )
    monkeypatch.setattr(
        torch.distributed,
        "new_group",
        Mock(side_effect=[group, RuntimeError("next group failed")]),
    )
    destroy = Mock()
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy)
    with pytest.raises(RuntimeError, match="next group failed"):
        feedback._make_feedback_cpu_group()
    destroy.assert_called_once_with(group)


@pytest.mark.parametrize("failing_resource", [None, "feedback", "workspace"])
def test_runner_shutdown_releases_feedback_then_workspace_then_native_state(
    monkeypatch, failing_resource
):
    events = []
    runner = runner_mod.K3ModelRunner.__new__(runner_mod.K3ModelRunner)
    handler = make_handler()

    def close(name):
        events.append(name)
        if name == failing_resource:
            raise RuntimeError(name + " failed")

    handler.close = lambda: close("feedback")
    workspace = SimpleNamespace(close=lambda: close("workspace"))
    runner.pp_handler = handler
    runner._tp8_prefill_workspace = workspace
    monkeypatch.setattr(
        runner_mod.GPUModelRunner, "shutdown", lambda _: events.append("native")
    )
    if failing_resource:
        with pytest.raises(RuntimeError, match=failing_resource):
            runner.shutdown()
    else:
        runner.shutdown()
    assert events == ["feedback", "workspace", "native"]
    assert runner.pp_handler is None and runner._tp8_prefill_workspace is None
    runner.shutdown()
    assert events == ["feedback", "workspace", "native", "native"]


def test_failed_runner_initialization_releases_its_feedback(monkeypatch):
    handler = make_handler()
    handler.close = Mock()

    def fail_init(self, *args, **kwargs):
        self.pp_handler = handler
        raise RuntimeError("runner failed")

    monkeypatch.setattr(runner_mod.GPUModelRunner, "__init__", fail_init)
    monkeypatch.setattr(
        runner_mod,
        "execution_config",
        lambda _: SimpleNamespace(enabled=True, host_pp_feedback=True),
    )
    with pytest.raises(RuntimeError, match="runner failed"):
        runner_mod.K3ModelRunner(
            SimpleNamespace(weight_transfer_config=None),
            torch.device("cpu"),
            diagnostics=Mock(),
        )
    handler.close.assert_called_once()


def test_failed_profile_releases_previously_reserved_resources(monkeypatch):
    runner = runner_mod.K3ModelRunner.__new__(runner_mod.K3ModelRunner)
    runner.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=0)
    )
    handler = make_handler()
    handler.close = Mock()
    runner.pp_handler = handler
    workspace = Mock()
    runner._tp8_prefill_workspace = workspace
    monkeypatch.setattr(
        runner_mod.GPUModelRunner,
        "profile_run",
        Mock(side_effect=RuntimeError("profile failed")),
    )
    with pytest.raises(RuntimeError, match="profile failed"):
        runner.profile_run()
    handler.close.assert_called_once()
    workspace.close.assert_called_once()
    assert runner._tp8_prefill_workspace is None


def test_closed_workspace_cannot_be_reopened_or_borrowed(monkeypatch):
    workspace = runner_mod.TP8PrefillWorkspace(torch.device("cuda:0"))
    sync = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    workspace.close()
    workspace.close()
    assert workspace.describe() == []
    sync.assert_not_called()
    with pytest.raises(RuntimeError, match="closed"):
        workspace.reserve()
    with pytest.raises(RuntimeError, match="closed"):
        workspace.get(torch.device("cuda:0"))


def test_feedback_drain_preserves_multiple_failures_and_still_cleans_up(monkeypatch):
    handler = make_handler()
    errors = (RuntimeError("send failed"), ValueError("receive failed"))
    for queue, error in zip(
        (handler.feedback_send_futures, handler.feedback_recv_futures), errors
    ):
        future = Future()
        future.set_exception(error)
        queue.append(future)
    destroy = Mock()
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy)
    with pytest.raises(feedback.FeedbackDrainError) as caught:
        handler.close()
    assert caught.value.errors == errors
    assert caught.value.__cause__ is errors[0]
    assert handler.feedback_executor is None and handler.feedback_cpu_group is None
    destroy.assert_called_once()
