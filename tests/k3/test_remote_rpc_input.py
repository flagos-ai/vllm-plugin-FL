# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The K3 hook is local to one input queue on a headless MP worker."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from vllm.distributed.device_communicators import shm_broadcast

from vllm_fl.distributed import kimi_k3_rpc as rpc


class Coordinator:
    def __init__(self, result):
        self.result = result

    def create_mq_broadcaster(
        self, writer_rank=0, external_writer_handle=None, blocking=True
    ):
        return self.result


def config(**other):
    return SimpleNamespace(
        **(
            dict(
                distributed_executor_backend="mp",
                nnodes_within_dp=4,
                pipeline_parallel_size=4,
                node_rank_within_dp=1,
            )
            | other
        )
    )


@pytest.mark.parametrize(
    "other",
    [
        dict(node_rank_within_dp=0),
        dict(nnodes_within_dp=1),
        dict(pipeline_parallel_size=1),
        dict(distributed_executor_backend="ray"),
    ],
)
def test_unaffected_workers_do_not_access_the_coordinator(monkeypatch, other):
    def unexpected():
        pytest.fail("looked up a coordinator outside the K3 remote MP path")

    monkeypatch.setattr(rpc, "get_inner_dp_world_group", unexpected)
    assert rpc.arm_remote_rpc_input(config(**other)) is None


def test_factory_is_instance_local_and_restored(monkeypatch):
    reader = SimpleNamespace(remote_socket=object())
    group = Coordinator(reader)
    other = Coordinator(reader)
    native = Coordinator.create_mq_broadcaster
    monkeypatch.setattr(rpc, "get_inner_dp_world_group", lambda: group)
    hook = rpc.arm_remote_rpc_input(config())
    assert rpc.arm_remote_rpc_input(config()) is hook
    assert other.create_mq_broadcaster.__func__ is native
    assert group.create_mq_broadcaster(blocking=True) is reader
    assert (
        group.create_mq_broadcaster(external_writer_handle=object(), blocking=False)
        is reader
    )
    adapted = group.create_mq_broadcaster(blocking=False)
    assert adapted.reader is reader
    assert group.create_mq_broadcaster.__func__ is native
    assert hook(blocking=False) is reader
    hook.restore()
    assert Coordinator.create_mq_broadcaster is native


def test_factory_failure_and_unconsumed_hook(monkeypatch):
    group = Coordinator(None)

    def fail(writer_rank=0, external_writer_handle=None, blocking=True):
        raise ValueError("factory failed")

    group.create_mq_broadcaster = fail
    monkeypatch.setattr(rpc, "get_inner_dp_world_group", lambda: group)
    rpc.arm_remote_rpc_input(config())
    with pytest.raises(ValueError, match="factory failed"):
        group.create_mq_broadcaster(blocking=False)
    assert group.create_mq_broadcaster is fail
    unused = rpc.arm_remote_rpc_input(config())
    unused.restore()
    assert group.create_mq_broadcaster is fail


def raise_decode_timeout():
    raise TimeoutError("payload decode timeout")


class DecodeFailure:
    def __reduce__(self):
        return raise_decode_timeout, ()


def test_native_delivery_and_shutdown(monkeypatch):
    make_context = shm_broadcast.Context
    contexts = []

    def context(*args, **kwargs):
        result = make_context(*args, **kwargs)
        contexts.append(result)
        return result

    monkeypatch.setattr(shm_broadcast, "Context", context)
    writer = shm_broadcast.MessageQueue(
        1, 0, max_chunks=2, max_chunk_bytes=4096, connect_ip="127.0.0.1"
    )
    raw = shm_broadcast.MessageQueue.create_from_handle(writer.export_handle(), 0)
    group = Coordinator(raw)
    monkeypatch.setattr(rpc, "get_inner_dp_world_group", lambda: group)
    rpc.arm_remote_rpc_input(config())
    reader = group.create_mq_broadcaster(blocking=False)
    writer.wait_until_ready()
    reader.wait_until_ready()
    try:
        writer.enqueue(("first", 1))
        assert reader.dequeue(timeout=0.2) == ("first", 1)
        with pytest.raises(TimeoutError):
            reader.dequeue(timeout=0)
        writer.enqueue(DecodeFailure())
        writer.enqueue(("second", 2))
        with pytest.raises(TimeoutError, match="payload decode timeout"):
            reader.dequeue(indefinite=True)
        assert reader.dequeue(indefinite=True) == ("second", 2)
        entered = threading.Event()

        def wait():
            entered.set()
            return reader.dequeue(indefinite=True)

        with ThreadPoolExecutor(max_workers=1) as pool:
            blocked = pool.submit(wait)
            assert entered.wait(1)
            try:
                with pytest.raises(TimeoutError):
                    blocked.result(timeout=0.15)
                reader.shutdown()
                with pytest.raises(RuntimeError, match="cancelled"):
                    blocked.result(timeout=1)
            finally:
                if not blocked.done():
                    writer.enqueue(("fixture end", 3))
    finally:
        reader.shutdown()
        writer.shutdown()
        for ctx in contexts:
            ctx.destroy(linger=0)


def test_late_payload_is_not_returned_during_shutdown():
    reader = SimpleNamespace(
        shutting_down=False, remote_socket=SimpleNamespace(poll=lambda _: True)
    )

    def receive(timeout, indefinite):
        reader.shutting_down = True
        return "late"

    reader.dequeue = receive
    with pytest.raises(RuntimeError, match="cancelled"):
        rpc.K3RemoteRpcInput(reader).dequeue(indefinite=True)


def test_runner_arms_only_after_successful_k3_load(monkeypatch):
    from vllm_fl.worker import kimi_k3 as module

    calls = []
    cfg = config()
    runner = module.K3ModelRunner.__new__(module.K3ModelRunner)
    runner.vllm_config = SimpleNamespace(parallel_config=cfg)
    monkeypatch.setattr(module, "arm_remote_rpc_input", lambda v: calls.append(v))
    monkeypatch.setattr(
        module.GPUModelRunner, "load_model", lambda self, *a, **k: "loaded"
    )
    runner.k3_execution = SimpleNamespace(enabled=False)
    assert runner.load_model() == "loaded"
    assert calls == []
    runner.k3_execution.enabled = True
    assert runner.load_model() == "loaded"
    assert calls == [cfg]

    def failed(*args, **kwargs):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(module.GPUModelRunner, "load_model", failed)
    with pytest.raises(RuntimeError, match="model load failed"):
        runner.load_model()
    assert calls == [cfg]
