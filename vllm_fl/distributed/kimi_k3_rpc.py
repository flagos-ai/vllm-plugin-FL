# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancellation for K3's remote scheduler input on the pinned multiprocess executor."""

from inspect import signature

from vllm.distributed.parallel_state import get_inner_dp_world_group


class K3RemoteRpcInput:
    """Keep the native packet decoder and make the worker's idle wait interruptible."""

    def __init__(self, reader):
        self.reader = reader

    def wait_until_ready(self):
        return self.reader.wait_until_ready()

    def shutdown(self):
        return self.reader.shutdown()

    def dequeue(self, timeout=None, indefinite=False):
        reader = self.reader
        if timeout is not None or not indefinite:
            return reader.dequeue(timeout=timeout, indefinite=indefinite)

        # The worker busy loop requests this indefinite wait. A ready socket
        # returns immediately. Never catch a TimeoutError from payload decoding.
        while not reader.shutting_down:
            if reader.remote_socket.poll(100):
                if not reader.shutting_down:
                    value = reader.dequeue(timeout=0, indefinite=True)
                    if not reader.shutting_down:
                        return value
                break
        raise RuntimeError("cancelled")


class _RemoteInputFactory:
    """One use of one coordinator instance, between model load and queue creation."""

    def __init__(self, group):
        self.group = group
        self.original = group.create_mq_broadcaster
        self.previous = vars(group).get("create_mq_broadcaster")

    def restore(self):
        group = self.group
        if group is not None and vars(group).get("create_mq_broadcaster") is self:
            if self.previous is None:
                del group.create_mq_broadcaster
            else:
                group.create_mq_broadcaster = self.previous
        self.group = self.previous = None

    def __call__(self, writer_rank=0, external_writer_handle=None, blocking=True):
        selected = (
            self.group is not None
            and writer_rank == 0
            and external_writer_handle is None
            and not blocking
        )
        if selected:
            self.restore()
        reader = self.original(
            writer_rank=writer_rank,
            external_writer_handle=external_writer_handle,
            blocking=blocking,
        )
        if selected and reader.remote_socket is not None:
            return K3RemoteRpcInput(reader)
        return reader


def arm_remote_rpc_input(parallel_config):
    if (
        parallel_config.distributed_executor_backend != "mp"
        or parallel_config.pipeline_parallel_size != 4
        or parallel_config.nnodes_within_dp < 2
        or parallel_config.node_rank_within_dp == 0
    ):
        return None
    group = get_inner_dp_world_group()
    factory = group.create_mq_broadcaster
    if isinstance(factory, _RemoteInputFactory):
        return factory
    if tuple(signature(factory).parameters) != (
        "writer_rank",
        "external_writer_handle",
        "blocking",
    ):
        raise RuntimeError("Unsupported vLLM scheduler queue factory for K3")
    hook = _RemoteInputFactory(group)
    group.create_mq_broadcaster = hook
    return hook
