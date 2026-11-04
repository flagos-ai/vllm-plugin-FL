# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU behavior tests for opt-in TXDA collective phase synchronization."""

import ast
import ctypes
import gc
import logging
import os
import threading
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


class Device:
    def __init__(self, name):
        self.type, _, index = name.partition(":")
        self.index = int(index or 0)


class Tensor:
    def __init__(self, device, count=4):
        self.device = device
        self.dtype = "float32"
        self.shape = (count,)

    def numel(self):
        return self.shape[0]

    def data_ptr(self):
        return id(self)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index = index[0]
        return Tensor(self.device, index.stop - index.start)


class Stream:
    def __init__(self, events, name="current"):
        self.events = events
        self.name = name
        self.npu_stream = 0
        self.on_sync = lambda: None

    def synchronize(self):
        self.events.append(("sync", self.name))
        self.on_sync()


class Group:
    rank = 0
    world_size = 2

    def __init__(self, events):
        self.events = events

    def broadcast_obj(self, value, src):
        return value

    def barrier(self):
        self.events.append(("barrier", "stateless"))


class Library:
    def __init__(self, events):
        self.events = events
        self.devHandle = object()

    def flagcxGetUniqueId(self):
        return object()

    def flagcxCommInitRank(self, world_size, unique_id, rank):
        self.events.append(("init",))
        return object()

    def adaptor_stream_copy(self, stream):
        name = getattr(stream, "name", "current")
        self.events.append(("copy", name))
        return name

    def adaptor_stream_free(self, stream):
        self.events.append(("free", stream))

    def flagcxGroupStart(self, comm):
        self.events.append(("group_start",))

    def flagcxGroupEnd(self, comm):
        self.events.append(("group_end",))

    def __getattr__(self, name):
        if name.startswith("flagcx"):

            def enqueue(*args):
                self.events.append(("enqueue", name))

            return enqueue
        raise AttributeError(name)


@pytest.fixture
def loaded(monkeypatch):
    monkeypatch.delenv("VLLM_TXDA_SYNC_COLLECTIVES", raising=False)
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm_fl/distributed/device_communicators/flagcx.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PyFlagcxCommunicator"
    )
    events = []
    stream = Stream(events)
    library = Library(events)
    allocations = []

    def empty_like(tensor):
        result = Tensor(tensor.device, tensor.numel())
        allocations.append(weakref.ref(result))
        return result

    runtime = SimpleNamespace(
        device=Device,
        txda=SimpleNamespace(
            device=lambda device: nullcontext(), current_stream=lambda device: stream
        ),
        cuda=SimpleNamespace(device=lambda device: nullcontext()),
        npu=SimpleNamespace(
            device=lambda device: nullcontext(), current_stream=lambda device: stream
        ),
        zeros=lambda count, device: Tensor(device, count),
        empty_like=empty_like,
    )
    process_group = object()

    def barrier(*, group):
        assert group is process_group
        events.append(("barrier", "process_group"))

    namespace = {
        "ctypes": ctypes,
        "logging": logging,
        "os": os,
        "threading": threading,
        "torch": runtime,
        "dist": SimpleNamespace(barrier=barrier),
        "StatelessProcessGroup": Group,
        "SimpleNamespace": SimpleNamespace,
        "ReduceOp": SimpleNamespace(SUM="sum"),
        "FLAGCXLibrary": lambda path: library,
        "flagcxDataTypeEnum": SimpleNamespace(from_torch=lambda dtype: dtype),
        "flagcxRedOpTypeEnum": SimpleNamespace(from_torch=lambda op: op),
        "buffer_type": ctypes.c_void_p,
        "logger": logging.getLogger("test.txda.flagcx.sync"),
        "current_stream": lambda: stream,
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return SimpleNamespace(
        cls=namespace["PyFlagcxCommunicator"],
        events=events,
        stream=stream,
        library=library,
        allocations=allocations,
        process_group=process_group,
        dist=namespace["dist"],
    )


def ready_comm(loaded, *, sync=True, stateless=False):
    comm = loaded.cls.__new__(loaded.cls)
    comm.device = Device("txda:0")
    comm.rank = 0
    comm.comm = object()
    comm.group = Group(loaded.events) if stateless else loaded.process_group
    comm.flagcx = loaded.library
    comm.disabled = False
    comm._legacy_unique_id_api = False
    comm._sync_collectives = sync
    comm._group_depth = 0
    comm._suspended_group_depth = 0
    comm._collective_phase_active = False
    return comm


@pytest.mark.parametrize(
    "operation",
    [
        "all_reduce",
        "all_gather",
        "broadcast",
        "reduce_scatter",
        "all_gatherv",
        "reduce_scatterv",
    ],
)
def test_full_group_collectives_close_gpu_phase_before_free(loaded, operation):
    comm = ready_comm(loaded)
    input_tensor, output_tensor = Tensor(comm.device), Tensor(comm.device, 8)
    if operation == "all_reduce":
        assert comm.all_reduce(input_tensor, output_tensor) is output_tensor
    elif operation == "broadcast":
        comm.broadcast(input_tensor, src=0)
    elif operation in {"all_gatherv", "reduce_scatterv"}:
        getattr(comm, operation)(output_tensor, input_tensor, [4, 4])
    else:
        getattr(comm, operation)(output_tensor, input_tensor)

    assert loaded.events[-4:] == [
        ("barrier", "process_group"),
        ("sync", "current"),
        ("barrier", "process_group"),
        ("free", "current"),
    ]
    phase_events = [
        event[0]
        for event in loaded.events
        if event[0] in {"sync", "barrier", "enqueue"}
    ]
    num_enqueues = 2 if operation in {"all_gatherv", "reduce_scatterv"} else 1
    assert (
        phase_events
        == ["sync", "barrier", "enqueue", "barrier", "sync", "barrier"] * num_enqueues
    )
    assert not any(event[0] in {"group_start", "group_end"} for event in loaded.events)


def test_allocated_all_reduce_output_lives_through_all_barriers_and_sync(loaded):
    comm = ready_comm(loaded)

    def assert_alive():
        gc.collect()
        assert loaded.allocations[-1]() is not None

    loaded.stream.on_sync = assert_alive

    def barrier(*, group):
        gc.collect()
        assert loaded.allocations[-1]() is not None
        loaded.events.append(("barrier", "process_group"))

    # All barriers inspect only a weak reference to the allocated output.
    loaded.dist.barrier = barrier
    result = comm.all_reduce(Tensor(comm.device))
    assert loaded.allocations[-1]() is result
    del result
    gc.collect()
    assert loaded.allocations[-1]() is None


@pytest.mark.parametrize("device", ["txda:0", "cuda:0", "npu:0"])
@pytest.mark.parametrize("env", [None, "0", "1"])
def test_constructor_enables_sync_only_for_explicit_txda_opt_in(
    loaded, monkeypatch, device, env
):
    if env is not None:
        monkeypatch.setenv("VLLM_TXDA_SYNC_COLLECTIVES", env)
    comm = loaded.cls(Group(loaded.events), Device(device), library_path="test")
    enabled = device.startswith("txda") and env == "1"
    assert comm._sync_collectives is enabled
    barriers = [event for event in loaded.events if event[0] == "barrier"]
    assert barriers == [("barrier", "stateless")] * (3 if enabled else 0)
    # Opt-in warmup finishes both producers and the collective; default keeps
    # its existing post-enqueue stream synchronization.
    assert sum(event[0] == "sync" for event in loaded.events) == (2 if enabled else 1)


def test_default_collective_only_enqueues_and_frees(loaded):
    comm = ready_comm(loaded, sync=False)
    comm.all_reduce(Tensor(comm.device))
    assert loaded.events == [
        ("copy", "current"),
        ("enqueue", "flagcxAllReduce"),
        ("free", "current"),
    ]


def test_stateless_control_group_uses_its_own_barrier(loaded):
    comm = ready_comm(loaded, stateless=True)
    comm.all_reduce(Tensor(comm.device))
    assert loaded.events[:2] == [("sync", "current"), ("barrier", "stateless")]
    assert loaded.events[-4:] == [
        ("barrier", "stateless"),
        ("sync", "current"),
        ("barrier", "stateless"),
        ("free", "current"),
    ]


def test_nested_group_serializes_each_full_collective_then_restores_native_groups(
    loaded,
):
    comm = ready_comm(loaded)
    comm.group_start()
    comm.group_start()
    input_tensor = Tensor(comm.device)
    first = comm.all_reduce(input_tensor)
    assert comm._group_depth == 2 and comm._suspended_group_depth == 0
    second = comm.all_reduce(input_tensor)
    assert first is not second
    assert comm._group_depth == 2 and comm._suspended_group_depth == 0

    comm.group_end()
    comm.group_end()

    one_phase = [
        ("group_end",),
        ("group_end",),
        ("sync", "current"),
        ("barrier", "process_group"),
        ("copy", "current"),
        ("enqueue", "flagcxAllReduce"),
        ("barrier", "process_group"),
        ("sync", "current"),
        ("barrier", "process_group"),
        ("free", "current"),
        ("group_start",),
        ("group_start",),
    ]
    assert loaded.events == [
        ("group_start",),
        ("group_start",),
    ] + one_phase * 2 + [("group_end",), ("group_end",)]
    assert comm._group_depth == 0


@pytest.mark.parametrize("operation", ["all_gatherv", "reduce_scatterv"])
def test_variable_collective_runs_each_root_outside_existing_outer_group(
    loaded, operation
):
    comm = ready_comm(loaded)
    comm.group_start()
    getattr(comm, operation)(Tensor(comm.device, 8), Tensor(comm.device), [4, 4])
    assert comm._group_depth == 1 and comm._suspended_group_depth == 0
    assert sum(event[0] == "barrier" for event in loaded.events) == 6
    # Every per-root operation temporarily closes and restores the native group.
    native_depth = 0
    for event in loaded.events:
        if event[0] == "group_start":
            native_depth += 1
        elif event[0] == "group_end":
            native_depth -= 1
        elif event[0] == "enqueue":
            assert native_depth == 0
    comm.group_end()
    assert loaded.events[-2:] == [("group_start",), ("group_end",)]
    assert comm._group_depth == 0


@pytest.mark.parametrize("grouped", [False, True])
def test_point_to_point_never_adds_a_full_group_barrier(loaded, grouped):
    comm = ready_comm(loaded)
    if grouped:
        comm.group_start()
    comm.send(Tensor(comm.device), dst=1)
    comm.recv(Tensor(comm.device), src=1)
    if grouped:
        comm.group_end()
    assert not any(event[0] in {"barrier", "sync"} for event in loaded.events)
    assert sum(event[0] == "group_start" for event in loaded.events) == int(grouped)
    assert sum(event[0] == "group_end" for event in loaded.events) == int(grouped)


@pytest.mark.parametrize("operation", ["all_gatherv", "reduce_scatterv"])
def test_default_variable_collectives_keep_the_native_group(loaded, operation):
    comm = ready_comm(loaded, sync=False)
    getattr(comm, operation)(Tensor(comm.device, 8), Tensor(comm.device), [4, 4])
    assert [event[0] for event in loaded.events] == [
        "copy",
        "group_start",
        "enqueue",
        "enqueue",
        "group_end",
        "free",
    ]


def test_mixed_group_submits_p2p_before_full_collective_then_reopens_for_p2p(loaded):
    comm = ready_comm(loaded)
    tensor = Tensor(comm.device)
    comm.group_start()
    comm.send(tensor, dst=1)
    comm.recv(tensor, src=1)
    comm.all_reduce(tensor)
    comm.send(tensor, dst=1)
    comm.recv(tensor, src=1)
    comm.group_end()

    assert loaded.events == [
        ("group_start",),
        ("copy", "current"),
        ("enqueue", "flagcxSend"),
        ("free", "current"),
        ("copy", "current"),
        ("enqueue", "flagcxRecv"),
        ("free", "current"),
        ("group_end",),
        ("sync", "current"),
        ("barrier", "process_group"),
        ("copy", "current"),
        ("enqueue", "flagcxAllReduce"),
        ("barrier", "process_group"),
        ("sync", "current"),
        ("barrier", "process_group"),
        ("free", "current"),
        ("group_start",),
        ("copy", "current"),
        ("enqueue", "flagcxSend"),
        ("free", "current"),
        ("copy", "current"),
        ("enqueue", "flagcxRecv"),
        ("free", "current"),
        ("group_end",),
    ]


@pytest.mark.parametrize("failure_phase", ["pre_sync", "enqueue", "post_sync"])
@pytest.mark.parametrize("group_depth", [0, 2])
def test_failed_grouped_collective_does_not_reopen_native_groups(
    loaded, failure_phase, group_depth
):
    comm = ready_comm(loaded)
    for _ in range(group_depth):
        comm.group_start()
    sync_count = 0

    def synchronize():
        nonlocal sync_count
        sync_count += 1
        if sync_count == (1 if failure_phase == "pre_sync" else 2):
            raise RuntimeError("phase failed")

    if failure_phase == "enqueue":

        def enqueue(*args):
            raise RuntimeError("phase failed")

        loaded.library.flagcxAllReduce = enqueue
    else:
        loaded.stream.on_sync = synchronize

    with pytest.raises(RuntimeError, match="phase failed"):
        comm.all_reduce(Tensor(comm.device))

    assert comm._group_depth == 0 and comm._suspended_group_depth == group_depth
    assert comm._collective_phase_active
    assert sum(event[0] == "group_start" for event in loaded.events) == group_depth
    assert sum(event[0] == "group_end" for event in loaded.events) == group_depth
    event_count = len(loaded.events)
    for action in (
        comm.group_start,
        comm.group_end,
        lambda: comm.all_reduce(Tensor(comm.device)),
        lambda: comm.send(Tensor(comm.device), dst=1),
        lambda: comm.recv(Tensor(comm.device), src=1),
    ):
        with pytest.raises(RuntimeError, match="abort the communicator"):
            action()
    assert len(loaded.events) == event_count


def test_consecutive_collectives_reenter_the_producer_sync_phase(loaded):
    comm = ready_comm(loaded)
    tensor = Tensor(comm.device)
    comm.all_reduce(tensor)
    comm.all_gather(Tensor(comm.device, 8), tensor)

    assert loaded.events == [
        ("sync", "current"),
        ("barrier", "process_group"),
        ("copy", "current"),
        ("enqueue", "flagcxAllReduce"),
        ("barrier", "process_group"),
        ("sync", "current"),
        ("barrier", "process_group"),
        ("free", "current"),
        ("sync", "current"),
        ("barrier", "process_group"),
        ("copy", "current"),
        ("enqueue", "flagcxAllGather"),
        ("barrier", "process_group"),
        ("sync", "current"),
        ("barrier", "process_group"),
        ("free", "current"),
    ]


def test_explicit_native_stream_is_used_for_both_sync_phases(loaded):
    comm = ready_comm(loaded)
    stream = Stream(loaded.events, "explicit")
    comm.all_reduce(Tensor(comm.device), stream=stream)
    assert [event for event in loaded.events if event[0] == "sync"] == [
        ("sync", "explicit"),
        ("sync", "explicit"),
    ]


def test_sync_mode_rejects_unmatched_group_end_before_native_call(loaded):
    comm = ready_comm(loaded)
    with pytest.raises(RuntimeError, match="without group_start"):
        comm.group_end()
    assert loaded.events == []
