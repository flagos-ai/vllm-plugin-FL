# Copyright (c) 2025 BAAI. All rights reserved.
"""Residual constructor/stream branches for ``PyFlagcxCommunicator``.

The ProcessGroup path is exercised with mocked ``torch.distributed`` helpers;
no collective is issued and no native library is loaded.
"""

import ctypes
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import torch

import vllm_fl.distributed.device_communicators.flagcx as fx


class FakeDevice:
    def __init__(self, name):
        if isinstance(name, FakeDevice):
            self.type, self.index = name.type, name.index
            return
        text = str(name)
        if ":" in text:
            self.type, raw = text.split(":", 1)
            self.index = int(raw)
        else:
            self.type, self.index = text, None

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _comm(disabled=False, device=None):
    obj = fx.PyFlagcxCommunicator.__new__(fx.PyFlagcxCommunicator)
    obj.comm = "COMM"
    obj.device = torch.device("cpu") if device is None else device
    obj.rank = 0
    obj.world_size = 2
    obj.disabled = disabled
    obj.available = True
    obj._all_reduce_stream = None
    obj._destroy_lock = threading.Lock()
    obj._destroy_started = False
    obj._destroy_thread = None
    obj.flagcx = SimpleNamespace(
        flagcxAllReduce=Mock(),
        adaptor_stream_copy=Mock(return_value="stream"),
        adaptor_stream_free=Mock(),
    )
    return obj


def test_all_reduce_defaults_to_current_stream(monkeypatch):
    marker = object()
    monkeypatch.setattr(
        fx, "buffer_type", lambda ptr=0: ("buf", ptr), raising=False
    )
    monkeypatch.setattr(
        fx, "flagcxDataTypeEnum", SimpleNamespace(from_torch=lambda d: d)
    )
    monkeypatch.setattr(
        fx, "flagcxRedOpTypeEnum", SimpleNamespace(from_torch=lambda o: o)
    )
    comm = _comm()
    monkeypatch.setattr(comm, "_current_stream", lambda: marker)

    out = comm.all_reduce(torch.ones(2))

    assert out.shape == (2,)
    comm.flagcx.adaptor_stream_copy.assert_called_once_with(marker)
    comm.flagcx.adaptor_stream_free.assert_called_once_with("stream")


def _install_group(monkeypatch, rank):
    group = object()
    uid = SimpleNamespace(internal=[0, 0, 0])
    ranks = [5, 6]
    broadcast = Mock()
    monkeypatch.setattr(fx, "StatelessProcessGroup", type("Base", (), {}))
    monkeypatch.setattr(fx.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(fx.dist, "get_backend", lambda _g: "gloo")
    monkeypatch.setattr(fx.dist, "get_rank", lambda _g: rank)
    monkeypatch.setattr(fx.dist, "get_world_size", lambda _g: 2)
    monkeypatch.setattr(fx.dist, "get_process_group_ranks", lambda _g: ranks)
    monkeypatch.setattr(fx.dist, "broadcast", broadcast)

    class Library:
        def __init__(self, path):
            self.path = path
            self.get_unique_calls = 0
            self.init_args = None

        def flagcxGetUniqueId(self):
            self.get_unique_calls += 1
            return uid

        def flagcxCommInitRank(self, world, unique, rank_):
            self.init_args = (world, unique, rank_)
            return "NATIVE"

    monkeypatch.setattr(fx, "FLAGCXLibrary", Library)
    monkeypatch.setattr(fx, "flagcxUniqueId", lambda: SimpleNamespace(internal=[0, 0, 0]))
    monkeypatch.setattr(fx, "_flagcx_wheel_library", None)
    monkeypatch.setattr(
        fx.PyFlagcxCommunicator,
        "_current_stream",
        lambda self: SimpleNamespace(synchronize=Mock()),
    )
    monkeypatch.setattr(fx.PyFlagcxCommunicator, "all_reduce", lambda self, data: None)
    monkeypatch.setattr(fx.torch, "zeros", lambda *a, **k: object())
    monkeypatch.setattr(fx.torch, "device", FakeDevice)
    return group, uid, broadcast, ranks


def test_ctor_broadcasts_unique_id_through_process_group(monkeypatch):
    group, uid, broadcast, ranks = _install_group(monkeypatch, rank=0)

    comm = fx.PyFlagcxCommunicator(group, FakeDevice("ptpu"), library_path="/fake/libflagcx.so")

    assert comm.rank == 0 and comm.world_size == 2
    assert comm.comm == "NATIVE"
    assert comm.device.type == "ptpu"
    assert broadcast.call_count == 1
    args, kwargs = broadcast.call_args
    assert kwargs == {"src": ranks[0], "group": group}
    tensor = args[0]
    assert tensor.tolist() == [0, 0, 0]
    assert comm.flagcx.get_unique_calls == 1
    assert comm.flagcx.init_args == (2, uid, 0)


def test_ctor_non_root_skips_unique_id_query(monkeypatch):
    group, uid, broadcast, ranks = _install_group(monkeypatch, rank=1)

    comm = fx.PyFlagcxCommunicator(group, FakeDevice("ptpu"), library_path="/fake/libflagcx.so")

    assert comm.flagcx.get_unique_calls == 0
    assert comm.unique_id.internal == [0, 0, 0]
    assert broadcast.call_count == 1
    assert broadcast.call_args.kwargs["src"] == ranks[0]
    assert comm.flagcx.init_args == (2, comm.unique_id, 1)


def test_ctor_legacy_api_uses_pointer_and_contents(monkeypatch):
    group, _, _, _ = _install_group(monkeypatch, rank=0)

    class Uid(ctypes.Structure):
        _fields_ = [("internal", ctypes.c_byte * 3)]

    uid = Uid()

    class LegacyLibrary:
        def __init__(self, path):
            self.handler = object()
            self.last = None

        def flagcxGetUniqueId(self):
            return SimpleNamespace(contents=uid)

        def flagcxCommInitRank(self, world, pointer, rank_):
            self.last = (world, pointer, rank_)
            return "NATIVE"

    monkeypatch.setattr(fx, "FLAGCXLibrary", LegacyLibrary)

    comm = fx.PyFlagcxCommunicator(group, FakeDevice("ptpu"), library_path="/fake/libflagcx.so")

    assert comm._legacy_unique_id_api is True
    assert comm.unique_id is uid
    pointer = comm.flagcx.last[1]
    assert isinstance(pointer, ctypes.POINTER(Uid))
    assert ctypes.addressof(pointer.contents) == ctypes.addressof(uid)

