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

"""CPU tests for the FlagCX communicator import ladder and collectives.

The real FlagCX wheel is absent from the NVIDIA CI image, so the sanctioned
external boundary is a fake ``flagcx``/``flagcx.api`` package installed into
``sys.modules`` (the same style as ``kv_transfer/_flagcx_env.py``).  The
import ladder, source-tree library-path resolution, disabled-by-default
behavior, and every collective wrapper are asserted against argument
recorders; no collective is executed.  Constructor success paths need a live
device stream and are marked ``gpu``.
"""

from __future__ import annotations

import ctypes
import importlib
import sys
import types
from contextlib import nullcontext

import pytest
import torch

from . import _imports  # noqa: F401  (installs CPU-only import placeholders)

MODULE_NAME = "vllm_fl.distributed.device_communicators.flagcx"
_REQUIRED = (
    "FLAGCXLibrary",
    "buffer_type",
    "flagcxDataTypeEnum",
    "flagcxUniqueId",
    "flagcxRedOpTypeEnum",
)


class _Recorder(list):
    def record(self, name, *args):
        self.append((name, *args))


class _Buffer:
    def __init__(self, ptr=None):
        self.ptr = ptr


class _DataTypeEnum:
    @staticmethod
    def from_torch(dtype):
        return f"dtype:{dtype}"


class _RedOpEnum:
    @staticmethod
    def from_torch(op):
        return f"op:{op}"


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_ubyte * 8)]


class _ContentsUniqueId:
    def __init__(self):
        self.contents = _UniqueId()


def _fake_api(recorder, *, library=None, include_all=True):
    module = types.ModuleType("flagcx.api")

    def factory(path=None):
        recorder.record("FLAGCXLibrary", path)
        return library

    module.FLAGCXLibrary = factory
    module.buffer_type = _Buffer
    module.flagcxDataTypeEnum = _DataTypeEnum
    module.flagcxUniqueId = _UniqueId
    module.flagcxRedOpTypeEnum = _RedOpEnum
    if not include_all:
        for name in _REQUIRED:
            delattr(module, name)
    return module


def _fake_flagcx_package(api):
    package = types.ModuleType("flagcx")
    package.__path__ = []
    if api is not None:
        package.api = api
    return package


def _load(monkeypatch, *, api, wrapper=None):
    """Re-import the communicator module against the supplied fakes."""
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)
    if api is not None:
        monkeypatch.setitem(sys.modules, "flagcx", _fake_flagcx_package(api))
        monkeypatch.setitem(sys.modules, "flagcx.api", api)
    else:
        monkeypatch.delitem(sys.modules, "flagcx", raising=False)
        monkeypatch.delitem(sys.modules, "flagcx.api", raising=False)
    if wrapper is not None:
        plugin = types.ModuleType("plugin")
        plugin.__path__ = []
        interservice = types.ModuleType("plugin.interservice")
        interservice.__path__ = []
        interservice.flagcx_wrapper = wrapper
        monkeypatch.setitem(sys.modules, "plugin", plugin)
        monkeypatch.setitem(sys.modules, "plugin.interservice", interservice)
        monkeypatch.setitem(sys.modules, "plugin.interservice.flagcx_wrapper", wrapper)
    else:
        for name in (
            "plugin",
            "plugin.interservice",
            "plugin.interservice.flagcx_wrapper",
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(MODULE_NAME)
    parent = importlib.import_module("vllm_fl.distributed.device_communicators")
    monkeypatch.setattr(parent, "flagcx", module, raising=False)
    return module


def test_wheel_shape_binds_every_public_name(monkeypatch):
    """An installed flagcx.api wheel enables the communicator."""
    recorder = _Recorder()
    module = _load(monkeypatch, api=_fake_api(recorder, library="LIB"))
    assert module._flagcx_available is True
    assert module._flagcx_module.__name__ == "flagcx.api"
    assert module.FLAGCXLibrary is module._flagcx_module.FLAGCXLibrary
    assert module.buffer_type is _Buffer
    assert module.flagcxUniqueId is _UniqueId
    # The wheel facade resolves its own library: no path is forced.
    assert module._resolve_flagcx_library_path(None) is None
    assert module._resolve_flagcx_library_path("/explicit.so") == "/explicit.so"


def test_partial_flagcx_api_disables_the_communicator(monkeypatch):
    """A pre-API flagcx package leaves every bound name unset."""
    recorder = _Recorder()
    module = _load(
        monkeypatch,
        api=_fake_api(recorder, library="LIB", include_all=False),
    )
    assert module._flagcx_available is False
    for name in _REQUIRED:
        assert getattr(module, name) is None


def test_absent_flagcx_disables_the_communicator(monkeypatch):
    """With neither wheel nor source wrapper the backend is unavailable."""
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    module = _load(monkeypatch, api=None)
    assert module._flagcx_available is False
    assert module.FLAGCXLibrary is None


def test_source_wrapper_import_and_library_path_ladder(monkeypatch, tmp_path):
    """FLAGCX_PATH resolves lib/ then build/lib/ for the source wrapper."""
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    recorder = _Recorder()
    wrapper = types.ModuleType("plugin.interservice.flagcx_wrapper")
    wrapper.FLAGCXLibrary = lambda path=None: recorder.record("LIB", path) or "LIB"
    wrapper.buffer_type = _Buffer
    wrapper.flagcxDataTypeEnum = _DataTypeEnum
    wrapper.flagcxUniqueId = _UniqueId
    wrapper.flagcxRedOpTypeEnum = _RedOpEnum
    module = _load(monkeypatch, api=None, wrapper=wrapper)
    assert module._flagcx_available is True
    assert module._flagcx_module is wrapper

    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    assert module._resolve_flagcx_library_path(None) is None

    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libflagcx.so").write_bytes(b"")
    monkeypatch.setenv("FLAGCX_PATH", str(tmp_path))
    assert module._resolve_flagcx_library_path(None) == str(
        tmp_path / "lib" / "libflagcx.so"
    )

    (tmp_path / "lib" / "libflagcx.so").unlink()
    (tmp_path / "build" / "lib").mkdir(parents=True)
    (tmp_path / "build" / "lib" / "libflagcx.so").write_bytes(b"")
    assert module._resolve_flagcx_library_path(None) == str(
        tmp_path / "build" / "lib" / "libflagcx.so"
    )


class _FakeLibrary:
    """Recording stand-in for ``FLAGCXLibrary``."""

    def __init__(self, recorder, *, legacy=False, unique_id=None):
        self.recorder = recorder
        self.unique_id = unique_id or _UniqueId()
        self.comm = "COMM"
        self.legacy_contents = None
        if legacy:
            self.handler = object()
            self.legacy_contents = _ContentsUniqueId()
        else:
            self.devHandle = object()

    def flagcxGetUniqueId(self):
        self.recorder.record("flagcxGetUniqueId")
        if self.legacy_contents is not None:
            return self.legacy_contents
        return self.unique_id

    def adaptor_stream_copy(self, stream):
        self.recorder.record("adaptor_stream_copy", stream)
        return ("flagcx-stream", id(stream))

    def adaptor_stream_free(self, stream):
        self.recorder.record("adaptor_stream_free", stream)

    def flagcxCommInitRank(self, world, unique_id, rank):
        self.recorder.record("flagcxCommInitRank", world, unique_id, rank)
        return self.comm

    def flagcxAllReduce(self, *args):
        self.recorder.record("flagcxAllReduce", *args)

    def flagcxAllGather(self, *args):
        self.recorder.record("flagcxAllGather", *args)

    def flagcxReduceScatter(self, *args):
        self.recorder.record("flagcxReduceScatter", *args)

    def flagcxReduce(self, *args):
        self.recorder.record("flagcxReduce", *args)

    def flagcxBroadcast(self, *args):
        self.recorder.record("flagcxBroadcast", *args)

    def flagcxSend(self, *args):
        self.recorder.record("flagcxSend", *args)

    def flagcxRecv(self, *args):
        self.recorder.record("flagcxRecv", *args)

    def flagcxGroupStart(self):
        self.recorder.record("flagcxGroupStart")

    def flagcxGroupEnd(self):
        self.recorder.record("flagcxGroupEnd")


def _wrapper_module(monkeypatch, library):
    recorder = library.recorder
    module = _load(
        monkeypatch,
        api=_fake_api(recorder, library=library),
    )
    monkeypatch.setattr(module, "current_stream", lambda: "current-stream")
    parent = importlib.import_module("vllm_fl.distributed.device_communicators")
    monkeypatch.setattr(parent, "flagcx", module, raising=False)
    comm = object.__new__(module.PyFlagcxCommunicator)
    comm.rank = 0
    comm.world_size = 2
    comm.device = torch.device("cpu")
    comm.comm = "COMM"
    comm.available = True
    comm.disabled = False
    comm.flagcx = library
    return module, comm


def test_all_reduce_defaults_and_records_every_argument(monkeypatch):
    """all_reduce allocates the output, forwards dtype/op, frees the stream."""
    recorder = _Recorder()
    library = _FakeLibrary(recorder)
    module, comm = _wrapper_module(monkeypatch, library)
    tensor = torch.arange(4, dtype=torch.float32)
    out = module.PyFlagcxCommunicator.all_reduce(
        comm, tensor, op=torch.distributed.ReduceOp.MAX
    )
    assert out.shape == tensor.shape
    assert out is not tensor
    copied = [call for call in recorder if call[0] == "adaptor_stream_copy"]
    assert copied == [("adaptor_stream_copy", "current-stream")]
    reduce_call = [call for call in recorder if call[0] == "flagcxAllReduce"][0]
    assert reduce_call[1].ptr == tensor.data_ptr()
    assert reduce_call[2].ptr == out.data_ptr()
    assert reduce_call[3] == 4
    assert reduce_call[4] == "dtype:torch.float32"
    assert reduce_call[5] == "op:" + str(torch.distributed.ReduceOp.MAX)
    assert reduce_call[6] == "COMM"
    assert reduce_call[7][0] == "flagcx-stream"
    assert copied[0][1] == "current-stream"
    assert recorder[-1][0] == "adaptor_stream_free"


def test_all_reduce_returns_explicit_output_and_accepts_stream(monkeypatch):
    """An explicit out tensor and stream are used verbatim."""
    recorder = _Recorder()
    library = _FakeLibrary(recorder)
    module, comm = _wrapper_module(monkeypatch, library)
    tensor = torch.ones(2)
    out = torch.zeros(2)
    result = module.PyFlagcxCommunicator.all_reduce(
        comm, tensor, out_tensor=out, stream="given-stream"
    )
    assert result is out
    assert ("adaptor_stream_copy", "given-stream") in recorder


def test_all_reduce_rejects_a_tensor_on_another_device(monkeypatch):
    """A tensor outside the communicator's device asserts before dispatch."""
    recorder = _Recorder()
    library = _FakeLibrary(recorder)
    module, comm = _wrapper_module(monkeypatch, library)
    comm.device = torch.device("cuda:0")
    with pytest.raises(AssertionError, match="created to work on"):
        module.PyFlagcxCommunicator.all_reduce(comm, torch.zeros(2))


@pytest.mark.parametrize(
    "method",
    [
        "all_reduce",
        "all_gather",
        "all_gatherv",
        "reduce_scatter",
        "reduce_scatterv",
        "send",
        "recv",
        "broadcast",
    ],
)
def test_disabled_communicator_returns_without_dispatch(monkeypatch, method):
    """Every collective is a no-op while the communicator is disabled."""
    recorder = _Recorder()
    library = _FakeLibrary(recorder)
    module, comm = _wrapper_module(monkeypatch, library)
    comm.disabled = True
    tensor = torch.zeros(2)
    args = {
        "all_reduce": (tensor,),
        "all_gather": (tensor, tensor),
        "all_gatherv": (tensor, tensor, [2]),
        "reduce_scatter": (tensor, tensor),
        "reduce_scatterv": (tensor, tensor, [2]),
        "send": (tensor, 1),
        "recv": (tensor, 0),
        "broadcast": (tensor, 0),
    }[method]
    assert getattr(module.PyFlagcxCommunicator, method)(comm, *args) is None
    assert recorder == []


def test_all_gather_records_pointer_arguments(monkeypatch):
    """all_gather forwards input/output buffers and the element count."""
    recorder = _Recorder()
    library = _FakeLibrary(recorder)
    module, comm = _wrapper_module(monkeypatch, library)
    inp = torch.ones(3)
    out = torch.zeros(6)
    module.PyFlagcxCommunicator.all_gather(comm, out, inp)
    call = [c for c in recorder if c[0] == "flagcxAllGather"][0]
    assert call[1].ptr == inp.data_ptr()
    assert call[2].ptr == out.data_ptr()
    assert call[3] == 3
    assert call[4] == "dtype:torch.float32"
    assert call[5] == "COMM"
    assert call[-1][0] == "flagcx-stream"


def test_all_gatherv_broadcasts_each_split_in_a_group(monkeypatch):
    """all_gatherv sizes the output, groups, and broadcasts each chunk."""
    recorder = _Recorder()
    library = _FakeLibrary(recorder)
    module, comm = _wrapper_module(monkeypatch, library)
    inp = torch.arange(3.0)
    out = torch.zeros(5)
    module.PyFlagcxCommunicator.all_gatherv(comm, out, inp, [2, 3])
    names = [call[0] for call in recorder]
    assert names.count("flagcxBroadcast") == 2
    assert names.index("flagcxGroupStart") < names.index("flagcxBroadcast")
    assert names.index("flagcxGroupEnd") > names.index("flagcxBroadcast")
    broadcasts = [c for c in recorder if c[0] == "flagcxBroadcast"]
    assert broadcasts[0][1].ptr == inp.data_ptr()
    assert broadcasts[0][2].ptr == out[0:2].data_ptr()
    assert broadcasts[0][3] == 2
    assert broadcasts[0][5] == 0
    assert broadcasts[1][2].ptr == out[2:5].data_ptr()
    assert broadcasts[1][3] == 3
    assert broadcasts[1][5] == 1
    with pytest.raises(AssertionError):
        module.PyFlagcxCommunicator.all_gatherv(comm, out, inp, [4, 3])


def test_reduce_scatter_and_reduce_scatterv_chunking(monkeypatch):
    """reduce_scatter passes output width; the v-form reduces each chunk."""
    recorder = _Recorder()
    library = _FakeLibrary(recorder)
    module, comm = _wrapper_module(monkeypatch, library)
    out = torch.zeros(2)
    inp = torch.ones(4)
    module.PyFlagcxCommunicator.reduce_scatter(comm, out, inp)
    call = [c for c in recorder if c[0] == "flagcxReduceScatter"][0]
    assert call[1].ptr == inp.data_ptr()
    assert call[2].ptr == out.data_ptr()
    assert call[3] == 2

    recorder.clear()
    big_out = torch.zeros(2)
    big_in = torch.arange(4.0)
    module.PyFlagcxCommunicator.reduce_scatterv(comm, big_out, big_in, [1, 3])
    names = [c[0] for c in recorder]
    assert names.count("flagcxReduce") == 2
    reduces = [c for c in recorder if c[0] == "flagcxReduce"]
    assert reduces[0][1].ptr == big_in[0:1].data_ptr()
    assert reduces[0][3] == 1
    assert reduces[0][6] == 0
    assert reduces[1][1].ptr == big_in[1:4].data_ptr()
    assert reduces[1][3] == 3
    assert reduces[1][6] == 1
    assert reduces[0][2].ptr == big_out.data_ptr()


def test_send_recv_and_broadcast_buffer_selection(monkeypatch):
    """send/recv forward the tensor; broadcast picks sender buffers by rank."""
    recorder = _Recorder()
    library = _FakeLibrary(recorder)
    module, comm = _wrapper_module(monkeypatch, library)
    tensor = torch.ones(2)
    module.PyFlagcxCommunicator.send(comm, tensor, 1)
    send = [c for c in recorder if c[0] == "flagcxSend"][0]
    assert send[1].ptr == tensor.data_ptr()
    assert send[2] == 2
    assert send[4] == 1

    recorder.clear()
    module.PyFlagcxCommunicator.recv(comm, tensor, 1)
    recv = [c for c in recorder if c[0] == "flagcxRecv"][0]
    assert recv[1].ptr == tensor.data_ptr()
    assert recv[4] == 1

    recorder.clear()
    comm.rank = 0
    module.PyFlagcxCommunicator.broadcast(comm, tensor, 0)
    src_call = [c for c in recorder if c[0] == "flagcxBroadcast"][0]
    assert src_call[1].ptr == tensor.data_ptr()
    assert src_call[2].ptr == tensor.data_ptr()

    recorder.clear()
    module.PyFlagcxCommunicator.broadcast(comm, tensor, 1)
    dst_call = [c for c in recorder if c[0] == "flagcxBroadcast"][0]
    assert dst_call[1].ptr is None
    assert dst_call[2].ptr == tensor.data_ptr()


def test_group_start_and_end_delegate(monkeypatch):
    """Group markers forward to the FlagCX library."""
    recorder = _Recorder()
    library = _FakeLibrary(recorder)
    module, comm = _wrapper_module(monkeypatch, library)
    module.PyFlagcxCommunicator.group_start(comm)
    module.PyFlagcxCommunicator.group_end(comm)
    assert recorder == [("flagcxGroupStart",), ("flagcxGroupEnd",)]


class _FakeStream:
    def synchronize(self):
        self.synchronized = True


def _cpu_device_boundary(monkeypatch, module):
    """Fake the CUDA device context and current stream for CPU runs."""
    monkeypatch.setattr(module, "current_stream", lambda: _FakeStream())
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())


class _FakeDist:
    """Process-group surface used before the library is constructed."""

    NCCL = "nccl"
    Backend = types.SimpleNamespace(NCCL="nccl")

    def __init__(self, rank=0, world_size=2, backend="mpi"):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.broadcasts = []

    def is_initialized(self):
        return True

    def get_backend(self, group):
        return self.backend

    def get_rank(self, group):
        return self.rank

    def get_world_size(self, group):
        return self.world_size

    def get_process_group_ranks(self, group):
        return [10, 11, 12, 13]

    def broadcast(self, tensor, src, group):
        self.broadcasts.append((tensor, src, group))
        if self.rank == 0:
            # The source rank already holds the bytes it broadcasts.
            tensor.fill_(0)
        else:
            tensor.copy_(torch.arange(1, tensor.numel() + 1, dtype=tensor.dtype))


def test_single_rank_group_is_disabled_without_constructing(monkeypatch):
    """world_size == 1 marks the communicator disabled before any library call."""
    recorder = _Recorder()
    module = _load(monkeypatch, api=_fake_api(recorder, library="LIB"))
    monkeypatch.setattr(module, "dist", _FakeDist(world_size=1))
    comm = module.PyFlagcxCommunicator(object(), "cpu")
    assert comm.available is False
    assert comm.disabled is True
    assert recorder == []


def test_library_construction_failure_disables_the_communicator(monkeypatch):
    """A raising FLAGCXLibrary leaves the communicator disabled, not failed."""
    recorder = _Recorder()
    module = _load(monkeypatch, api=_fake_api(recorder, library="LIB"))
    monkeypatch.setattr(module, "dist", _FakeDist())

    def raising(path):
        raise OSError("no library")

    monkeypatch.setattr(module, "FLAGCXLibrary", raising)
    comm = module.PyFlagcxCommunicator(object(), "cpu")
    assert comm.available is False
    assert comm.disabled is True


@pytest.mark.parametrize("rank", [0, 1])
def test_constructor_broadcasts_unique_id_and_warms_up(monkeypatch, rank):
    """Every rank initializes the library; rank 0 shares its unique id."""
    recorder = _Recorder()
    library = _FakeLibrary(recorder)
    module = _load(monkeypatch, api=_fake_api(recorder, library=library))
    fake_dist = _FakeDist(rank=rank)
    monkeypatch.setattr(module, "dist", fake_dist)
    monkeypatch.setattr(module, "FLAGCXLibrary", lambda path: library)
    _cpu_device_boundary(monkeypatch, module)
    comm = module.PyFlagcxCommunicator(object(), "cpu")
    assert comm.available is True
    assert comm.disabled is False
    assert comm.device == torch.device("cpu")
    assert comm.comm == "COMM"
    init = [c for c in recorder if c[0] == "flagcxCommInitRank"][0]
    assert init[1] == 2
    assert init[3] == rank
    if rank == 0:
        assert init[2] is library.unique_id
        assert list(comm.unique_id.internal) == [0] * 8
    else:
        assert list(comm.unique_id.internal) == list(range(1, 9))
    # Every rank takes part in the process-group broadcast from group rank 0.
    assert fake_dist.broadcasts[0][1] == 10
    reductions = [c for c in recorder if c[0] == "flagcxAllReduce"]
    assert len(reductions) == 1
    assert reductions[0][3] == 1


def test_constructor_legacy_api_passes_unique_id_by_pointer(monkeypatch):
    """The legacy handler API is detected and receives a ctypes pointer."""
    recorder = _Recorder()
    library = _FakeLibrary(recorder, legacy=True)
    module = _load(monkeypatch, api=_fake_api(recorder, library=library))
    monkeypatch.setattr(module, "dist", _FakeDist(rank=0))
    monkeypatch.setattr(module, "FLAGCXLibrary", lambda path: library)
    _cpu_device_boundary(monkeypatch, module)
    comm = module.PyFlagcxCommunicator(object(), "cpu")
    init = [c for c in recorder if c[0] == "flagcxCommInitRank"][0]
    assert comm._legacy_unique_id_api is True
    assert type(init[2]).__name__.startswith("LP_")
    assert list(init[2].contents.internal) == [0] * 8
