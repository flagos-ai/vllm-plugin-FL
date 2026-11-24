# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for vllm_fl.distributed.device_communicators.flagcx.

CPU-only tests for the loader/dispatch layer and the collective-call plumbing
of ``PyFlagcxCommunicator`` that the core tests do not reach:

* ``_is_kunlunxin`` (GEMS_VENDOR first, then current_platform fallback).
* ``_load_kunlunxin_flagcx_api`` (source-dir wrapper, wheel ``flagcx.api``
  fallback, all-missing error).
* Module import in the degraded no-FLAGCX_PATH mode (ImportError swallowed,
  ctypes names None) and with a fake ``flagcx.api`` / fake
  ``plugin.interservice.flagcx_wrapper`` installed in sys.modules.
* ``is_trace_tensor`` for real/meta/fake tensors.
* ``PyFlagcxCommunicator`` collective entry points (all_reduce/all_gather/
  all_gatherv/reduce_scatter/reduce_scatterv/send/recv/broadcast/group_*)
  driven against a recording fake ``FLAGCXLibrary``: exact call contracts,
  stream copy/free pairing, disabled short-circuit, and trace-tensor
  short-circuit.
* Constructor library-path resolution and the disabled-on-init path.

All sys.modules surgery is save/install/verify/restore inside try/finally so
tests stay order-independent. Fakes are never installed through
monkeypatch.setitem on entries this suite pops.
"""

from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

pytestmark = [pytest.mark.supplementary]

_MODULE = "vllm_fl.distributed.device_communicators.flagcx"


def _import_fresh():
    """Import the flagcx module with its cached entry dropped.

    The supplementary tests need module-level branches (FLAGCX_PATH handling,
    vendor dispatch) to re-execute under controlled sys.modules contents, so
    the cached entry is popped, a fresh copy executed, and the caller-visible
    entry restored afterwards.
    """
    import importlib

    saved = sys.modules.pop(_MODULE, None)
    try:
        return importlib.import_module(_MODULE)
    finally:
        if saved is not None:
            sys.modules[_MODULE] = saved


def _install_fake_flagcx_api():
    """Install a fake ``flagcx`` package with ``flagcx.api`` in sys.modules.

    Returns (restore, api_module). Restore puts every touched entry back to
    its previous state (or removes it).
    """
    api_mod = types.ModuleType("flagcx.api")
    api_mod.FLAGCXLibrary = type("FakeLib", (), {})
    api_mod.buffer_type = lambda ptr=None: ("buf", ptr)
    api_mod.flagcxComm_t = object()
    api_mod.flagcxDataTypeEnum = SimpleNamespace(
        from_torch=lambda dtype: f"dtype:{dtype}"
    )
    api_mod.flagcxRedOpTypeEnum = SimpleNamespace(from_torch=lambda op: f"op:{op}")
    api_mod.flagcxUniqueId = type("FakeUniqueId", (), {"internal": [1, 2, 3, 4]})
    pkg = types.ModuleType("flagcx")
    pkg.api = api_mod
    pkg.__path__ = []  # mark as package for submodule resolution

    touched = ("flagcx", "flagcx.api")
    saved = {name: sys.modules.get(name) for name in touched}
    sys.modules["flagcx"] = pkg
    sys.modules["flagcx.api"] = api_mod

    def restore():
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev

    return restore, api_mod


def _write_wrapper(root):
    """Write a minimal plugin/interservice/flagcx_wrapper.py under root."""
    interservice_dir = root / "plugin" / "interservice"
    interservice_dir.mkdir(parents=True, exist_ok=True)
    (interservice_dir / "flagcx_wrapper.py").write_text(
        "FLAGCXLibrary = type('WL', (), {})\n"
        "buffer_type = lambda p=None: ('buf', p)\n"
        "flagcxComm_t = object()\n"
        "flagcxDataTypeEnum = object()\n"
        "flagcxUniqueId = object()\n"
        "flagcxRedOpTypeEnum = object()\n",
        encoding="utf-8",
    )


def _pop_plugin_modules():
    """Remove a freshly imported fake plugin tree from sys.modules."""
    for name in (
        "plugin.interservice.flagcx_wrapper",
        "plugin.interservice",
        "plugin",
    ):
        sys.modules.pop(name, None)


class RecordingLibrary:
    """Fake FLAGCXLibrary recording every collective/stream call."""

    def __init__(self, comm_handle="comm-0", stream="stream-0"):
        self.comm_handle = comm_handle
        self.stream = stream
        self.calls = []
        self.stream_copies = 0
        self.stream_frees = 0

    def __call__(self, library_path=None):
        self.constructor_path = library_path
        return self

    # -- comm lifecycle -------------------------------------------------------
    def flagcxGetUniqueId(self):
        self.calls.append(("GetUniqueId",))
        return SimpleNamespace(internal=[7, 8, 9, 10])

    def flagcxCommInitRank(self, world_size, unique_id, rank):
        self.calls.append(("CommInitRank", world_size, unique_id, rank))
        return self.comm_handle

    # -- stream adaptor -------------------------------------------------------
    def adaptor_stream_copy(self, stream):
        self.stream_copies += 1
        self.copied_stream = stream
        return self.stream

    def adaptor_stream_free(self, stream):
        self.stream_frees += 1

    # -- collectives ----------------------------------------------------------
    def flagcxAllReduce(self, inbuf, outbuf, count, dtype, op, comm, stream):
        self.calls.append(("AllReduce", inbuf, outbuf, count, dtype, op, comm, stream))

    def flagcxAllGather(self, inbuf, outbuf, count, dtype, comm, stream):
        self.calls.append(("AllGather", inbuf, outbuf, count, dtype, comm, stream))

    def flagcxBroadcast(self, inbuf, outbuf, count, dtype, root, comm, stream):
        self.calls.append(("Broadcast", inbuf, outbuf, count, dtype, root, comm, stream))

    def flagcxReduceScatter(self, inbuf, outbuf, count, dtype, op, comm, stream):
        self.calls.append(("ReduceScatter", inbuf, outbuf, count, dtype, op, comm, stream))

    def flagcxReduce(self, inbuf, outbuf, count, dtype, op, root, comm, stream):
        self.calls.append(("Reduce", inbuf, outbuf, count, dtype, op, root, comm, stream))

    def flagcxSend(self, inbuf, count, dtype, dst, comm, stream):
        self.calls.append(("Send", inbuf, count, dtype, dst, comm, stream))

    def flagcxRecv(self, inbuf, count, dtype, src, comm, stream):
        self.calls.append(("Recv", inbuf, count, dtype, src, comm, stream))

    def flagcxGroupStart(self, comm=None):
        self.calls.append(("GroupStart", comm))

    def flagcxGroupEnd(self, comm=None):
        self.calls.append(("GroupEnd", comm))


class StatelessGroup:
    """Minimal StatelessProcessGroup-shaped stand-in."""

    def __init__(self, rank=0, world_size=2):
        self.rank = rank
        self.world_size = world_size

    def broadcast_obj(self, value, src=0):
        self.broadcasted = (value, src)
        return value


def _fake_device(device: str):
    """torch.device for cpu/cuda; a SimpleNamespace for vendor device types
    that this torch build refuses to construct (npu/txda/musa/ptpu)."""
    prefix = device.split(":")[0]
    if prefix in ("cpu", "cuda"):
        return torch.device(device)
    type_name, _, index = device.partition(":")
    return SimpleNamespace(type=type_name, index=int(index or 0))


def _make_comm(module, library, group=None, device="cpu", world_size=2, rank=0):
    """Bypass __init__ (device contexts/warmup collectives) and build a comm."""
    comm = module.PyFlagcxCommunicator.__new__(module.PyFlagcxCommunicator)
    comm.comm = library.comm_handle
    comm.flagcx = library
    comm.group = group if group is not None else StatelessGroup(rank, world_size)
    comm.rank = rank
    comm.world_size = world_size
    comm.device = _fake_device(device)
    comm.available = True
    comm.disabled = False
    comm._all_reduce_stream = None
    comm._destroy_lock = threading.Lock()
    comm._destroy_started = False
    comm._destroy_thread = None
    return comm


class TestIsKunlunxin:
    def test_env_vendor_name_wins(self, monkeypatch):
        module = _import_fresh()
        monkeypatch.setenv("GEMS_VENDOR", "kunlunxin")
        assert module._is_kunlunxin() is True

    def test_env_vendor_name_other_vendor(self, monkeypatch):
        module = _import_fresh()
        monkeypatch.setenv("GEMS_VENDOR", "hygon")
        assert module._is_kunlunxin() is False

    def test_env_vendor_name_whitespace_is_stripped(self, monkeypatch):
        module = _import_fresh()
        monkeypatch.setenv("GEMS_VENDOR", "  Kunlunxin  ")
        assert module._is_kunlunxin() is True

    def test_no_env_falls_back_to_current_platform(self, monkeypatch):
        module = _import_fresh()
        monkeypatch.delenv("GEMS_VENDOR", raising=False)

        from vllm.platforms import current_platform

        expected = getattr(current_platform, "vendor_name", None) == "kunlunxin"
        assert module._is_kunlunxin() is expected


class TestLoadKunlunxinFlagcxApi:
    def test_source_dir_wrapper_is_imported_and_returned(
        self, monkeypatch, tmp_path
    ):
        """A FLAGCX_PATH dir containing plugin/interservice wins over flagcx.api."""
        module = _import_fresh()
        monkeypatch.delenv("GEMS_VENDOR", raising=False)
        source_root = tmp_path / "flagcx-source"
        source_root.mkdir()
        _write_wrapper(source_root)

        api_restore, _ = _install_fake_flagcx_api()
        saved_path = list(sys.path)
        try:
            api = module._load_kunlunxin_flagcx_api(str(source_root))
            assert api.__name__ == "plugin.interservice.flagcx_wrapper"
            assert hasattr(api, "FLAGCXLibrary")
            assert str(source_root) in sys.path
        finally:
            api_restore()
            sys.path[:] = saved_path
            _pop_plugin_modules()

    def test_no_wrapper_falls_back_to_flagcx_api(self):
        module = _import_fresh()
        api_restore, api_mod = _install_fake_flagcx_api()
        try:
            api = module._load_kunlunxin_flagcx_api("")
            assert api is api_mod
            assert api.__name__ == "flagcx.api"
        finally:
            api_restore()

    def test_missing_everything_raises_import_error(self, monkeypatch):
        module = _import_fresh()
        monkeypatch.delenv("GEMS_VENDOR", raising=False)
        with pytest.raises(ImportError):
            module._load_kunlunxin_flagcx_api(None)


class TestModuleImportModes:
    def test_degraded_mode_without_flagcx_path(self, monkeypatch):
        """Without FLAGCX_PATH the module import degrades, not raises."""
        monkeypatch.delenv("FLAGCX_PATH", raising=False)
        monkeypatch.delenv("GEMS_VENDOR", raising=False)

        module = _import_fresh()

        assert module._flagcx_available is False
        assert module.FLAGCXLibrary is None
        assert module.buffer_type is None
        assert module.flagcxComm_t is None
        assert module.flagcxDataTypeEnum is None
        assert module.flagcxUniqueId is None
        assert module.flagcxRedOpTypeEnum is None
        assert module._flagcx_import_error is not None

    def test_kunlunxin_with_fake_flagcx_api_marks_packaged(self, monkeypatch):
        monkeypatch.setenv("GEMS_VENDOR", "kunlunxin")
        monkeypatch.setenv("FLAGCX_PATH", "/nonexistent/flagcx")

        api_restore, api_mod = _install_fake_flagcx_api()
        try:
            module = _import_fresh()
            assert module._flagcx_available is True
            assert module._flagcx_packaged_library is True
            assert module.FLAGCXLibrary is api_mod.FLAGCXLibrary
        finally:
            api_restore()

    def test_kunlunxin_with_fake_source_wrapper(self, monkeypatch, tmp_path):
        """A source layout (plugin.interservice.flagcx_wrapper) under FLAGCX_PATH."""
        monkeypatch.setenv("GEMS_VENDOR", "kunlunxin")
        source_root = tmp_path / "src"
        source_root.mkdir()
        _write_wrapper(source_root)
        monkeypatch.setenv("FLAGCX_PATH", str(source_root))

        saved_path = list(sys.path)
        try:
            module = _import_fresh()
            assert module._flagcx_available is True
            assert module._flagcx_packaged_library is False
            assert module.flagcxUniqueId is not None
            assert str(source_root) in sys.path
        finally:
            sys.path[:] = saved_path
            _pop_plugin_modules()

    def test_plain_source_layout_import(self, monkeypatch, tmp_path):
        """Non-kunlunxin path: FLAGCX_PATH dir is appended and the wrapper
        (fake installed via sys.modules) is imported."""
        monkeypatch.delenv("GEMS_VENDOR", raising=False)
        source_root = tmp_path / "plain"
        source_root.mkdir()
        monkeypatch.setenv("FLAGCX_PATH", str(source_root))

        wrapper = types.ModuleType("plugin.interservice.flagcx_wrapper")
        wrapper.FLAGCXLibrary = type("WL", (), {})
        wrapper.buffer_type = lambda p=None: p
        wrapper.flagcxComm_t = object()
        wrapper.flagcxDataTypeEnum = object()
        wrapper.flagcxUniqueId = object()
        wrapper.flagcxRedOpTypeEnum = object()
        plugin = types.ModuleType("plugin")
        interservice = types.ModuleType("plugin.interservice")
        interservice.flagcx_wrapper = wrapper
        plugin.interservice = interservice

        touched = (
            "plugin",
            "plugin.interservice",
            "plugin.interservice.flagcx_wrapper",
        )
        saved = {n: sys.modules.get(n) for n in touched}
        saved_path = list(sys.path)
        try:
            sys.modules.update(
                {
                    "plugin": plugin,
                    "plugin.interservice": interservice,
                    "plugin.interservice.flagcx_wrapper": wrapper,
                }
            )
            module = _import_fresh()
            assert module._flagcx_available is True
            assert module.FLAGCXLibrary is wrapper.FLAGCXLibrary
            assert str(source_root) in sys.path
        finally:
            for name, prev in saved.items():
                if prev is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prev
            sys.path[:] = saved_path


class TestIsTraceTensor:
    def test_none_and_non_tensor_are_not_trace(self):
        module = _import_fresh()
        assert module.is_trace_tensor(None) is False
        assert module.is_trace_tensor([1, 2, 3]) is False
        assert module.is_trace_tensor("tensor") is False

    def test_cpu_tensor_is_not_trace(self):
        module = _import_fresh()
        assert module.is_trace_tensor(torch.zeros(4)) is False

    def test_meta_tensor_is_trace(self):
        module = _import_fresh()
        assert module.is_trace_tensor(torch.zeros(4, device="meta")) is True

    def test_fake_tensor_is_trace(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        module = _import_fresh()
        with FakeTensorMode():
            fake = torch.zeros(4)
        assert module.is_trace_tensor(fake) is True


def _patch_collective_globals(monkeypatch, module):
    """Give the degraded module the ctypes names the collectives reference."""
    monkeypatch.setattr(module, "buffer_type", lambda ptr=None: ("buf", ptr))
    monkeypatch.setattr(
        module,
        "flagcxDataTypeEnum",
        SimpleNamespace(from_torch=lambda dtype: f"dtype:{dtype}"),
    )
    monkeypatch.setattr(
        module,
        "flagcxRedOpTypeEnum",
        SimpleNamespace(from_torch=lambda op: f"op:{op}"),
    )
    monkeypatch.setattr(module, "current_stream", lambda: "default-stream")


class TestCollectivesAgainstRecordingLibrary:
    """Drive each collective entry point with a recording library."""

    @pytest.fixture
    def comm(self, monkeypatch):
        module = _import_fresh()
        _patch_collective_globals(monkeypatch, module)
        library = RecordingLibrary()
        return module, library, _make_comm(module, library)

    def test_all_reduce_allocates_output_and_calls_library(self, comm):
        module, library, c = comm
        inp = torch.arange(4, dtype=torch.float32)

        out = c.all_reduce(inp)

        assert out is not inp
        assert out.shape == inp.shape
        assert len(library.calls) == 1
        name, inbuf, outbuf, count, dtype, op, commh, stream = library.calls[0]
        assert name == "AllReduce"
        assert inbuf == ("buf", inp.data_ptr())
        assert outbuf == ("buf", out.data_ptr())
        assert count == 4
        assert dtype == f"dtype:{torch.float32}"
        assert op == f"op:{torch.distributed.ReduceOp.SUM}"
        assert commh == library.comm_handle
        assert stream == library.stream
        # Stream copy/free are paired.
        assert library.stream_copies == library.stream_frees == 1

    def test_all_reduce_with_explicit_out_tensor(self, comm):
        module, library, c = comm
        inp = torch.arange(3, dtype=torch.float32)
        out = torch.zeros(3, dtype=torch.float32)
        explicit_stream = object()
        result = c.all_reduce(inp, out_tensor=out, stream=explicit_stream)
        assert result is out
        assert library.calls[0][2] == ("buf", out.data_ptr())

    def test_all_reduce_trace_tensor_short_circuits(self, comm):
        module, library, c = comm
        meta = torch.zeros(4, device="meta")
        out = c.all_reduce(meta)
        # torch.compile tracing: an empty stand-in is returned, no native call.
        assert out is not None and out.shape == meta.shape
        assert library.calls == []
        assert library.stream_copies == 0

    def test_all_reduce_disabled_returns_none(self, comm):
        module, library, c = comm
        c.disabled = True
        assert c.all_reduce(torch.zeros(4)) is None
        assert library.calls == []

    def test_all_gather_calls_library_and_returns(self, comm):
        module, library, c = comm
        inp = torch.arange(2, dtype=torch.float32)
        out = torch.zeros(4, dtype=torch.float32)

        c.all_gather(out, inp)

        name, inbuf, outbuf, count, dtype, commh, stream = library.calls[0]
        assert name == "AllGather"
        assert inbuf == ("buf", inp.data_ptr())
        assert outbuf == ("buf", out.data_ptr())
        assert count == 2
        assert library.stream_copies == library.stream_frees == 1

    def test_all_gather_trace_tensor_short_circuits(self, comm):
        module, library, c = comm
        out = c.all_gather(torch.zeros(4), torch.zeros(2, device="meta"))
        assert out is None
        assert library.calls == []

    def test_all_gatherv_broadcasts_each_root_slice(self, comm):
        module, library, c = comm
        inp = torch.arange(3, dtype=torch.float32)
        out = torch.zeros(7, dtype=torch.float32)

        c.all_gatherv(out, inp, sizes=[3, 4])

        names = [call[0] for call in library.calls]
        assert names == ["GroupStart", "Broadcast", "Broadcast", "GroupEnd"]
        # No comm handle passed to group start/end on the non-npu path.
        assert library.calls[0][1] is None
        roots = [call[5] for call in library.calls[1:3]]
        assert roots == [0, 1]
        counts = [call[3] for call in library.calls[1:3]]
        assert counts == [3, 4]
        outbufs = [call[2] for call in library.calls[1:3]]
        assert outbufs[0] == ("buf", out[0:3].data_ptr())
        assert outbufs[1] == ("buf", out[3:7].data_ptr())
        assert library.stream_copies == library.stream_frees == 1

    def test_all_gatherv_trace_tensor_short_circuits(self, comm):
        module, library, c = comm
        c.device = torch.device("meta")
        meta = torch.zeros(3, device="meta")
        out2 = torch.zeros(6, device="meta")
        c.all_gatherv(out2, meta, sizes=[3, 3])
        assert library.calls == []
        assert library.stream_copies == 0

    def test_reduce_scatter_calls_library(self, comm):
        module, library, c = comm
        inp = torch.arange(8, dtype=torch.float32)
        out = torch.zeros(4, dtype=torch.float32)

        c.reduce_scatter(out, inp)

        name, inbuf, outbuf, count, dtype, op, commh, stream = library.calls[0]
        assert name == "ReduceScatter"
        assert count == 4  # output numel
        assert op == f"op:{torch.distributed.ReduceOp.SUM}"
        assert library.stream_copies == library.stream_frees == 1

    def test_reduce_scatterv_reduces_each_root_chunk(self, comm):
        module, library, c = comm
        inp = torch.arange(7, dtype=torch.float32)
        out = torch.zeros(3, dtype=torch.float32)

        c.reduce_scatterv(out, inp, sizes=[3, 4])

        names = [call[0] for call in library.calls]
        assert names == ["GroupStart", "Reduce", "Reduce", "GroupEnd"]
        reduces = library.calls[1:3]
        assert reduces[0][1] == ("buf", inp[0:3].data_ptr())
        assert reduces[1][1] == ("buf", inp[3:7].data_ptr())
        assert reduces[0][3] == 3
        assert reduces[1][3] == 4
        # flagcxReduce(inbuf, outbuf, count, dtype, op, root, comm, stream)
        assert [r[6] for r in reduces] == [0, 1]
        assert all(r[2] == ("buf", out.data_ptr()) for r in reduces)
        assert library.stream_copies == library.stream_frees == 1

    def test_send_calls_library_with_dst(self, comm):
        module, library, c = comm
        t = torch.arange(5, dtype=torch.float32)

        c.send(t, dst=1)

        name, inbuf, count, dtype, dst, commh, stream = library.calls[0]
        assert name == "Send"
        assert dst == 1
        assert count == 5
        assert library.stream_copies == library.stream_frees == 1

    def test_recv_calls_library_with_src(self, comm):
        module, library, c = comm
        t = torch.zeros(5, dtype=torch.float32)

        c.recv(t, src=1)

        name, inbuf, count, dtype, src, commh, stream = library.calls[0]
        assert name == "Recv"
        assert src == 1
        assert count == 5

    def test_broadcast_root_uses_send_and_recv_buffers(self, comm):
        module, library, c = comm
        t = torch.arange(6, dtype=torch.float32)

        c.broadcast(t, src=c.rank)

        name, sendbuff, recvbuff, count, dtype, src, commh, stream = library.calls[0]
        assert name == "Broadcast"
        assert sendbuff == ("buf", t.data_ptr())
        assert recvbuff == ("buf", t.data_ptr())

    def test_broadcast_non_root_uses_empty_send_buffer(self, comm):
        module, library, c = comm
        t = torch.arange(6, dtype=torch.float32)

        c.broadcast(t, src=1)

        name, sendbuff, recvbuff, count, dtype, src, commh, stream = library.calls[0]
        assert sendbuff == ("buf", None)  # buffer_type() with no pointer
        assert recvbuff == ("buf", t.data_ptr())
        assert src == 1

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("broadcast", (torch.zeros(4, device="meta"), 0)),
            ("send", (torch.zeros(4, device="meta"), 1)),
            ("recv", (torch.zeros(4, device="meta"), 0)),
        ],
    )
    def test_point_to_point_trace_tensor_short_circuits(self, comm, method, args):
        module, library, c = comm
        # The device assert runs before the trace check, so the comm must be
        # bound to the same (meta) device the traced tensor lives on.
        c.device = torch.device("meta")
        getattr(c, method)(*args)
        assert library.calls == []
        assert library.stream_copies == 0

    def test_reduce_scatter_trace_tensor_short_circuits(self, comm):
        module, library, c = comm
        c.device = torch.device("meta")
        out = torch.zeros(2, device="meta")
        meta = torch.zeros(4, device="meta")
        c.reduce_scatter(out, meta)
        assert library.calls == []

    def test_reduce_scatterv_trace_tensor_short_circuits(self, comm):
        module, library, c = comm
        c.device = torch.device("meta")
        out = torch.zeros(2, device="meta")
        meta = torch.zeros(4, device="meta")
        c.reduce_scatterv(out, meta, sizes=[2, 2])
        assert library.calls == []


class TestGroupStartEnd:
    @pytest.mark.parametrize("device", ["cpu", "npu:0"])
    def test_group_start_end_passes_comm_only_for_npu(self, device):
        module = _import_fresh()
        library = RecordingLibrary()
        c = _make_comm(module, library, device=device)

        c.group_start()
        c.group_end()

        assert [call[0] for call in library.calls] == ["GroupStart", "GroupEnd"]
        expected = library.comm_handle if device.startswith("npu") else None
        assert library.calls[0][1] == expected
        assert library.calls[1][1] == expected

    def test_amd_source_mode_passes_comm(self, monkeypatch):
        module = _import_fresh()
        monkeypatch.setattr(module, "_flagcx_amd_source", True)
        library = RecordingLibrary()
        c = _make_comm(module, library)

        c.group_start()
        c.group_end()

        assert library.calls[0][1] == library.comm_handle
        assert library.calls[1][1] == library.comm_handle


class TestConstructorPaths:
    @pytest.fixture
    def dist_mocks(self, monkeypatch):
        """Satisfy the torch.distributed prefix of __init__ on a fake group."""
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_backend", lambda group: dist.Backend.GLOO)
        monkeypatch.setattr(dist, "get_rank", lambda group: 0)
        monkeypatch.setattr(dist, "get_world_size", lambda group: 2)

    def test_world_size_one_is_disabled_without_library(self, dist_mocks):
        module = _import_fresh()
        library = RecordingLibrary()
        c = module.PyFlagcxCommunicator(
            StatelessGroup(world_size=1), "cpu", library_path="unused"
        )
        assert c.available is False
        assert c.disabled is True
        assert library.calls == []

    def test_init_error_when_flagcx_library_missing(self, dist_mocks, monkeypatch):
        module = _import_fresh()
        monkeypatch.setattr(
            module,
            "FLAGCXLibrary",
            Mock(side_effect=ImportError("bindings gone")),
        )
        monkeypatch.setattr(module, "_flagcx_packaged_library", True)
        c = module.PyFlagcxCommunicator(StatelessGroup(), "cpu", library_path=None)
        assert c.disabled is True
        assert c.available is False
        assert isinstance(c.init_error, ImportError)

    def test_library_path_resolution_from_flagcx_path(
        self, dist_mocks, monkeypatch
    ):
        """Without FLAGCX_LIB_PATH, FLAGCX_PATH/build/lib/libflagcx.so is used."""
        module = _import_fresh()
        constructor_paths = []

        class PathRecorder:
            def __init__(self, path):
                constructor_paths.append(path)
                raise RuntimeError("stop after recording the path")

        monkeypatch.setattr(module, "FLAGCXLibrary", PathRecorder)
        monkeypatch.setattr(module, "_flagcx_packaged_library", False)
        monkeypatch.setattr(module, "_flagcx_wheel_library", None)
        monkeypatch.delenv("FLAGCX_LIB_PATH", raising=False)
        monkeypatch.setenv("FLAGCX_PATH", "/opt/flagcx")

        c = module.PyFlagcxCommunicator(StatelessGroup(), "cpu", library_path=None)
        assert constructor_paths == ["/opt/flagcx/build/lib/libflagcx.so"]
        assert isinstance(c.init_error, RuntimeError)
        assert c.disabled is True

    def test_explicit_library_path_is_forwarded(self, dist_mocks, monkeypatch):
        module = _import_fresh()
        constructor_paths = []

        class PathRecorder:
            def __init__(self, path):
                constructor_paths.append(path)
                raise RuntimeError("stop here")

        monkeypatch.setattr(module, "FLAGCXLibrary", PathRecorder)
        monkeypatch.setattr(module, "_flagcx_packaged_library", False)
        monkeypatch.setattr(module, "_flagcx_wheel_library", None)
        monkeypatch.setenv("FLAGCX_PATH", "/opt/flagcx")

        c = module.PyFlagcxCommunicator(
            StatelessGroup(), "cpu", library_path="/custom/lib.so"
        )
        assert constructor_paths == ["/custom/lib.so"]
        assert isinstance(c.init_error, RuntimeError)

    def test_wheel_library_default_when_no_flagcx_path(
        self, dist_mocks, monkeypatch
    ):
        module = _import_fresh()
        constructor_paths = []

        class PathRecorder:
            def __init__(self, path):
                constructor_paths.append(path)
                raise RuntimeError("stop after recording the path")

        monkeypatch.setattr(module, "FLAGCXLibrary", PathRecorder)
        monkeypatch.setattr(module, "_flagcx_packaged_library", False)
        monkeypatch.setattr(module, "_flagcx_wheel_library", "/wheel/libflagcx.so")
        monkeypatch.delenv("FLAGCX_LIB_PATH", raising=False)
        monkeypatch.delenv("FLAGCX_PATH", raising=False)

        c = module.PyFlagcxCommunicator(StatelessGroup(), "cpu", library_path=None)
        assert constructor_paths == ["/wheel/libflagcx.so"]
        assert isinstance(c.init_error, RuntimeError)


class TestCurrentStreamSelection:
    def test_txda_selects_torch_txda_current_stream(self, monkeypatch):
        module = _import_fresh()
        expected = object()
        seen = []
        monkeypatch.setattr(
            torch,
            "txda",
            SimpleNamespace(
                current_stream=lambda device: seen.append(device) or expected
            ),
            raising=False,
        )
        c = _make_comm(module, RecordingLibrary(), device="txda:1")
        assert c._current_stream() is expected
        assert seen == [_fake_device("txda:1")]

    def test_npu_selects_torch_npu_current_stream(self, monkeypatch):
        module = _import_fresh()
        expected = object()
        monkeypatch.setattr(
            torch,
            "npu",
            SimpleNamespace(current_stream=lambda device: expected),
            raising=False,
        )
        c = _make_comm(module, RecordingLibrary(), device="npu:0")
        assert c._current_stream() is expected

    def test_cuda_uses_upstream_helper(self, monkeypatch):
        module = _import_fresh()
        expected = object()
        monkeypatch.setattr(module, "current_stream", lambda: expected)
        c = _make_comm(module, RecordingLibrary(), device="cuda:0")
        assert c._current_stream() is expected

    def test_amd_source_selects_torch_cuda_current_stream(self, monkeypatch):
        module = _import_fresh()
        monkeypatch.setattr(module, "_flagcx_amd_source", True)
        expected = object()
        monkeypatch.setattr(
            torch.cuda, "current_stream", lambda device: expected
        )
        c = _make_comm(module, RecordingLibrary(), device="cuda:0")
        assert c._current_stream() is expected


class TestCopyFlagcxStream:
    def test_npu_wraps_stream_with_cuda_stream_attribute(self, monkeypatch):
        module = _import_fresh()
        library = RecordingLibrary()
        c = _make_comm(module, library, device="npu:0")
        npu_stream = SimpleNamespace(npu_stream=0xBEEF)
        monkeypatch.setattr(
            torch, "npu", SimpleNamespace(stream=lambda s: None), raising=False
        )
        c._copy_flagcx_stream(npu_stream)
        # The adaptor received a namespace carrying cuda_stream.
        assert library.stream_copies == 1
        assert library.copied_stream.cuda_stream == 0xBEEF

    def test_cpu_device_passes_stream_through(self):
        module = _import_fresh()
        library = RecordingLibrary()
        c = _make_comm(module, library, device="cpu")
        raw = object()
        c._copy_flagcx_stream(raw)
        assert library.stream_copies == 1
        assert library.copied_stream is raw
