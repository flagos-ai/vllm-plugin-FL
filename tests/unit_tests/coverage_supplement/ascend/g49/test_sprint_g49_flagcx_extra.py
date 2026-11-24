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

"""G49 residue for the two FlagCX modules (814b tree).

Targets only regions the earlier sprint files do not reach:

  * ``device_communicators/flagcx.py`` -- the StatelessProcessGroup entrance,
    the world-size-1 short circuit, the TXDA/MUSA constructor + ``destroy``
    state machine, ``_current_stream`` vendor branches, the disabled
    short circuits, and the collectives g5 never exercised
    (``send``/``recv``/``broadcast``/``reduce_scatterv``/group start-end).
  * ``kv_transfer/flagcx_connector.py`` -- facade role construction, the
    scheduler plan/finalisation branches, the worker block-size sync,
    region/plan helpers, ``_build_transfer_params``, the receiver coroutines,
    ``start_load_kv``, ``get_finished`` and teardown.

Everything is CPU-only: the FlagCX C facade, ZMQ sockets, vLLM base classes
and stats objects are recording fakes.  No NPU/triton and no real library.
"""

from __future__ import annotations

import asyncio
import ctypes
import importlib
import sys
import threading
import types
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

DEV_MODULE = "vllm_fl.distributed.device_communicators.flagcx"
CONNECTOR = "vllm_fl.distributed.kv_transfer.flagcx_connector"
WRAPPER = "plugin.interservice.flagcx_wrapper"


# ===========================================================================
# Part 1 - device_communicators/flagcx.py
# ===========================================================================

try:
    _DEV = importlib.import_module(DEV_MODULE)
except Exception:  # pragma: no cover - environment guard
    _DEV = None


@pytest.fixture()
def dev():
    if _DEV is None:  # pragma: no cover - environment guard
        pytest.skip("flagcx device module not importable")
    return _DEV


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


class _Stream:
    def __init__(self):
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True


class _FakeCuda:
    def device(self, device):
        return nullcontext()


class _FakeLibrary:
    """Recording stand-in for ``FLAGCXLibrary``."""

    def __init__(self, recorder, *, legacy=False):
        self.recorder = recorder
        self.comm = "COMM"
        self.unique_id = _UniqueId()
        if legacy:
            self.handler = object()
            self.legacy_contents = SimpleNamespace(contents=_UniqueId())
        else:
            self.devHandle = object()

    def _rec(self, name, *args):
        self.recorder.append((name,) + args)

    def flagcxGetUniqueId(self):
        self._rec("flagcxGetUniqueId")
        if getattr(self, "legacy_contents", None) is not None:
            return self.legacy_contents
        return self.unique_id

    def flagcxCommInitRank(self, world, unique_id, rank):
        self._rec("flagcxCommInitRank", world, unique_id, rank)
        return self.comm

    def adaptor_stream_copy(self, stream):
        self._rec("adaptor_stream_copy", stream)
        return ("flagcx-stream", id(stream))

    def adaptor_stream_free(self, stream):
        self._rec("adaptor_stream_free", stream)

    def _collective(self, name):
        def _call(*args):
            self._rec(name, *args)

        return _call

    def __getattr__(self, name):
        if name.startswith("flagcx"):
            return self._collective(name)
        raise AttributeError(name)


def _patch_symbols(monkeypatch, module, stream_factory=_Stream):
    monkeypatch.setattr(module, "buffer_type", _Buffer, raising=False)
    monkeypatch.setattr(module, "flagcxDataTypeEnum", _DataTypeEnum, raising=False)
    monkeypatch.setattr(module, "flagcxRedOpTypeEnum", _RedOpEnum, raising=False)
    monkeypatch.setattr(module, "flagcxUniqueId", _UniqueId, raising=False)
    monkeypatch.setattr(module, "flagcxComm_t", object, raising=False)
    monkeypatch.setattr(module, "current_stream", lambda: stream_factory(), raising=False)
    monkeypatch.setattr(module, "_flagcx_amd_source", False, raising=False)


def _make_comm(module, library, *, device="cpu", rank=0, world_size=2, disabled=False):
    comm = object.__new__(module.PyFlagcxCommunicator)
    comm.rank = rank
    comm.world_size = world_size
    comm.device = torch.device(device) if isinstance(device, str) else device
    comm.comm = "COMM"
    comm.available = True
    comm.disabled = disabled
    comm.flagcx = library
    return comm


def _calls(recorder, name):
    return [call for call in recorder if call[0] == name]


class _FakeDist:
    NCCL = "nccl"
    Backend = SimpleNamespace(NCCL="nccl")

    def __init__(self, rank=0, world_size=2, backend="gloo"):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend

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

    def broadcast(self, tensor, src=None, group=None):
        if self.rank == 0:
            tensor.fill_(0)
        else:
            tensor.copy_(torch.arange(1, tensor.numel() + 1, dtype=tensor.dtype))


def _fake_torch_device(monkeypatch, module):
    """Allow ``torch.device("txda"|"musa")`` on a CPU-only torch build."""

    class _Dev:
        def __init__(self, value=None):
            self.type = str(value)

        def __eq__(self, other):
            return getattr(other, "type", None) == self.type

        def __hash__(self):
            return hash(self.type)

        def __str__(self):
            return self.type

    monkeypatch.setattr(module.torch, "device", _Dev)
    return _Dev


def test_device_constructor_stateless_process_group_broadcasts_obj(monkeypatch, dev):
    module = dev
    recorder = []
    library = _FakeLibrary(recorder)
    _patch_symbols(monkeypatch, module)

    class _SPG:
        def __init__(self, rank, world_size):
            self.rank = rank
            self.world_size = world_size
            self.broadcasted = None

        def broadcast_obj(self, obj, src=0):
            self.broadcasted = (obj, src)
            return obj

    monkeypatch.setattr(module, "StatelessProcessGroup", _SPG, raising=False)
    monkeypatch.setattr(module, "FLAGCXLibrary", lambda path: library, raising=False)
    monkeypatch.setattr(module.torch, "cuda", _FakeCuda(), raising=False)
    group = _SPG(rank=1, world_size=2)

    comm = module.PyFlagcxCommunicator(group, "cpu", "libfake.so")

    assert comm.available is True and comm.disabled is False
    assert group.broadcasted[1] == 0
    assert isinstance(comm.unique_id, _UniqueId)
    init = _calls(recorder, "flagcxCommInitRank")[0]
    assert init[1] == 2 and init[3] == 1
    assert _calls(recorder, "flagcxAllReduce")

    monkeypatch.setattr(module, "dist", _FakeDist(rank=0, world_size=1), raising=False)
    alone = module.PyFlagcxCommunicator(object(), "cpu", "libfake.so")
    assert alone.available is False
    assert alone.disabled is True
    assert not hasattr(alone, "flagcx")


def test_device_constructor_txda_warmup_failure_destroys(monkeypatch, dev):
    module = dev
    recorder = []
    library = _FakeLibrary(recorder)
    _patch_symbols(monkeypatch, module)
    _fake_torch_device(monkeypatch, module)
    monkeypatch.setattr(module, "FLAGCXLibrary", lambda path: library, raising=False)
    monkeypatch.setattr(module, "dist", _FakeDist(rank=0), raising=False)
    monkeypatch.setattr(
        module.torch,
        "txda",
        SimpleNamespace(
            device=lambda device: nullcontext(),
            current_stream=lambda device: _Stream(),
        ),
        raising=False,
    )
    destroyed = []
    monkeypatch.setattr(
        module.PyFlagcxCommunicator, "destroy", lambda self: destroyed.append(True)
    )
    monkeypatch.setattr(
        module.torch,
        "zeros",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("warmup")),
    )

    with pytest.raises(Exception):
        module.PyFlagcxCommunicator(object(), "txda", "libfake.so")

    assert destroyed == [True]
    assert _calls(recorder, "flagcxCommInitRank")


def test_device_constructor_musa_warmup_failure_reraises(monkeypatch, dev):
    module = dev
    recorder = []
    library = _FakeLibrary(recorder)
    _patch_symbols(monkeypatch, module)
    _fake_torch_device(monkeypatch, module)
    monkeypatch.setattr(module, "FLAGCXLibrary", lambda path: library, raising=False)
    monkeypatch.setattr(module, "dist", _FakeDist(rank=0), raising=False)
    monkeypatch.setattr(
        module.torch,
        "musa",
        SimpleNamespace(device=lambda device: nullcontext()),
        raising=False,
    )
    monkeypatch.setattr(
        module.torch,
        "zeros",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("warmup")),
    )

    with pytest.raises(Exception):
        module.PyFlagcxCommunicator(object(), "musa", "libfake.so")

    assert _calls(recorder, "flagcxCommInitRank")


def test_device_current_stream_vendor_branches(monkeypatch, dev):
    module = dev
    _patch_symbols(monkeypatch, module)
    sentinel = object()
    monkeypatch.setattr(module, "current_stream", lambda: sentinel, raising=False)
    monkeypatch.setattr(
        module.torch, "npu", SimpleNamespace(current_stream=lambda d: "npu"), raising=False
    )
    monkeypatch.setattr(
        module.torch, "txda", SimpleNamespace(current_stream=lambda d: "txda"), raising=False
    )
    monkeypatch.setattr(
        module.torch, "cuda", SimpleNamespace(current_stream=lambda d: "cuda"), raising=False
    )
    comm = object.__new__(module.PyFlagcxCommunicator)

    comm.device = SimpleNamespace(type="npu")
    assert comm._current_stream() == "npu"
    comm.device = SimpleNamespace(type="txda")
    assert comm._current_stream() == "txda"
    comm.device = SimpleNamespace(type="cpu")
    assert comm._current_stream() is sentinel
    monkeypatch.setattr(module, "_flagcx_amd_source", True, raising=False)
    assert comm._current_stream() == "cuda"


def test_device_disabled_collectives_are_noops(monkeypatch, dev):
    module = dev
    recorder = []
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, _FakeLibrary(recorder), disabled=True)
    tensor = torch.zeros(2)

    assert comm.all_reduce(tensor) is None
    assert comm.all_gather(torch.zeros(4), tensor) is None
    assert comm.all_gatherv(torch.zeros(4), tensor, [2, 2]) is None
    assert comm.reduce_scatter(torch.zeros(1), tensor) is None
    assert comm.reduce_scatterv(torch.zeros(2), tensor, [2, 2]) is None
    assert comm.send(tensor, 1) is None
    assert comm.recv(tensor, 1) is None
    assert comm.broadcast(tensor, 1) is None
    assert recorder == []


def test_device_reduce_scatterv_records_splits(monkeypatch, dev):
    module = dev
    recorder = []
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, _FakeLibrary(recorder), rank=0)
    inp = torch.arange(4, dtype=torch.float32)
    out = torch.zeros(2)

    comm.reduce_scatterv(out, inp, [2, 2])

    reduces = _calls(recorder, "flagcxReduce")
    assert len(reduces) == 2
    assert reduces[0][1].ptr == inp.data_ptr()
    assert reduces[0][2].ptr == out.data_ptr()
    assert reduces[1][1].ptr == inp[2:].data_ptr()
    names = [call[0] for call in recorder]
    assert names.index("flagcxGroupStart") < names.index("flagcxReduce")
    assert names.index("flagcxGroupEnd") > names.index("flagcxReduce", 1)
    assert names.count("flagcxGroupStart") == 1
    assert recorder[-1][0] == "adaptor_stream_free"


def test_device_reduce_scatterv_trace_tensor_noop(monkeypatch, dev):
    module = dev
    recorder = []
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, _FakeLibrary(recorder), device="meta")

    assert comm.reduce_scatterv(
        torch.zeros(1, device="meta"), torch.zeros(2, device="meta"), [1]
    ) is None
    assert recorder == []


def test_device_send_recv_record(monkeypatch, dev):
    module = dev
    recorder = []
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, _FakeLibrary(recorder), rank=0)
    tensor = torch.arange(3, dtype=torch.float32)

    comm.send(tensor, 1)
    comm.recv(tensor, 1)

    send = _calls(recorder, "flagcxSend")[0]
    recv = _calls(recorder, "flagcxRecv")[0]
    assert send[1].ptr == tensor.data_ptr() and send[2] == 3 and send[4] == 1
    assert recv[1].ptr == tensor.data_ptr() and recv[2] == 3 and recv[4] == 1
    assert recv[-1][0] == "flagcx-stream"
    assert recorder[-1][0] == "adaptor_stream_free"


def test_device_broadcast_rank_buffers_trace_and_noop(monkeypatch, dev):
    module = dev
    recorder = []
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, _FakeLibrary(recorder), rank=1)
    tensor = torch.ones(3)

    comm.broadcast(tensor, 1)
    call = _calls(recorder, "flagcxBroadcast")[0]
    assert call[1].ptr == tensor.data_ptr()
    assert call[2].ptr == tensor.data_ptr()

    comm.broadcast(tensor, 0)
    call = _calls(recorder, "flagcxBroadcast")[1]
    assert call[1].ptr is None
    assert call[2].ptr == tensor.data_ptr()

    meta_comm = _make_comm(module, _FakeLibrary([]), device="meta")
    meta_comm.broadcast(torch.zeros(2, device="meta"), 0)
    assert recorder[-1][0] == "adaptor_stream_free"


def test_device_group_start_end_cpu_and_npu(monkeypatch, dev):
    module = dev
    recorder = []
    _patch_symbols(monkeypatch, module)
    comm = _make_comm(module, _FakeLibrary(recorder), device="cpu")

    comm.group_start()
    comm.group_end()
    starts = _calls(recorder, "flagcxGroupStart")
    assert starts[0] == ("flagcxGroupStart",)

    npu_comm = _make_comm(module, _FakeLibrary(recorder), device="cpu")
    npu_comm.device = SimpleNamespace(type="npu")
    npu_comm.group_start()
    npu_comm.group_end()
    assert _calls(recorder, "flagcxGroupStart")[-1][1] == "COMM"

    amd_comm = _make_comm(module, _FakeLibrary(recorder), device="cpu")
    monkeypatch.setattr(module, "_flagcx_amd_source", True, raising=False)
    amd_comm.group_end()
    assert _calls(recorder, "flagcxGroupEnd")[-1][1] == "COMM"


def _destroy_worker(module, **attrs):
    worker = object.__new__(module.PyFlagcxCommunicator)
    worker.device = SimpleNamespace(type="txda")
    worker.comm = "COMM"
    worker.available = True
    worker.disabled = False
    worker._destroy_lock = threading.Lock()
    worker._destroy_started = False
    worker._destroy_thread = None
    worker.flagcx = SimpleNamespace(
        lib={"flagcxCommAbort": lambda comm: 0},
        FLAGCX_CHECK=lambda rc: rc,
    )
    for key, value in attrs.items():
        setattr(worker, key, value)
    return worker


def test_device_destroy_non_txda_and_txda_success(monkeypatch, dev):
    module = dev
    _patch_symbols(monkeypatch, module)
    cpu_worker = object.__new__(module.PyFlagcxCommunicator)
    cpu_worker.device = SimpleNamespace(type="cpu")
    cpu_worker.destroy()  # no attributes beyond device are touched

    calls = []

    def _native(comm):
        calls.append(comm)
        return 0

    worker = _destroy_worker(module)
    worker.flagcx.lib["flagcxCommAbort"] = _native
    monkeypatch.setattr(
        module.torch,
        "txda",
        SimpleNamespace(device=lambda device: nullcontext()),
        raising=False,
    )

    worker.destroy()
    assert worker.comm is None
    assert worker.available is False and worker.disabled is True
    assert worker._destroy_started is True
    assert worker._destroy_thread is not None
    assert calls == ["COMM"]

    worker.destroy()  # idempotent
    assert calls == ["COMM"]


def test_device_destroy_txda_error_paths(monkeypatch, dev):
    module = dev
    _patch_symbols(monkeypatch, module)
    events = []

    class _Log:
        def exception(self, *args, **kwargs):
            events.append(("exception", args))

        def warning(self, *args, **kwargs):
            events.append(("warning", args))

    monkeypatch.setattr(module, "logger", _Log(), raising=False)
    monkeypatch.setattr(
        module.torch,
        "txda",
        SimpleNamespace(device=lambda device: nullcontext()),
        raising=False,
    )

    def _boom(comm):
        raise RuntimeError("abort-failed")

    aborted = _destroy_worker(module)
    aborted.flagcx.lib["flagcxCommAbort"] = _boom
    aborted.destroy()
    assert any(kind == "exception" for kind, _ in events)

    class _StartFailThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("cannot-start")

    monkeypatch.setattr(module.threading, "Thread", _StartFailThread)
    start_fail = _destroy_worker(module)
    start_fail.destroy()
    assert any(kind == "exception" for kind, _ in events)
    assert start_fail.available is False

    class _AliveThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True

    monkeypatch.setattr(module.threading, "Thread", _AliveThread)
    slow = _destroy_worker(module)
    slow.destroy()
    assert any(kind == "warning" for kind, _ in events)


# ===========================================================================
# Part 2 - kv_transfer/flagcx_connector.py
# ===========================================================================


def _install_wrapper(monkeypatch):
    recorder = []
    wrapper = types.ModuleType(WRAPPER)

    class _Lib:
        def __init__(self, path=None):
            self.recorder = recorder

        def flagcxP2pEngineCreate(self):
            self.recorder.append("create")
            return "ENG"

    wrapper.FLAGCXLibrary = _Lib
    plugin = types.ModuleType("plugin")
    plugin.__path__ = []
    inter = types.ModuleType("plugin.interservice")
    inter.__path__ = []
    inter.flagcx_wrapper = wrapper
    monkeypatch.setitem(sys.modules, "plugin", plugin)
    monkeypatch.setitem(sys.modules, "plugin.interservice", inter)
    monkeypatch.setitem(sys.modules, WRAPPER, wrapper)


def _drop_connector():
    sys.modules.pop(CONNECTOR, None)


@pytest.fixture()
def conn(monkeypatch):
    _install_wrapper(monkeypatch)
    _drop_connector()
    try:
        module = importlib.import_module(CONNECTOR)
    except Exception as exc:  # pragma: no cover - environment guard
        _drop_connector()
        pytest.skip(f"flagcx connector not importable: {exc!r}")
    yield module
    _drop_connector()


def _scheduler(conn, **attrs):
    obj = object.__new__(conn.FlagCXConnectorScheduler)
    obj._reqs_need_recv = {}
    obj._reqs_need_send = {}
    obj.kv_role = "kv_both"
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def _worker(conn, **attrs):
    obj = object.__new__(conn.FlagCXConnectorWorker)
    obj.tp_rank = 0
    obj.tp_size = 1
    obj._abort_request_timeout = 1
    obj.kv_role = "kv_both"
    obj.reqs_need_send = conn.SendReqMeta(reqs={}, lock=threading.Lock())
    obj.finished_sending_reqs = conn.FinishedSendReqSet(set(), threading.Lock())
    obj.finished_recving_reqs = conn.FinishedReceiveReqSet(set(), asyncio.Lock())
    obj._pull_pending = {}
    obj.kv_caches_base_addr = []
    obj.block_len_per_layer = []
    obj.kv_block_len_per_layer = []
    obj.registered_layer_names = []
    obj.registered_layer_indices = []
    obj.registered_group_indices = []
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


class _Fut:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _Topo:
    def __init__(self, *, blocks_first=True, replicated=True, targets=None):
        self.is_kv_layout_blocks_first = blocks_first
        self.local_replicates_kv_cache = replicated
        self._targets = targets or [0]

    def handshake_target_ranks(self, remote_tp_size):
        return list(self._targets)


def _region(conn, name="a", base=1000, block=64, kv=64, group=0):
    return conn.TransferRegion(
        layer_name=name,
        layer_index=0,
        base_addr=base,
        block_len=block,
        kv_block_len=kv,
        group_index=group,
    )


def test_connector_facade_constructs_roles(monkeypatch, conn):
    resume = []
    monkeypatch.setattr(
        conn.KVConnectorBase_V1, "__init__", lambda self, *a, **k: resume.append(a)
    )
    monkeypatch.setattr(
        conn, "FlagCXConnectorScheduler", lambda *a: SimpleNamespace(args=a), raising=False
    )
    monkeypatch.setattr(
        conn, "FlagCXConnectorWorker", lambda *a: SimpleNamespace(args=a), raising=False
    )
    cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(engine_id="eng")
    )

    sched = object.__new__(conn.FlagCXConnector)
    conn.FlagCXConnector.__init__(
        sched, cfg, conn.KVConnectorRole.SCHEDULER, kv_cache_config="KVC"
    )
    assert sched.engine_id == "eng"
    assert sched.connector_worker is None
    assert sched.connector_scheduler.args[1] == "eng"

    worker = object.__new__(conn.FlagCXConnector)
    conn.FlagCXConnector.__init__(worker, cfg, conn.KVConnectorRole.WORKER)
    assert worker.connector_scheduler is None
    assert worker.connector_worker.args[1] == "eng"

    broken = object.__new__(conn.FlagCXConnector)
    with pytest.raises(AssertionError):
        conn.FlagCXConnector.__init__(
            broken, cfg, conn.KVConnectorRole.SCHEDULER, kv_cache_config=None
        )


def test_connector_scheduler_init_blocks_per_sw(monkeypatch, conn):
    class _Full:
        pass

    class _SW:
        def __init__(self, sliding_window):
            self.sliding_window = sliding_window
            self.block_size = 16

    monkeypatch.setattr(conn, "FullAttentionSpec", _Full, raising=False)
    monkeypatch.setattr(conn, "SlidingWindowSpec", _SW, raising=False)
    monkeypatch.setattr(conn, "get_ip", lambda: "10.0.0.1", raising=False)
    monkeypatch.setattr(conn, "_get_side_channel_port", lambda cfg: 7000, raising=False)

    groups = [
        SimpleNamespace(kv_cache_spec=_Full(), layer_names=["l0"]),
        SimpleNamespace(kv_cache_spec=_SW(8), layer_names=["l1"]),
    ]
    kv_cache_config = SimpleNamespace(kv_cache_groups=groups, has_mamba_layers=True)
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        kv_transfer_config=SimpleNamespace(kv_role="kv_both"),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(
            data_parallel_rank=0, tensor_parallel_size=1
        ),
    )

    s = conn.FlagCXConnectorScheduler(vllm_config, "eng", kv_cache_config)

    assert s.side_channel_host == "10.0.0.1"
    assert s.side_channel_port == 7000
    assert s._has_mamba is True
    assert s._is_hma_required is True
    assert s.blocks_per_sw == [0, 2]
    assert s._reqs_need_recv == {} and s._reqs_need_send == {}


def test_connector_truncate_mamba_request_paths(conn):
    s = _scheduler(conn)
    request = SimpleNamespace(
        kv_transfer_params={},
        num_prompt_tokens=3,
        prompt_token_ids=[1, 2, 3],
        prompt_embeds=None,
        _all_token_ids=[1, 2, 3],
        max_tokens=5,
    )
    s._truncate_mamba_request_for_prefill(request)
    assert request.prompt_token_ids == [1, 2]
    assert request.num_prompt_tokens == 2
    assert request.max_tokens == 1
    assert request.kv_transfer_params["_p_side_truncated"] is True

    embeds = SimpleNamespace(
        kv_transfer_params={},
        num_prompt_tokens=3,
        prompt_token_ids=None,
        prompt_embeds=torch.zeros(3, 4),
        _all_token_ids=[1, 2, 3],
        max_tokens=5,
    )
    s._truncate_mamba_request_for_prefill(embeds)
    assert tuple(embeds.prompt_embeds.shape) == (2, 4)

    noop = SimpleNamespace(
        kv_transfer_params={},
        num_prompt_tokens=3,
        prompt_token_ids=None,
        prompt_embeds=None,
        _all_token_ids=[1, 2, 3],
        max_tokens=5,
    )
    s._truncate_mamba_request_for_prefill(noop)
    assert noop.num_prompt_tokens == 3

    already = SimpleNamespace(
        kv_transfer_params={"_p_side_truncated": True},
        num_prompt_tokens=3,
        prompt_token_ids=[1, 2, 3],
        prompt_embeds=None,
        _all_token_ids=[1, 2, 3],
        max_tokens=5,
    )
    s._truncate_mamba_request_for_prefill(already)
    assert already.num_prompt_tokens == 3


def test_connector_scheduler_num_new_matched_tokens(conn):
    s = _scheduler(conn)
    s._has_mamba = False

    req = SimpleNamespace(
        kv_transfer_params={"do_remote_prefill": True},
        prompt_token_ids=[1, 2, 3, 4, 5],
    )
    assert s.get_num_new_matched_tokens(req, 2) == (3, True)
    assert s.get_num_new_matched_tokens(req, 5) == (0, False)

    called = []
    s._has_mamba = True
    s._truncate_mamba_request_for_prefill = lambda req: called.append(req)
    decode = SimpleNamespace(
        kv_transfer_params={"do_remote_decode": True},
        prompt_token_ids=[1, 2],
    )
    assert s.get_num_new_matched_tokens(decode, 0) == (0, False)
    assert called == [decode]

    plain = SimpleNamespace(kv_transfer_params=None, prompt_token_ids=[])
    assert s.get_num_new_matched_tokens(plain, 0) == (0, False)


def test_connector_scheduler_update_state_after_alloc(monkeypatch, conn):
    warns = []
    monkeypatch.setattr(
        conn,
        "logger",
        SimpleNamespace(
            warning=lambda *a, **k: warns.append(a),
            info=lambda *a, **k: None,
        ),
        raising=False,
    )
    s = _scheduler(conn)
    s.get_sw_clipped_blocks = lambda ids: ids

    s.update_state_after_alloc(SimpleNamespace(kv_transfer_params=None), None, 0)
    assert s._reqs_need_recv == {}

    class _Blocks:
        def get_unhashed_block_ids_all_groups(self):
            return [[1], [2]]

    params = {
        "do_remote_prefill": True,
        "remote_host": "h",
        "remote_port": 1,
        "transfer_id": "t",
    }
    req = SimpleNamespace(request_id="r", kv_transfer_params=params)
    s.update_state_after_alloc(req, _Blocks(), 3)
    assert s._reqs_need_recv["r"][1] == [[1], [2]]
    assert params["do_remote_prefill"] is False

    bad = {"do_remote_prefill": True}
    s.update_state_after_alloc(
        SimpleNamespace(request_id="b", kv_transfer_params=bad), _Blocks(), 0
    )
    assert "b" not in s._reqs_need_recv
    assert bad["do_remote_prefill"] is False
    assert warns

    s.update_state_after_alloc(
        SimpleNamespace(
            request_id="d", kv_transfer_params={"do_remote_decode": True}
        ),
        None,
        0,
    )
    assert "d" not in s._reqs_need_send

    s.update_state_after_alloc(
        SimpleNamespace(
            request_id="e",
            kv_transfer_params={"do_remote_decode": True, "transfer_id": "t2"},
        ),
        None,
        0,
    )
    assert s._reqs_need_send["e"][0].request_id == "e"


def test_connector_scheduler_build_connector_meta_roles(conn):
    recv_req = SimpleNamespace(
        kv_transfer_params={
            "transfer_id": "tr",
            "remote_host": "h",
            "remote_port": 9,
        }
    )
    send_req = SimpleNamespace(kv_transfer_params={"transfer_id": "ts"})

    s = _scheduler(conn)
    s._reqs_need_recv = {"r": (recv_req, [[1]])}
    s._reqs_need_send = {"s": (send_req, [[2]])}
    meta = s.build_connector_meta(object())
    assert "r" in meta.reqs_to_recv and "s" in meta.reqs_to_send
    assert s._reqs_need_recv == {} and s._reqs_need_send == {}

    producer = _scheduler(conn, kv_role="kv_producer")
    producer._reqs_need_recv = {"r": (recv_req, [[1]])}
    producer._reqs_need_send = {"s": (send_req, [[2]])}
    meta = producer.build_connector_meta(object())
    assert not meta.reqs_to_recv and "s" in meta.reqs_to_send

    consumer = _scheduler(conn, kv_role="kv_consumer")
    consumer._reqs_need_recv = {"r": (recv_req, [[1]])}
    consumer._reqs_need_send = {"s": (send_req, [[2]])}
    meta = consumer.build_connector_meta(object())
    assert "r" in meta.reqs_to_recv and not meta.reqs_to_send


def test_connector_scheduler_request_finished_decode(conn):
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=4)
    )
    s = _scheduler(conn)
    s.vllm_config = vllm_config
    s.side_channel_host = "host"
    s.side_channel_port = 1234
    s.get_sw_clipped_blocks = lambda ids: ids

    req = SimpleNamespace(
        request_id="p",
        kv_transfer_params={"transfer_id": "t", "do_remote_decode": True},
        status=conn.RequestStatus.FINISHED_LENGTH_CAPPED,
    )
    delay, params = s.request_finished(req, ([1, 2],))
    assert delay is True
    assert s._reqs_need_send["p"][1] == ([1, 2],)
    assert params == {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_host": "host",
        "remote_port": 1234,
        "remote_tp_size": 4,
        "transfer_id": "t",
    }

    empty = SimpleNamespace(
        request_id="q",
        kv_transfer_params={"transfer_id": "t2", "do_remote_decode": True},
        status=conn.RequestStatus.FINISHED_LENGTH_CAPPED,
    )
    assert s.request_finished(empty, ([],))[0] is False
    assert "q" not in s._reqs_need_send

    other = SimpleNamespace(
        request_id="z",
        kv_transfer_params={"transfer_id": "t3", "do_remote_decode": True},
        status=conn.RequestStatus.WAITING,
    )
    assert s.request_finished(other, ([1],)) == (False, None)


def test_connector_worker_sync_block_size(monkeypatch, conn):
    monkeypatch.setattr(
        conn, "get_current_attn_backends", lambda cfg: ["b"], raising=False
    )
    monkeypatch.setattr(
        conn, "select_common_block_size", lambda bs, backends: 8, raising=False
    )
    w = _worker(conn)
    w.vllm_config = object()
    w.block_size = 16
    w._physical_blocks_per_logical_kv_block = 1

    w._sync_block_size_with_kernel()
    assert w.block_size == 8
    assert w._physical_blocks_per_logical_kv_block == 2

    w.block_size = 8
    w._physical_blocks_per_logical_kv_block = 1
    w._sync_block_size_with_kernel()
    assert w.block_size == 8
    assert w._physical_blocks_per_logical_kv_block == 1


def test_connector_worker_regions_and_plan(monkeypatch, conn):
    w = _worker(conn)
    w.kv_topo = _Topo(blocks_first=True, replicated=True)
    w._layer_specs = {"a": object()}

    regions = w._get_transfer_regions(
        base_addrs=[100],
        block_lens=[64],
        kv_block_lens=[64],
        layer_names=["a"],
        layer_indices=[0],
        group_indices=[0],
    )
    assert [r.base_addr for r in regions] == [100, 164]
    assert w._producer_cache_is_replicated() is True

    w.kv_topo = _Topo(blocks_first=False, replicated=False)
    whole = w._get_transfer_regions([100], [64], [64], ["a"], [0], [0])
    assert len(whole) == 1

    captured = {}
    monkeypatch.setattr(
        conn,
        "_compute_sender_transfer_plan",
        lambda **kwargs: captured.update(kwargs) or (True, 1, 2, 3),
        raising=False,
    )
    assert w._get_sender_transfer_plan(64, 128, 1, 2) == (True, 1, 2, 3)
    assert captured["local_tp_rank"] == 0
    assert captured["producer_cache_replicated"] is False


def test_connector_worker_build_transfer_params_coalesce(conn):
    w = _worker(conn)
    w._physical_blocks_per_logical_kv_block = 1
    w.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
    )
    w._get_sender_transfer_plan = lambda **kwargs: (True, 0, 0, 64)
    local = [_region(conn, base=1000, block=64, kv=64)]
    remote = [_region(conn, base=2000, block=64, kv=64)]
    agent = SimpleNamespace(
        remote_tp_rank=0, remote_tp_size=1, req_blocks={"d": ("t", [[3, 4]])}
    )
    send = conn.SendBlockMeta(
        p_req_id="p", transfer_id="t", local_block_ids=[[1, 2]],
        ready=threading.Event(),
    )

    src, dst, sizes = w._build_transfer_params([("d", send)], agent, local, remote)
    assert src == [1000 + 1 * 64]
    assert dst == [2000 + 3 * 64]
    assert sizes == [128]

    w._get_sender_transfer_plan = lambda **kwargs: (True, 0, 0, 64)
    big = [_region(conn, base=1000, block=128, kv=64)]
    big_remote = [_region(conn, base=2000, block=128, kv=64)]
    src2, dst2, sizes2 = w._build_transfer_params([("d", send)], agent, big, big_remote)
    assert src2 == [1000 + 128, 1000 + 2 * 128]
    assert dst2 == [2000 + 3 * 128, 2000 + 4 * 128]
    assert sizes2 == [64, 64]


def test_connector_worker_build_transfer_params_skips(conn, monkeypatch):
    class _Mamba:
        pass

    monkeypatch.setattr(conn, "MambaSpec", _Mamba, raising=False)
    monkeypatch.setattr(conn, "NULL_BLOCK_ID", -1, raising=False)
    w = _worker(conn)
    w._physical_blocks_per_logical_kv_block = 1
    w.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=_Mamba())]
    )
    w._get_sender_transfer_plan = lambda **kwargs: (True, 0, 0, 64)
    local = [_region(conn, base=1000)]
    remote = [_region(conn, base=2000)]

    empty = SimpleNamespace(
        remote_tp_rank=0, remote_tp_size=1, req_blocks={"d": ("t", [])}
    )
    send = conn.SendBlockMeta(
        p_req_id="p", transfer_id="t", local_block_ids=[[1]],
        ready=threading.Event(),
    )
    assert w._build_transfer_params([("d", send)], empty, local, remote) == ([], [], [])

    fewer = SimpleNamespace(
        remote_tp_rank=0, remote_tp_size=1, req_blocks={"d": ("t", [[3, 4]])}
    )
    assert w._build_transfer_params([("d", send)], fewer, local, remote) == ([], [], [])

    mamba_send = conn.SendBlockMeta(
        p_req_id="p", transfer_id="t", local_block_ids=[[-1, 1]],
        ready=threading.Event(),
    )
    mamba_agent = SimpleNamespace(
        remote_tp_rank=0, remote_tp_size=1, req_blocks={"d": ("t", [[-1, 2]])}
    )
    src, dst, sizes = w._build_transfer_params(
        [("d", mamba_send)], mamba_agent, local, remote
    )
    assert src == [1000 + 64] and dst == [2000 + 2 * 64] and sizes == [64]


def test_connector_worker_send_kv_to_decode(monkeypatch, conn):
    stats = SimpleNamespace(
        writes=[], transfers=[], record_failed_transfer=lambda: None,
        record_transfer=lambda **kw: stats.transfers.append(kw),
    )
    writes = []
    w = _worker(conn)
    w.kv_topo = _Topo(targets=[0])
    w._get_transfer_regions = lambda *a: [_region(conn)]
    monkeypatch.setattr(
        conn, "_align_transfer_regions", lambda l, r: (l, r, None), raising=False
    )
    monkeypatch.setattr(
        conn, "_validate_asymmetric_region_lengths", lambda *a: None, raising=False
    )
    w._build_transfer_params = lambda *a: ([11], [22], [64])
    w._get_conn = lambda session: "CONN"
    w.flagcx = SimpleNamespace(
        flagcxP2pBatchWriteSync=lambda c, s, d, n: writes.append((c, s, d, n))
    )
    w.xfer_stats = stats
    send = conn.SendBlockMeta(
        p_req_id="p", transfer_id="t", local_block_ids=[[1]],
        ready=threading.Event(),
    )
    send.ready.set()
    w.reqs_need_send.reqs["t"] = send
    meta = conn.FlagCXAgentMetadata(
        remote_hostname="h", remote_port=7, remote_tp_size=1, remote_tp_rank=0,
        req_blocks={"d": ("t", [[1]])}, kv_caches_base_addr=[], block_lens=[],
        kv_block_lens=[], registered_layer_names=[], registered_layer_indices=[],
        registered_group_indices=[],
    )

    w._send_kv_to_decode(meta)
    assert writes == [("CONN", [11], [22], [64])]
    assert "t" not in w.reqs_need_send.reqs
    assert w.finished_sending_reqs.set == {"p"}
    assert stats.transfers and stats.transfers[0]["num_descs"] == 1

    # Empty descriptor list skips the one-sided write but still finalises.
    w2 = _worker(conn)
    w2.kv_topo = _Topo(targets=[0])
    w2._get_transfer_regions = lambda *a: []
    w2._build_transfer_params = lambda *a: ([], [], [])
    w2._get_conn = lambda session: "CONN"
    w2.flagcx = SimpleNamespace(flagcxP2pBatchWriteSync=lambda *a: writes.append(a))
    w2.xfer_stats = stats
    send2 = conn.SendBlockMeta(
        p_req_id="p2", transfer_id="t2", local_block_ids=[[1]],
        ready=threading.Event(),
    )
    send2.ready.set()
    w2.reqs_need_send.reqs["t2"] = send2
    meta2 = conn.FlagCXAgentMetadata(
        remote_hostname="h", remote_port=7, remote_tp_size=1, remote_tp_rank=0,
        req_blocks={"d": ("t2", [[1]])}, kv_caches_base_addr=[], block_lens=[],
        kv_block_lens=[], registered_layer_names=[], registered_layer_indices=[],
        registered_group_indices=[],
    )
    w2._send_kv_to_decode(meta2)
    assert len(writes) == 1
    assert w2.finished_sending_reqs.set == {"p2"}


def test_connector_worker_send_kv_to_decode_timeout(conn):
    w = _worker(conn)
    w.kv_topo = _Topo(targets=[0])
    w._abort_request_timeout = 0
    send = conn.SendBlockMeta(
        p_req_id="p", transfer_id="t", local_block_ids=[[1]],
        ready=threading.Event(),
    )
    w.reqs_need_send.reqs["t"] = send
    meta = conn.FlagCXAgentMetadata(
        remote_hostname="h", remote_port=7, remote_tp_size=1, remote_tp_rank=0,
        req_blocks={"d": ("t", [[1]])}, kv_caches_base_addr=[], block_lens=[],
        kv_block_lens=[], registered_layer_names=[], registered_layer_indices=[],
        registered_group_indices=[],
    )
    with pytest.raises(RuntimeError, match="Timed out"):
        w._send_kv_to_decode(meta)


def test_connector_worker_receiver_helpers(monkeypatch, conn):
    loop = SimpleNamespace(
        set_event_loop=lambda l: loop.__dict__.setdefault("set", l),
        run_forever=lambda: loop.__dict__.setdefault("ran", True),
    )
    conn.FlagCXConnectorWorker._receiver_loop_fn(object.__new__(conn.FlagCXConnectorWorker), loop)
    assert loop.set is loop and loop.ran is True

    async def _fetch():
        w = _worker(conn)
        w.finished_recving_reqs = conn.FinishedReceiveReqSet({"r"}, asyncio.Lock())
        return await w._fetch_finished_recving(), w

    got, w = asyncio.run(_fetch())
    assert got == {"r"}
    assert w.finished_recving_reqs.set == set()

    scheduled = []

    def _ensure_future(coro):
        coro.close()
        scheduled.append(True)

    monkeypatch.setattr(conn.asyncio, "ensure_future", _ensure_future)
    monkeypatch.setattr(
        conn, "make_zmq_path", lambda a, b, c: f"{a}://{b}:{c}", raising=False
    )
    w2 = _worker(conn)
    w2.kv_topo = _Topo(targets=[0, 1])
    w2.tp_size = 2
    pull = {
        "r": conn.RecvReqMeta(
            local_block_ids=[[1]], remote_host="h", remote_port=100,
            transfer_id="t", remote_tp_size=2,
        )
    }
    asyncio.run(w2._group_kv_pull(pull))
    assert w2._pull_pending == {"r": 2}
    assert scheduled == [True, True]


def test_connector_worker_receive_kv_paths(monkeypatch, conn):
    stats = SimpleNamespace(
        failed=[], record_failed_recv=lambda: stats.failed.append(True)
    )

    class _Sock:
        def __init__(self, reply):
            self.reply = reply
            self.closed = False

        def setsockopt(self, *args):
            pass

        async def send(self, data):
            return None

        async def recv(self):
            if isinstance(self.reply, Exception):
                raise self.reply
            return self.reply

        def close(self):
            self.closed = True

    holder = {}

    def _make_sock(ctx, path, stype, bind=True, linger=None):
        return holder["sock"]

    monkeypatch.setattr(conn, "make_zmq_socket", _make_sock, raising=False)

    def _worker_for(reply):
        w = _worker(conn)
        w.hostname = "h"
        w.rpc_port = 9
        w._encoder = SimpleNamespace(encode=lambda meta: b"blob")
        w.xfer_stats = stats
        return w

    async def _run(w, reply):
        holder["sock"] = _Sock(reply)
        await w._receive_kv("tcp://p:1", {"r": ("t", [[1]])})
        return holder["sock"]

    w = _worker_for(conn.TRANS_DONE)
    used = asyncio.run(_run(w, conn.TRANS_DONE))
    assert used.closed is True
    assert w.finished_recving_reqs.set == {"r"}

    w2 = _worker_for(conn.TRANS_ERROR)
    asyncio.run(_run(w2, conn.TRANS_ERROR))
    assert w2.finished_recving_reqs.set == set()
    assert stats.failed

    w3 = _worker_for(conn.TRANS_DONE)
    asyncio.run(_run(w3, RuntimeError("sock-fail")))
    assert len(stats.failed) >= 2


def test_connector_worker_start_load_kv(monkeypatch, conn):
    scheduled = []

    def _schedule(coro, loop):
        coro.close()
        scheduled.append(loop)
        return None

    monkeypatch.setattr(conn.asyncio, "run_coroutine_threadsafe", _schedule)
    w = _worker(conn, kv_role="kv_both", receiver_loop="LOOP")

    meta = conn.FlagCXConnectorMetadata()
    meta.reqs_to_recv["r"] = object()
    meta.reqs_to_send["p"] = ("t", [[1]])
    w.start_load_kv(meta)
    assert scheduled == ["LOOP"]
    send_meta = w.reqs_need_send.reqs["t"]
    assert send_meta.p_req_id == "p"
    assert send_meta.local_block_ids == [[1]]
    assert send_meta.ready.is_set() is True

    meta2 = conn.FlagCXConnectorMetadata()
    meta2.reqs_to_send["p2"] = ("t", [])
    w.start_load_kv(meta2)
    assert w.reqs_need_send.reqs["t"] is send_meta
    assert send_meta.p_req_id == "p2"

    consumer = _worker(conn, kv_role="kv_consumer")
    meta3 = conn.FlagCXConnectorMetadata()
    meta3.reqs_to_send["p3"] = ("t3", [[2]])
    consumer.start_load_kv(meta3)
    assert consumer.reqs_need_send.reqs == {}


def test_connector_worker_get_finished_and_stats(monkeypatch, conn):
    stats = SimpleNamespace(
        expired=[], record_kv_expired_req=lambda: stats.expired.append(True),
        empty=True, clone="STATS",
    )
    monkeypatch.setattr(
        conn.asyncio,
        "run_coroutine_threadsafe",
        lambda coro, loop: (coro.close(), _Fut({"r"}))[1],
    )
    w = _worker(conn, kv_role="kv_both", receiver_loop=object())
    w.xfer_stats = stats
    w.finished_sending_reqs = conn.FinishedSendReqSet({"p"}, threading.Lock())
    expired = conn.SendBlockMeta(
        p_req_id="p2", transfer_id="te", local_block_ids=[],
        ready=threading.Event(), expire_time=0.0,
    )
    keep = conn.SendBlockMeta(
        p_req_id="p3", transfer_id="tk", local_block_ids=[],
        ready=threading.Event(),
    )
    w.reqs_need_send.reqs.update({"te": expired, "tk": keep})

    sending, recving = w.get_finished()
    assert sending == {"p", "p2"}
    assert recving == {"r"}
    assert "te" not in w.reqs_need_send.reqs and "tk" in w.reqs_need_send.reqs
    assert stats.expired == [True]

    w.xfer_stats = SimpleNamespace(is_empty=lambda: True)
    assert w.get_kv_connector_stats() is None
    w.xfer_stats = SimpleNamespace(is_empty=lambda: False, clone_and_reset=lambda: "STATS")
    assert w.get_kv_connector_stats() == "STATS"


def test_connector_worker_shutdown_and_del(monkeypatch, conn):
    events = []

    class _Ctx:
        def term(self):
            events.append("term")

    class _Exec:
        def shutdown(self, wait=True):
            events.append(("exec", wait))

    class _ThreadHandle:
        def join(self, timeout=None):
            events.append(("join", timeout))

    class _Loop:
        def is_running(self):
            return True

        def call_soon_threadsafe(self, fn):
            fn()

        def stop(self):
            events.append("stop")

    class _Lib:
        def flagcxP2pEngineDestroy(self, engine):
            events.append(("destroy", engine))

    w = _worker(conn, kv_role="kv_both", engine="ENG", flagcx=_Lib())
    w.zmq_ctx = _Ctx()
    w.async_zmq_ctx = _Ctx()
    w._sender_executor = _Exec()
    w._sender_t = _ThreadHandle()
    w.receiver_loop = _Loop()
    w._receiver_t = _ThreadHandle()

    w.shutdown()
    assert ("destroy", "ENG") in events and w.engine is None
    assert events.count("term") == 2

    failing = _worker(conn, kv_role="kv_both", engine="E2")

    class _BadLib:
        def flagcxP2pEngineDestroy(self, engine):
            raise RuntimeError("destroy-failed")

    warnings = []
    monkeypatch.setattr(
        conn,
        "logger",
        SimpleNamespace(warning=lambda *a, **k: warnings.append(a)),
        raising=False,
    )
    failing.flagcx = _BadLib()
    failing.zmq_ctx = _Ctx()
    failing.async_zmq_ctx = _Ctx()
    failing._sender_executor = _Exec()
    failing._sender_t = None
    failing.receiver_loop = _Loop()
    failing._receiver_t = _ThreadHandle()
    failing.shutdown()
    assert warnings and failing.engine is None

    called = []
    dele = _worker(conn)
    dele.shutdown = lambda: called.append(True)
    conn.FlagCXConnectorWorker.__del__(dele)
    assert called == [True]
