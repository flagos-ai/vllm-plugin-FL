# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for the remaining ``flagcx.py`` communicator lines.

``test_flagcx_transfer_plan.py`` (which must not be modified) already covers
the API loaders' success paths, the constructor's cpu/npu/ptpu/txda matrix,
``destroy`` on TXDA and every collective's main-line library call with an
explicit stream.  Still uncovered after that file runs:

* the loader guards for a path ``importlib`` cannot resolve into a spec
  (``_load_flagcx_wheel_bindings`` and ``_load_amd_flagcx_api`` raise), and
  ``_load_kunlunxin_flagcx_api`` with a real directory whose
  ``plugin.interservice`` import fails (the ``except ImportError: pass``
  fall-through to the ``flagcx`` package);
* ``__init__`` with a legacy-API library at rank 0 (the ``.contents``
  unique-id read) and with a ``musa`` device (the ``torch.musa.device``
  context manager);
* ``destroy`` corner branches: a failing ``thread.start()`` and the
  timeout-expired warning when the abort thread stays alive;
* ``_current_stream`` on an AMD-source module (``torch.cuda.current_stream``);
* the remaining collective guard branches the mainline tests always pass a
  stream around: the disabled early return, the ``stream=None`` resolution
  through ``_current_stream()`` and the trace-tensor early return of
  ``all_gather``/``all_gatherv``/``reduce_scatter``/``reduce_scatterv``/
  ``send``/``recv``/``broadcast``.

The FlagCX C library is never present: the module under test is loaded from
source under a unique name against the same stubbed
``plugin.interservice.flagcx_wrapper`` boundary as the sibling file.  The
doubles are duplicated here rather than imported from a test module, keeping
this file self-contained and collection-order independent.
"""

import contextlib
import ctypes
import importlib.util
import logging
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = pytest.mark.supplementary

_SOURCE_ROOT = Path(__file__).resolve().parents[4]
_FLAGCX_SOURCE = _SOURCE_ROOT / "vllm_fl/distributed/device_communicators/flagcx.py"
_COMMUNICATOR_SUBJECT = "_test_flagcx_guards_communicator"

# Collectives whose stream=None branch is still uncovered (all_gather's
# disabled/trace returns are already covered by the sibling file).
_STREAM_COLLECTIVES = [
    "all_gather",
    "all_gatherv",
    "reduce_scatter",
    "reduce_scatterv",
    "send",
    "recv",
    "broadcast",
]

# Collectives whose disabled/trace returns are still uncovered.
_GUARDED_COLLECTIVES = _STREAM_COLLECTIVES[1:]


class _FakeUniqueId(ctypes.Structure):
    """ctypes-backed unique id double (legacy API needs ``ctypes.pointer``)."""

    _fields_ = [("internal", ctypes.c_ubyte * 8)]


class _StubGroup:
    """StatelessProcessGroup double for rank 0 of a 2-rank world."""

    rank = 0
    world_size = 2

    def broadcast_obj(self, value, src):
        return value


class _FakeDevice:
    """Duck-typed torch.device for exotic vendor types (txda here).

    ``torch.device`` cannot be subclassed, so the txda branch of ``destroy``
    is exercised through this double installed on a per-module torch proxy.
    """

    def __new__(cls, value):
        if isinstance(value, _FakeDevice):
            return value  # torch.device(self.device) re-wrap: identity.
        return super().__new__(cls)

    def __init__(self, value):
        if isinstance(value, _FakeDevice):
            return
        name = str(value)
        type_, _, idx = name.partition(":")
        self._type = type_
        self.index = int(idx) if idx else 0
        self.entered = []

    @property
    def type(self):
        return self._type

    def __eq__(self, other):
        return (
            isinstance(other, _FakeDevice)
            and self._type == other._type
            and self.index == other.index
        )

    def __hash__(self):
        return hash((self._type, self.index))

    def __repr__(self):
        return f"{self._type}:{self.index}"

    def __enter__(self):
        self.entered.append(True)
        return self

    def __exit__(self, *args):
        return False


class _TorchProxy:
    """Proxy over the real torch letting tests swap per-device namespaces."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, item):
        return getattr(self._real, item)


def _install_wrapper_stub(monkeypatch):
    """Install the six-name plugin.interservice.flagcx_wrapper boundary."""
    wrapper = SimpleNamespace(
        FLAGCXLibrary=lambda path: Mock(name="flagcx-lib"),
        buffer_type=lambda *args: ("buffer", args),
        flagcxComm_t=int,
        flagcxDataTypeEnum=SimpleNamespace(from_torch=lambda dtype: "dtype"),
        flagcxUniqueId=_FakeUniqueId,
        flagcxRedOpTypeEnum=SimpleNamespace(from_torch=lambda op: "redop"),
    )
    monkeypatch.setitem(sys.modules, "plugin.interservice.flagcx_wrapper", wrapper)
    return wrapper


@pytest.fixture
def load_communicator(monkeypatch):
    """Yield a loader executing the real flagcx.py source under stubs.

    Same contract as the sibling file's fixture: the module runs under a
    unique subject name so each load re-executes the import-time vendor
    dispatch, and the ``_wheel_api``/``_amd_api`` child entries are cleaned
    up afterwards.
    """
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.utils",
        SimpleNamespace(StatelessProcessGroup=_StubGroup),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.logger",
        SimpleNamespace(init_logger=lambda name: logging.getLogger(name)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.torch_utils",
        SimpleNamespace(current_stream=Mock()),
    )
    monkeypatch.delenv("GEMS_VENDOR", raising=False)
    monkeypatch.delenv("FLAGCX_PATH", raising=False)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    _install_wrapper_stub(monkeypatch)
    wheel_api = f"{_COMMUNICATOR_SUBJECT}._wheel_api"
    amd_api = f"{_COMMUNICATOR_SUBJECT}._amd_api"
    monkeypatch.delitem(sys.modules, wheel_api, raising=False)
    monkeypatch.delitem(sys.modules, amd_api, raising=False)

    def load():
        spec = importlib.util.spec_from_file_location(
            _COMMUNICATOR_SUBJECT, _FLAGCX_SOURCE
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    yield load
    sys.modules.pop(wheel_api, None)
    sys.modules.pop(amd_api, None)


def _cpu_communicator(module):
    """Build a CPU communicator by ``__new__`` with a mocked native library."""
    comm = module.PyFlagcxCommunicator.__new__(module.PyFlagcxCommunicator)
    comm.disabled = False
    comm.device = torch.device("cpu")
    comm.flagcx = Mock()
    comm.comm = "comm-handle"
    comm.rank = 0
    comm._all_reduce_stream = None
    return comm


def _txda_comm(module):
    """Build a TXDA communicator by ``__new__`` for ``destroy`` tests."""
    comm = module.PyFlagcxCommunicator.__new__(module.PyFlagcxCommunicator)
    comm.device = _FakeDevice("txda:0")
    comm._destroy_lock = threading.Lock()
    comm._destroy_started = False
    comm._destroy_thread = None
    lib = Mock()
    lib.lib = {"flagcxCommAbort": Mock(return_value=0)}
    lib.FLAGCX_CHECK = Mock()
    comm.flagcx = lib
    comm.comm = "comm-handle"
    comm.available = True
    comm.disabled = False
    return comm


def _invoke_collective(name, comm, *, device="cpu", stream="s"):
    """Invoke the named collective with fresh tensors on *device*."""

    def t(*shape):
        return torch.zeros(*shape, device=device)

    if name == "all_gather":
        comm.all_gather(t(4), t(2), stream=stream)
    elif name == "all_gatherv":
        comm.all_gatherv(t(4), t(2), [2, 2], stream=stream)
    elif name == "reduce_scatter":
        comm.reduce_scatter(t(2), t(4), stream=stream)
    elif name == "reduce_scatterv":
        comm.reduce_scatterv(t(2), t(4), [2, 2], stream=stream)
    elif name == "send":
        comm.send(t(3), 1, stream=stream)
    elif name == "recv":
        comm.recv(t(3), 2, stream=stream)
    elif name == "broadcast":
        comm.broadcast(t(3), 0, stream=stream)
    else:  # pragma: no cover - exhaustiveness guard
        raise AssertionError(f"unknown collective {name}")


@contextlib.contextmanager
def _patched_cuda_device(monkeypatch):
    """Make torch.cuda.device a nullcontext factory (CPU builds raise)."""

    def fake_device(device):
        return contextlib.nullcontext()

    monkeypatch.setattr(torch.cuda, "device", fake_device, raising=False)
    yield


# ---------------------------------------------------------------------------
# flagcx.py: loader guard branches
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
def test_wheel_loader_unresolvable_path_raises(load_communicator, tmp_path):
    """A path importlib cannot turn into a spec raises the ImportError."""
    module = load_communicator()
    bogus = tmp_path / "unknown.xyzw"
    bogus.write_text("not a python module\n", encoding="utf-8")
    with pytest.raises(ImportError, match="Cannot load FlagCX API"):
        module._load_flagcx_wheel_bindings(bogus)
    assert f"{_COMMUNICATOR_SUBJECT}._wheel_api" not in sys.modules


@pytest.mark.supplementary
def test_amd_loader_unresolvable_path_raises(load_communicator, tmp_path):
    """The AMD loader raises the same ImportError for an unresolvable path
    and leaves no private sys.modules entry behind."""
    module = load_communicator()
    bogus = tmp_path / "unknown.xyzw"
    bogus.write_text("not a python module\n", encoding="utf-8")
    with pytest.raises(ImportError, match="Cannot load FlagCX API"):
        module._load_amd_flagcx_api(bogus)
    assert f"{_COMMUNICATOR_SUBJECT}._amd_api" not in sys.modules


@pytest.mark.supplementary
def test_kunlunxin_loader_directory_without_wrapper_falls_back(
    load_communicator, monkeypatch, tmp_path
):
    """A real directory is appended to sys.path; a failing
    plugin.interservice import falls through to the flagcx package."""
    module = load_communicator()
    root = tmp_path / "kx-no-wrapper"
    root.mkdir()

    api_mod = types.ModuleType("flagcx.api")
    api_mod.MARKER = "package-fallback"
    pkg = types.ModuleType("flagcx")
    pkg.api = api_mod
    monkeypatch.setitem(sys.modules, "flagcx", pkg)
    monkeypatch.setitem(sys.modules, "flagcx.api", api_mod)

    api = module._load_kunlunxin_flagcx_api(str(root))
    assert api.MARKER == "package-fallback"
    assert str(root) in sys.path  # the directory was appended for the wrapper


# ---------------------------------------------------------------------------
# flagcx.py: PyFlagcxCommunicator.__init__ branches
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
def test_init_rank0_legacy_api_reads_contents(load_communicator, monkeypatch):
    """Rank 0 with a legacy-API library keeps the ``.contents`` object of the
    unique id and inits the comm through ``ctypes.pointer``."""
    module = load_communicator()
    # spec= pins the attribute set: `handler` present, `devHandle` absent,
    # which is exactly the legacy unique-id API detection.
    lib = Mock(
        spec=[
            "handler",
            "flagcxGetUniqueId",
            "flagcxCommInitRank",
            "flagcxAllReduce",
            "adaptor_stream_copy",
            "adaptor_stream_free",
        ]
    )
    unique = _FakeUniqueId()
    lib.flagcxGetUniqueId.return_value = SimpleNamespace(contents=unique)
    lib.flagcxCommInitRank.return_value = "comm-legacy-r0"

    monkeypatch.setattr(module, "FLAGCXLibrary", lambda path: lib)
    monkeypatch.setattr(module, "_flagcx_wheel_library", None)
    monkeypatch.setenv("FLAGCX_PATH", "")
    monkeypatch.setattr(module, "current_stream", Mock(return_value=Mock()))

    with _patched_cuda_device(monkeypatch):
        comm = module.PyFlagcxCommunicator(_StubGroup(), "cpu")
    # Rank 0 (the _StubGroup default) stored the .contents object itself.
    assert comm.unique_id is unique
    init_args = lib.flagcxCommInitRank.call_args.args
    assert (init_args[0], init_args[2]) == (2, 0)
    # Legacy init: ctypes.pointer wraps the unique id for the native call
    # (.contents re-wraps the same memory, so compare the bytes, not ids).
    assert isinstance(init_args[1], ctypes._Pointer)
    assert list(init_args[1].contents.internal) == list(unique.internal)


@pytest.mark.supplementary
def test_init_musa_device_enters_musa_context(load_communicator, monkeypatch):
    """A musa device inits the comm inside the torch.musa.device context."""
    module = load_communicator()
    lib = Mock()
    lib.flagcxGetUniqueId.return_value = _FakeUniqueId()
    lib.flagcxCommInitRank.return_value = "comm-musa"
    monkeypatch.setattr(module, "FLAGCXLibrary", lambda path: lib)
    monkeypatch.setattr(module, "_flagcx_wheel_library", None)
    monkeypatch.setenv("FLAGCX_PATH", "")

    entered = []

    class _MusaCtx:
        def __enter__(self):
            entered.append(True)
            return self

        def __exit__(self, *args):
            return False

    proxy = _TorchProxy(torch)
    object.__setattr__(proxy, "musa", SimpleNamespace(device=lambda device: _MusaCtx()))
    # torch.device rejects "musa", so the constructor's string conversion goes
    # through the _FakeDevice double like the sibling file's npu/ptpu tests.
    object.__setattr__(proxy, "device", _FakeDevice)
    # The warmup tensor is meta so the real collective short-circuits.
    object.__setattr__(
        proxy, "zeros", lambda *args, **kwargs: torch.zeros(1, device="meta")
    )
    monkeypatch.setattr(module, "torch", proxy, raising=False)
    monkeypatch.setattr(module, "current_stream", Mock(return_value=Mock()))

    comm = module.PyFlagcxCommunicator(_StubGroup(), "musa:0")
    assert comm.device.type == "musa"
    assert entered == [True]  # the musa device context was entered
    assert comm.comm == "comm-musa"
    assert lib.flagcxAllReduce.called is False  # meta warmup never reached it


# ---------------------------------------------------------------------------
# flagcx.py: destroy() corner branches
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
def test_destroy_thread_start_failure_logged(load_communicator, monkeypatch, caplog):
    """A failing thread.start() logs the exception, tears the state down and
    returns without joining."""
    module = load_communicator()
    comm = _txda_comm(module)

    class _DeadThread:
        def start(self):
            raise RuntimeError("cannot start thread")

    monkeypatch.setattr(
        module, "threading", SimpleNamespace(Thread=lambda *a, **k: _DeadThread())
    )
    with caplog.at_level(logging.ERROR, logger=module.logger.name):
        comm.destroy()  # must not raise
    assert any(
        "Could not start TXDA FlagCX communicator abort" in r.message
        for r in caplog.records
    )
    assert comm.available is False
    assert comm.disabled is True
    assert comm.comm is None


@pytest.mark.supplementary
def test_destroy_slow_abort_thread_warns(load_communicator, monkeypatch, caplog):
    """An abort thread still alive after the join timeout logs the warning;
    the communicator state is torn down regardless."""
    module = load_communicator()
    comm = _txda_comm(module)

    done = threading.Event()

    def slow_abort(code):
        time.sleep(0.6)  # longer than the 0.2s join timeout below
        done.set()

    comm.flagcx.lib["flagcxCommAbort"] = slow_abort
    proxy = _TorchProxy(torch)
    object.__setattr__(proxy, "txda", SimpleNamespace(device=lambda device: device))
    object.__setattr__(proxy, "device", _FakeDevice)
    monkeypatch.setattr(module, "torch", proxy)
    module._TXDA_ABORT_TIMEOUT_SECONDS = 0.2

    with caplog.at_level(logging.WARNING, logger=module.logger.name):
        comm.destroy()
    assert any("did not finish within" in r.message for r in caplog.records)
    assert comm.comm is None
    assert comm.available is False
    assert done.wait(timeout=5)  # the daemon thread completed eventually


# ---------------------------------------------------------------------------
# flagcx.py: _current_stream on an AMD-source module
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
def test_current_stream_amd_source_uses_torch_cuda(load_communicator, monkeypatch):
    """An AMD-source module reads torch.cuda.current_stream for the device,
    bypassing the (possibly stale) vLLM stream cache."""
    module = load_communicator()
    monkeypatch.setattr(module, "_flagcx_amd_source", True)
    comm = _cpu_communicator(module)

    seen = []
    cuda_stream = SimpleNamespace(name="amd-cuda-stream")

    def fake_current_stream(device):
        seen.append(device)
        return cuda_stream

    proxy = _TorchProxy(torch)
    object.__setattr__(
        proxy, "cuda", SimpleNamespace(current_stream=fake_current_stream)
    )
    monkeypatch.setattr(module, "torch", proxy)

    assert comm._current_stream() is cuda_stream
    assert seen == [comm.device]


# ---------------------------------------------------------------------------
# flagcx.py: collective guard branches
# ---------------------------------------------------------------------------


@pytest.mark.supplementary
@pytest.mark.parametrize("name", _STREAM_COLLECTIVES)
def test_collective_without_stream_uses_current_stream(
    name, load_communicator, monkeypatch
):
    """Without a stream argument the collective resolves the current stream
    through ``_current_stream()`` and still reaches the library."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    stream_mock = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(module, "current_stream", stream_mock)

    _invoke_collective(name, comm, stream=None)

    assert stream_mock.called
    assert comm.flagcx.mock_calls  # the native library was reached


@pytest.mark.supplementary
@pytest.mark.parametrize("name", _GUARDED_COLLECTIVES)
def test_collective_disabled_returns_before_library(load_communicator, name):
    """A disabled communicator returns before the device assert and any
    library call (valid shapes prove the early exit, not a later failure)."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    comm.disabled = True

    _invoke_collective(name, comm, stream="s")

    assert comm.flagcx.mock_calls == []


@pytest.mark.supplementary
@pytest.mark.parametrize("name", _GUARDED_COLLECTIVES)
def test_collective_trace_tensor_returns_before_library(load_communicator, name):
    """Meta (tracing) tensors never reach the native library; the return
    happens before the group/collective machinery runs."""
    module = load_communicator()
    comm = _cpu_communicator(module)
    # The trace check follows the device assert in these collectives, so the
    # communicator's device must match the meta tensors.
    comm.device = torch.device("meta")

    _invoke_collective(name, comm, device="meta", stream="s")

    assert comm.flagcx.mock_calls == []
