# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU-only lifecycle tests for the installed TXDA FlagCX C ABI."""

import ast
import ctypes
import gc
import logging
import threading
import time
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[3]


class Device:
    def __init__(self, name):
        self.type, _, index = name.partition(":")
        self.index = int(index or 0)

    def __str__(self):
        return f"{self.type}:{self.index}"


class Group:
    def __init__(self, world_size=2):
        self.rank = 0
        self.world_size = world_size

    def broadcast_obj(self, value, src):
        return value


class NativeLibrary:
    def __init__(self, abort):
        self.abort = abort
        self.lookups = []

    def __getitem__(self, name):
        self.lookups.append(name)
        if name != "flagcxCommAbort":
            raise AssertionError(f"Unexpected native API: {name}")
        if self.abort is None:
            raise AttributeError(name)
        return self.abort


class Library:
    def __init__(self, abort):
        self.lib = NativeLibrary(abort)
        self.flagcxCommDestroy = Mock()
        self.flagcxCommInitRank = Mock(return_value=ctypes.c_void_p(0x123456789))
        self.flagcxGetUniqueId = Mock(return_value=object())

    def FLAGCX_CHECK(self, code):
        if code != 0:
            raise RuntimeError(f"FlagCX error: {code}")


@pytest.fixture
def loaded():
    """Execute the actual class while replacing optional vLLM/SDK imports."""
    path = ROOT / "vllm_fl/distributed/device_communicators/flagcx.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PyFlagcxCommunicator"
    )
    entered = []

    @contextmanager
    def device_context(device):
        entered.append((device, threading.current_thread().name))
        yield

    runtime = SimpleNamespace(
        device=Device,
        txda=SimpleNamespace(device=device_context),
        cuda=SimpleNamespace(device=device_context),
        npu=SimpleNamespace(device=device_context),
        musa=SimpleNamespace(device=device_context),
        zeros=Mock(return_value=object()),
    )
    namespace = {
        "ctypes": ctypes,
        "threading": threading,
        "torch": runtime,
        "dist": Mock(),
        "StatelessProcessGroup": Group,
        "ReduceOp": SimpleNamespace(SUM="sum"),
        "FLAGCXLibrary": Mock(),
        "flagcxComm_t": ctypes.c_void_p,
        "logger": logging.getLogger("test.txda.flagcx.cleanup"),
        "_TXDA_ABORT_TIMEOUT_SECONDS": 0.03,
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
        namespace=namespace,
        entered=entered,
        torch=runtime,
    )


def ready_comm(loaded, library, device="txda:3"):
    comm = loaded.cls.__new__(loaded.cls)
    comm.device = Device(device)
    comm.flagcx = library
    comm.comm = ctypes.c_void_p(0x123456789)
    comm.available = True
    comm.disabled = False
    comm._destroy_lock = threading.Lock()
    comm._destroy_started = False
    comm._destroy_thread = None
    return comm


def test_world_size_one_does_not_bind_or_abort(loaded):
    comm = loaded.cls(Group(world_size=1), "txda:0", library_path="unused")
    comm.destroy()
    comm.destroy()
    assert comm.comm is None
    assert comm.disabled and not comm.available
    loaded.namespace["FLAGCXLibrary"].assert_not_called()
    assert loaded.entered == []


def test_missing_library_does_not_abort(loaded):
    loaded.namespace["FLAGCXLibrary"].side_effect = RuntimeError("SDK absent")
    comm = loaded.cls(Group(), "txda:0", library_path="unused")
    comm.destroy()
    assert isinstance(comm.init_error, RuntimeError)
    assert comm.comm is None
    assert comm.disabled and not comm.available
    assert loaded.entered == []


def test_native_init_failure_has_no_returned_handle_to_abort(loaded):
    library = Library(Mock(return_value=0))
    library.flagcxCommInitRank.side_effect = RuntimeError("init failed")
    loaded.namespace["FLAGCXLibrary"].return_value = library
    comm = loaded.cls.__new__(loaded.cls)
    with pytest.raises(RuntimeError, match="init failed"):
        comm.__init__(Group(), "txda:0", library_path="unused")
    comm.destroy()
    assert comm.comm is None
    assert library.lib.lookups == []


def test_warmup_failure_aborts_created_txda_handle(loaded, monkeypatch):
    abort = Mock(return_value=0)
    library = Library(abort)
    loaded.namespace["FLAGCXLibrary"].return_value = library
    monkeypatch.setattr(loaded.cls, "_current_stream", lambda self: Mock())
    monkeypatch.setattr(
        loaded.cls, "all_reduce", Mock(side_effect=RuntimeError("warmup"))
    )
    comm = loaded.cls.__new__(loaded.cls)
    with pytest.raises(RuntimeError, match="warmup"):
        comm.__init__(Group(), "txda:0", library_path="unused")
    assert abort.call_count == 1
    assert comm.disabled and not comm.available and comm.comm is None
    library.flagcxCommDestroy.assert_not_called()


def test_abort_uses_exact_c_abi_and_device_once(loaded):
    received = []
    callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)(
        lambda handle: received.append(handle) or 0
    )
    library = Library(callback)
    comm = ready_comm(loaded, library)
    device = comm.device
    comm.destroy()
    comm.destroy()
    assert received == [0x123456789]
    assert callback.argtypes == [ctypes.c_void_p]
    assert callback.restype is ctypes.c_int
    assert library.lib.lookups == ["flagcxCommAbort"]
    assert loaded.entered == [(device, "txda-flagcx-abort")]
    assert comm._destroy_thread.daemon
    assert comm.disabled and not comm.available and comm.comm is None
    assert comm.all_reduce(object()) is None
    library.flagcxCommDestroy.assert_not_called()


def test_initialized_but_temporarily_disabled_comm_is_still_aborted(loaded):
    abort = Mock(return_value=0)
    comm = ready_comm(loaded, Library(abort))
    comm.disabled = True
    comm.destroy()
    abort.assert_called_once()


def test_timeout_keeps_native_inputs_alive_without_blocking_or_retry(loaded, caplog):
    started, release = threading.Event(), threading.Event()

    def abort(handle):
        started.set()
        assert release.wait(2)
        return 0

    native_abort = Mock(side_effect=abort)
    library = Library(native_abort)
    library_ref = weakref.ref(library)
    comm = ready_comm(loaded, library)
    handle_ref = weakref.ref(comm.comm)
    with caplog.at_level(logging.WARNING):
        before = time.monotonic()
        comm.destroy()
        elapsed = time.monotonic() - before
    thread = comm._destroy_thread
    try:
        assert started.is_set() and thread.is_alive()
        assert elapsed < 0.5
        assert "cleanup is incomplete" in caplog.text
        assert "Do not reinitialize" in caplog.text
        comm.destroy()
        del library, comm
        gc.collect()
        assert library_ref() is not None
        assert handle_ref() is not None
        assert native_abort.call_count == 1
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    gc.collect()
    assert library_ref() is None


def test_concurrent_destroy_starts_one_abort(loaded):
    started, release = threading.Event(), threading.Event()

    def abort(handle):
        started.set()
        assert release.wait(2)
        return 0

    native_abort = Mock(side_effect=abort)
    comm = ready_comm(loaded, Library(native_abort))
    callers = [threading.Thread(target=comm.destroy) for _ in range(8)]
    try:
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join(1)
        assert all(not caller.is_alive() for caller in callers)
        assert started.is_set() and native_abort.call_count == 1
    finally:
        release.set()
        if comm._destroy_thread is not None:
            comm._destroy_thread.join(2)


@pytest.mark.parametrize("failure", ["return_code", "missing_symbol", "exception"])
def test_abort_failure_is_logged_and_never_retried(loaded, caplog, failure):
    abort = None if failure == "missing_symbol" else Mock(return_value=7)
    if failure == "exception":
        abort.side_effect = RuntimeError("native failure")
    library = Library(abort)
    comm = ready_comm(loaded, library)
    with caplog.at_level(logging.ERROR):
        comm.destroy()
        comm.destroy()
    assert "abort failed" in caplog.text
    assert "cleanup is incomplete" in caplog.text
    if failure == "return_code":
        assert "FlagCX error: 7" in caplog.text
    assert library.lib.lookups == ["flagcxCommAbort"]
    library.flagcxCommDestroy.assert_not_called()
    assert comm.disabled and not comm.available


@pytest.mark.parametrize("device", ["cuda:0", "npu:0", "musa:0", "ptpu:0", "cpu"])
def test_other_backends_keep_existing_cleanup_behavior(loaded, device):
    library = Library(Mock(return_value=0))
    comm = ready_comm(loaded, library, device)
    handle = comm.comm
    comm.destroy()
    assert comm.comm is handle
    assert comm.available and not comm.disabled
    assert library.lib.lookups == []
    assert loaded.entered == []


@pytest.mark.parametrize("device", ["txda:0", "cuda:0", "npu:0", "musa:0", "ptpu:0"])
def test_outer_communicator_delegates_only_txda_and_preserves_all2all(device):
    path = ROOT / "vllm_fl/distributed/communicator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "destroy"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    child, manager = Mock(), Mock()
    comm = SimpleNamespace(
        device=Device(device), pyflagcx_comm=child, all2all_manager=manager
    )
    namespace["destroy"](comm)
    namespace["destroy"](comm)
    assert child.destroy.call_count == int(device.startswith("txda"))
    manager.destroy.assert_called_once_with()
    assert comm.pyflagcx_comm is None and comm.all2all_manager is None


def test_failure_to_start_abort_thread_is_logged_without_retry(
    loaded, caplog, monkeypatch
):
    library = Library(Mock(return_value=0))
    comm = ready_comm(loaded, library)
    thread = Mock()
    thread.start.side_effect = RuntimeError("cannot start thread")
    factory = Mock(return_value=thread)
    monkeypatch.setitem(loaded.namespace, "threading", SimpleNamespace(Thread=factory))
    with caplog.at_level(logging.ERROR):
        comm.destroy()
        comm.destroy()
    assert "Could not start TXDA FlagCX" in caplog.text
    assert "cleanup is incomplete" in caplog.text
    factory.assert_called_once()
    thread.join.assert_not_called()
    assert library.lib.lookups == []
    assert comm.disabled and not comm.available


@pytest.mark.parametrize("device", ["cuda:0", "npu:0", "musa:0"])
def test_other_backend_warmup_failure_does_not_abort(loaded, monkeypatch, device):
    library = Library(Mock(return_value=0))
    loaded.namespace["FLAGCXLibrary"].return_value = library
    monkeypatch.setattr(loaded.cls, "_current_stream", lambda self: Mock())
    monkeypatch.setattr(
        loaded.cls, "all_reduce", Mock(side_effect=RuntimeError("warmup"))
    )
    comm = loaded.cls.__new__(loaded.cls)
    with pytest.raises(RuntimeError, match="warmup"):
        comm.__init__(Group(), device, library_path="unused")
    assert library.lib.lookups == []
    assert comm.comm is library.flagcxCommInitRank.return_value
    assert comm.available and not comm.disabled


def test_warmup_stream_synchronization_failure_aborts_handle(loaded, monkeypatch):
    abort = Mock(return_value=0)
    library = Library(abort)
    loaded.namespace["FLAGCXLibrary"].return_value = library
    stream = Mock()
    stream.synchronize.side_effect = RuntimeError(
        "warmup device synchronization failed"
    )
    monkeypatch.setattr(loaded.cls, "_current_stream", lambda self: stream)
    all_reduce = Mock()
    monkeypatch.setattr(loaded.cls, "all_reduce", all_reduce)
    comm = loaded.cls.__new__(loaded.cls)
    with pytest.raises(RuntimeError, match="warmup device synchronization failed"):
        comm.__init__(Group(), "txda:0", library_path="unused")
    all_reduce.assert_called_once()
    stream.synchronize.assert_called_once()
    abort.assert_called_once()
    assert comm.disabled and not comm.available and comm.comm is None
    library.flagcxCommDestroy.assert_not_called()
