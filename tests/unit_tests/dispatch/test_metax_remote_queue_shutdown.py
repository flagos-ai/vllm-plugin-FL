"""Cancellation and timeout contracts for the remote vLLM queue bridge."""

import importlib.util
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl/dispatch/backends/vendor/metax/patches/remote_queue_shutdown.py"
)
spec = importlib.util.spec_from_file_location("metax_remote_queue_test", SOURCE)
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


def _receive(state, timeout=None, indefinite=False):
    state.calls.append((timeout, indefinite))
    try:
        return state.messages.get(timeout=timeout)
    except queue.Empty as error:
        raise TimeoutError from error


def _remote():
    return SimpleNamespace(
        _is_remote_reader=True, shutting_down=False, messages=queue.Queue(), calls=[]
    )


def test_remote_shutdown_wakes_indefinite_reader():
    state = _remote()
    result = []

    def receive():
        try:
            compat._with_remote_shutdown(_receive)(state, indefinite=True)
        except Exception as error:
            result.append(error)

    thread = threading.Thread(target=receive, daemon=True)
    thread.start()
    while not state.calls:
        time.sleep(0.001)
    state.shutting_down = True
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert len(result) == 1
    assert isinstance(result[0], RuntimeError)
    assert str(result[0]) == "cancelled"
    assert all(indefinite for _, indefinite in state.calls)


@pytest.mark.parametrize("timeout", [0, -1, 0.025, 0.25])
def test_remote_timeout_keeps_total_deadline(timeout):
    state = _remote()
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        compat._with_remote_shutdown(_receive)(state, timeout=timeout)
    elapsed = time.monotonic() - start
    assert max(timeout, 0) <= elapsed < max(timeout, 0) + 1
    assert all(0 <= interval <= 0.1 for interval, _ in state.calls)
    assert len(state.calls) >= (3 if timeout == 0.25 else 1)


def test_remote_ready_message_does_not_wait_for_poll_interval():
    state = _remote()
    item = object()
    state.messages.put(item)
    assert compat._with_remote_shutdown(_receive)(state, timeout=0) is item
    assert state.calls == [(0, False)]


def test_local_queue_keeps_original_call_and_exception():
    state = SimpleNamespace(_is_remote_reader=False)
    error = RuntimeError("local cancellation")
    calls = []

    def original(actual, **kwargs):
        calls.append((actual, kwargs))
        raise error

    with pytest.raises(RuntimeError) as caught:
        compat._with_remote_shutdown(original)(state, timeout=123, indefinite=True)
    assert caught.value is error
    assert calls == [(state, {"timeout": 123, "indefinite": True})]


@pytest.mark.parametrize(
    "vendor,version,installed",
    [
        ("metax", "0.32.0", True),
        ("metax", "0.32.0+empty", True),
        ("metax", "0.33.0", False),
        ("metax", "0.32.1", False),
        ("nvidia", "0.32.0", False),
    ],
)
def test_install_only_affected_metax_version_and_is_idempotent(
    monkeypatch, vendor, version, installed
):
    original = lambda *args, **kwargs: None
    message_queue = SimpleNamespace(dequeue=original)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__=version))
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.shm_broadcast",
        SimpleNamespace(MessageQueue=message_queue),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        SimpleNamespace(current_platform=SimpleNamespace(vendor_name=vendor)),
    )
    compat.install_remote_queue_shutdown()
    first = message_queue.dequeue
    assert (first is not original) is installed
    compat.install_remote_queue_shutdown()
    assert message_queue.dequeue is first
