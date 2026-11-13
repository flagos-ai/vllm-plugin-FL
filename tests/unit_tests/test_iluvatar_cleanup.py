"""Regression tests for Iluvatar autotune and worker cleanup contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_fl.dispatch.backends.vendor.iluvatar import shutdown
from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
    _release_benchmark_tensors,
)


def test_release_benchmark_tensors_clears_diagnostic_references_on_success():
    calls = []

    def run(self, *args, **kwargs):
        calls.append((args, kwargs))
        return "selected"

    wrapped = _release_benchmark_tensors(run)
    tuner = SimpleNamespace(
        _last_benchmark_args=(object(),),
        _last_benchmark_meta={"activation": object()},
        nargs={"x": object()},
    )
    assert wrapped(tuner, 1, x=2) == "selected"
    assert calls == [((1,), {"x": 2})]
    assert tuner._last_benchmark_args == ()
    assert tuner._last_benchmark_meta == {}
    assert tuner.nargs is None


def test_release_benchmark_tensors_clears_diagnostic_references_on_error():
    def run(self, *args, **kwargs):
        raise ValueError("kernel failure")

    tuner = SimpleNamespace(
        _last_benchmark_args=(object(),),
        _last_benchmark_meta={"x": object()},
        nargs={"x": object()},
    )
    with pytest.raises(ValueError, match="kernel failure"):
        _release_benchmark_tensors(run)(tuner)
    assert (tuner._last_benchmark_args, tuner._last_benchmark_meta, tuner.nargs) == (
        (),
        {},
        None,
    )


def test_release_benchmark_tensors_is_idempotent():
    def run(self):
        return 3

    first = _release_benchmark_tensors(run)
    assert _release_benchmark_tensors(first) is first
    assert first(SimpleNamespace()) == 3


def test_snapshot_descendants_deduplicates_and_ignores_unavailable(monkeypatch):
    child = shutdown.psutil.Process()
    p1 = SimpleNamespace(pid=1)
    p2 = SimpleNamespace(pid=2)
    process = Mock()
    process.children.side_effect = [[child], shutdown.psutil.NoSuchProcess(2), [child]]
    monkeypatch.setattr(shutdown.psutil, "Process", lambda pid: process)
    assert shutdown._snapshot_descendants([p1, p2, p1]) == [child]


def test_cleanup_escalates_remaining_workers(monkeypatch):
    worker = Mock(pid=7)
    worker.is_running.side_effect = [True, True, False]
    worker.status.return_value = "running"
    monkeypatch.setattr(shutdown, "WORKER_CLEANUP_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(
        shutdown,
        "_wait_for_workers",
        lambda processes, timeout: [worker] if worker.is_running() else [],
    )
    shutdown._cleanup_workers([worker])
    worker.terminate.assert_called_once_with()
    worker.kill.assert_called_once_with()


def test_release_drops_activations_but_preserves_cache_across_calls():
    import gc
    import weakref

    import torch

    def run(self, x):
        self._last_benchmark_args = (x,)
        self._last_benchmark_meta = {"x": x}
        self.nargs = {"x": x}
        self.cache[tuple(x.shape)] = "selected-config"
        return x + 1

    tuner = SimpleNamespace(cache={})
    wrapped = _release_benchmark_tensors(run)
    for size in (2, 5, 2):
        activation = torch.ones(size)
        reference = weakref.ref(activation)
        result = wrapped(tuner, activation)
        assert torch.equal(result, torch.full((size,), 2.0))
        del activation
        gc.collect()
        assert reference() is None
    assert tuner.cache == {(2,): "selected-config", (5,): "selected-config"}


@pytest.mark.parametrize("fail", [False, True])
def test_shutdown_cleanup_preserves_return_or_original_exception(monkeypatch, fail):
    descendants = [object()]
    cleanup = Mock()
    monkeypatch.setattr(shutdown, "_snapshot_descendants", lambda _: descendants)
    monkeypatch.setattr(shutdown, "_cleanup_workers", cleanup)

    def original(processes, timeout=None):
        assert timeout == 7
        if fail:
            raise ValueError("upstream failure")
        return "finished"

    wrapped = shutdown._with_worker_cleanup(original)
    assert shutdown._with_worker_cleanup(wrapped) is wrapped
    if fail:
        with pytest.raises(ValueError, match="upstream failure"):
            wrapped([], timeout=7)
    else:
        assert wrapped([], timeout=7) == "finished"
    cleanup.assert_called_once_with(descendants)


@pytest.mark.timeout(20)
@pytest.mark.parametrize("ignore_term", [False, True])
def test_shutdown_cleans_owned_process_tree(tmp_path, monkeypatch, ignore_term):
    import subprocess
    import sys

    child_code = (
        "import signal,time; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + "print('ready', flush=True); time.sleep(60)"
    )
    script = tmp_path / "owned_worker_tree.py"
    script.write_text(
        "import subprocess,sys,time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "stdout=subprocess.PIPE, text=True)\n"
        "assert child.stdout.readline().strip() == 'ready'\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, text=True
    )
    child = None
    try:
        child = shutdown.psutil.Process(int(parent.stdout.readline()))
        descendants = shutdown._snapshot_descendants([parent])
        assert [p.pid for p in descendants] == [child.pid]
        monkeypatch.setattr(shutdown, "WORKER_CLEANUP_TIMEOUT_SECONDS", 0)

        def original(processes, timeout=None):
            for process in processes:
                process.terminate()
                process.wait(timeout=3)
            return "upstream-complete"

        assert shutdown._with_worker_cleanup(original)([parent]) == "upstream-complete"
        assert shutdown._live_processes([child]) == []
        assert parent.poll() is not None
    finally:
        if child is not None and shutdown._live_processes([child]):
            child.kill()
            shutdown.psutil.wait_procs([child], timeout=3)
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=3)
        parent.stdout.close()


@pytest.mark.parametrize("cleanup", [False, True])
def test_provider_cached_run_activation_retention(cleanup):
    """Exercise the installed provider's run body, not a copy of its state writes."""
    import gc
    import inspect
    import weakref

    import torch
    import triton
    from flag_gems.utils.libentry import LibTuner

    config = triton.Config({"BLOCK": 16})
    tuner = object.__new__(LibTuner)
    tuner.arg_names = ["x"]
    tuner.configs = [config, triton.Config({"BLOCK": 32})]
    tuner.cache = {(2,): config, (5,): config}
    tuner.get_key = lambda args: tuple(args["x"].shape)
    tuner.get_benchmark_key = tuner.get_key
    tuner.fn = SimpleNamespace(run=lambda x, **kwargs: x + 1)
    run = inspect.unwrap(LibTuner.run)
    if cleanup:
        run = _release_benchmark_tensors(run)
    for size in (2, 5, 2):
        activation = torch.ones(size)
        reference = weakref.ref(activation)
        result = run(tuner, activation)
        assert torch.equal(result, torch.full((size,), 2.0))
        del activation, result
        gc.collect()
        assert (reference() is None) is cleanup
        assert tuner.best_config is config
        assert tuner.cache == {(2,): config, (5,): config}
