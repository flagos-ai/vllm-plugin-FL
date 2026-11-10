import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

MODULE_PATH = (
    Path(__file__).parents[2] / "vllm_fl/dispatch/backends/vendor/iluvatar/shutdown.py"
)
spec = importlib.util.spec_from_file_location("iluvatar_shutdown", MODULE_PATH)
shutdown = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shutdown)


def test_snapshot_preserves_workers_before_core_exits(monkeypatch):
    worker = Mock(pid=42)
    core = Mock()
    core.children.return_value = [worker]
    monkeypatch.setattr(shutdown.psutil, "Process", Mock(return_value=core))
    cleanup = Mock()
    monkeypatch.setattr(shutdown, "_cleanup_workers", cleanup)
    events = []

    def terminate(processes, timeout=None):
        events.append("terminate")
        assert core.children.called
        assert timeout == 0
        return "done"

    wrapped = shutdown._with_worker_cleanup(terminate)
    assert wrapped([SimpleNamespace(pid=10)], timeout=0) == "done"
    cleanup.assert_called_once_with([worker])
    core.children.assert_called_once_with(recursive=True)
    assert shutdown._with_worker_cleanup(wrapped) is wrapped


def test_install_worker_cleanup_is_idempotent(monkeypatch):
    import sys
    from types import ModuleType

    engine_utils = ModuleType("vllm.v1.engine.utils")

    def original_shutdown(processes, timeout=None):
        return None

    engine_utils.shutdown = original_shutdown
    engine = ModuleType("vllm.v1.engine")
    engine.utils = engine_utils
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", engine)
    shutdown.install_worker_cleanup()
    installed = engine_utils.shutdown
    assert installed is not original_shutdown
    shutdown.install_worker_cleanup()
    assert engine_utils.shutdown is installed
    assert installed.__wrapped__ is original_shutdown


def test_exception_still_cleans_workers(monkeypatch):
    monkeypatch.setattr(shutdown, "_snapshot_descendants", lambda processes: [42])
    cleanup = Mock()
    monkeypatch.setattr(shutdown, "_cleanup_workers", cleanup)

    def fail(processes, timeout=None):
        raise ValueError("upstream failed")

    with pytest.raises(ValueError, match="upstream failed"):
        shutdown._with_worker_cleanup(fail)([])
    cleanup.assert_called_once_with([42])


def test_cleanup_escalates_retained_worker(monkeypatch):
    worker = Mock(pid=42)
    wait = Mock(side_effect=[[worker], [worker], []])
    monkeypatch.setattr(shutdown, "_wait_for_workers", wait)
    shutdown._cleanup_workers([worker])
    worker.terminate.assert_called_once()
    worker.kill.assert_called_once()
    assert [call.args[1] for call in wait.call_args_list] == [30.0, 4.0, 5.0]


def test_cleanup_does_not_signal_exited_worker(monkeypatch):
    worker = Mock(pid=42)
    monkeypatch.setattr(shutdown, "_wait_for_workers", lambda processes, timeout: [])
    shutdown._cleanup_workers([worker])
    worker.terminate.assert_not_called()
    worker.kill.assert_not_called()


def test_zombies_do_not_hold_device_context():
    zombie = Mock()
    zombie.status.return_value = psutil.STATUS_ZOMBIE
    gone = Mock()
    gone.is_running.return_value = False
    assert shutdown._live_processes([zombie, gone]) == []


def test_orphaned_worker_is_reaped_after_supervisor_exits(monkeypatch, tmp_path):
    import subprocess
    import sys

    supervisor_script = tmp_path / "supervisor.py"
    supervisor_script.write_text(
        "import subprocess, sys, time\n"
        "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "print(worker.pid, flush=True)\n"
        "time.sleep(120)\n"
    )
    supervisor = subprocess.Popen(
        [sys.executable, str(supervisor_script)], stdout=subprocess.PIPE, text=True
    )
    worker_pid = int(supervisor.stdout.readline())
    worker = psutil.Process(worker_pid)
    monkeypatch.setattr(shutdown, "WORKER_CLEANUP_TIMEOUT_SECONDS", 0.05)

    def stop_supervisor(processes, timeout=None):
        for process in processes:
            process.terminate()
            process.wait(timeout=5)

    try:
        shutdown._with_worker_cleanup(stop_supervisor)([supervisor], timeout=0)
        assert supervisor.poll() is not None
        assert not shutdown._live_processes([worker])
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.wait(timeout=5)
        if shutdown._live_processes([worker]):
            worker.kill()
        supervisor.stdout.close()


def test_snapshot_handles_missing_and_unstarted_supervisors(monkeypatch):
    process_lookup = Mock(side_effect=psutil.NoSuchProcess(10))
    monkeypatch.setattr(shutdown.psutil, "Process", process_lookup)
    assert (
        shutdown._snapshot_descendants(
            [SimpleNamespace(pid=None), SimpleNamespace(pid=10)]
        )
        == []
    )
    process_lookup.assert_called_once_with(10)


def test_cleanup_reports_unkillable_worker(monkeypatch):
    worker = Mock(pid=42)
    monkeypatch.setattr(
        shutdown, "_wait_for_workers", lambda processes, timeout: [worker]
    )
    with pytest.raises(RuntimeError, match="42"):
        shutdown._cleanup_workers([worker])


@pytest.mark.parametrize(
    "error_type", [psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess]
)
@pytest.mark.parametrize("failure_site", ["lookup", "children"])
def test_snapshot_errors_do_not_prevent_shutdown(monkeypatch, error_type, failure_site):
    inaccessible = Mock()
    error = error_type(10)
    if failure_site == "children":
        inaccessible.children.side_effect = error
        first = inaccessible
    else:
        first = error
    worker = Mock(pid=42)
    accessible = Mock()
    accessible.children.return_value = [worker]
    monkeypatch.setattr(
        shutdown.psutil, "Process", Mock(side_effect=[first, accessible])
    )
    cleanup = Mock()
    monkeypatch.setattr(shutdown, "_cleanup_workers", cleanup)
    calls = []

    def original(processes, timeout=None):
        calls.append((processes, timeout))
        return "shutdown completed"

    processes = [SimpleNamespace(pid=10), SimpleNamespace(pid=20)]
    assert (
        shutdown._with_worker_cleanup(original)(processes, timeout=3)
        == "shutdown completed"
    )
    assert calls == [(processes, 3)]
    cleanup.assert_called_once_with([worker])
