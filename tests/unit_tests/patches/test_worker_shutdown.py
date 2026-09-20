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

"""CPU process regressions; these run in every platform's unit-test job."""

import ast
import logging
import multiprocessing
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from vllm_fl.patches import worker_shutdown as shutdown


def _config(request_timeout=0, *, dp=1, nodes=1):
    return SimpleNamespace(
        shutdown_timeout=request_timeout,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp, nnodes_within_dp=nodes, numa_bind=False
        ),
    )


@pytest.fixture
def fake_modules(monkeypatch):
    class Executor:
        def _init_executor(self):
            pass

        def shutdown(self):
            pass

        @staticmethod
        def _ensure_worker_termination(processes):
            raise AssertionError("original reaper must not run")

    class Engine:
        @staticmethod
        def run_engine_core(*args, **kwargs):
            return args, kwargs

    class Manager:
        def __init__(self, a, b, c, vllm_config, local, address, executor_class):
            assert (
                self._vllm_fl_request_shutdown_timeout == vllm_config.shutdown_timeout
            )
            self.shutdown_calls = []

        def shutdown(self, timeout=None):
            self.shutdown_calls.append(timeout)

    modules = SimpleNamespace(
        utils=SimpleNamespace(CoreEngineProcManager=Manager),
        core=SimpleNamespace(EngineCoreProc=Engine),
        executor=SimpleNamespace(MultiprocExecutor=Executor),
    )
    imports = {
        "vllm.v1.engine.utils": modules.utils,
        "vllm.v1.engine.core": modules.core,
        "vllm.v1.executor.multiproc_executor": modules.executor,
    }
    monkeypatch.setattr(shutdown, "import_module", imports.__getitem__)
    monkeypatch.setattr(shutdown.sys, "platform", "linux")
    return modules


@pytest.mark.parametrize(
    ("request_timeout", "process_timeout", "expected"),
    [(0, 0, 15.0), (0, None, None), (0, 7, 7), (7, None, None), (7, 0, 0)],
)
def test_timeout_semantics(request_timeout, process_timeout, expected):
    assert (
        shutdown.get_engine_process_shutdown_timeout(request_timeout, process_timeout)
        == expected
    )


@pytest.mark.parametrize(
    "version",
    ["0.20.2", "0.20.2+vendor", "0.20.2.dev0", "0.20.3.dev0+gbc150f502.d20260601"],
)
def test_patch_is_idempotent_and_target_is_pickleable(fake_modules, version):
    modules = fake_modules
    assert shutdown.patch_worker_shutdown(
        modules.utils, modules.core, vllm_version=version
    )
    first_init = modules.utils.CoreEngineProcManager.__init__
    assert not shutdown.patch_worker_shutdown(
        modules.utils, modules.core, vllm_version=version
    )
    assert modules.utils.CoreEngineProcManager.__init__ is first_init
    target = modules.core.EngineCoreProc.run_engine_core
    assert target is shutdown.run_engine_core
    assert pickle.loads(pickle.dumps(target)) is target
    assert shutdown.patch_multiproc_worker_termination(modules.executor)
    assert not shutdown.patch_multiproc_worker_termination(modules.executor)


@pytest.mark.parametrize(
    "version",
    ["0.20.1", "0.20.3", "0.20.3.dev0", "0.20.3.dev0+g12345678", "0.24.0", "unknown"],
)
def test_patch_rejects_unreviewed_versions(fake_modules, version):
    modules = fake_modules
    original = modules.core.EngineCoreProc.run_engine_core
    assert not shutdown.patch_worker_shutdown(
        modules.utils, modules.core, vllm_version=version
    )
    assert modules.core.EngineCoreProc.run_engine_core is original


def test_ci_vllm_version_is_covered(fake_modules):
    # A green vendor job must not silently test an unsupported, unpatched build.
    from vllm.version import __version__

    assert shutdown.patch_worker_shutdown(
        fake_modules.utils, fake_modules.core, vllm_version=__version__
    ), f"The shutdown backport must be reviewed for this CI image: {__version__}"


@pytest.mark.parametrize(
    ("request_timeout", "process_timeout", "expected"),
    [(0, 0, 15.0), (0, None, None), (0, 23, 23), (12, 0, 0), (12, None, None)],
)
def test_manager_preserves_request_drain_semantics(
    fake_modules, request_timeout, process_timeout, expected
):
    modules = fake_modules
    shutdown.patch_worker_shutdown(modules.utils, modules.core, vllm_version="0.20.2")
    manager = modules.utils.CoreEngineProcManager(
        1,
        0,
        0,
        _config(request_timeout),
        True,
        "unused",
        modules.executor.MultiprocExecutor,
    )
    manager.shutdown(timeout=process_timeout)
    assert manager.shutdown_calls == [expected]


@pytest.mark.parametrize(
    ("dp", "nodes", "is_mp"), [(2, 1, True), (1, 2, True), (1, 1, False)]
)
def test_manager_does_not_change_other_executors(fake_modules, dp, nodes, is_mp):
    modules = fake_modules
    shutdown.patch_worker_shutdown(modules.utils, modules.core, vllm_version="0.20.2")
    executor = modules.executor.MultiprocExecutor if is_mp else object
    manager = modules.utils.CoreEngineProcManager(
        a=1,
        b=0,
        c=0,
        vllm_config=_config(dp=dp, nodes=nodes),
        local=True,
        address="unused",
        executor_class=executor,
    )
    manager.shutdown(timeout=0)
    assert manager.shutdown_calls == [0]


def test_other_executor_entrypoint_delegates(fake_modules):
    modules = fake_modules
    shutdown.patch_worker_shutdown(modules.utils, modules.core, vllm_version="0.20.2")
    config = _config(dp=2)
    args, kwargs = shutdown.run_engine_core(
        "sentinel",
        vllm_config=config,
        executor_class=object,
        dp_rank=3,
        local_dp_rank=2,
    )
    assert args == ("sentinel",)
    assert kwargs == dict(
        vllm_config=config, executor_class=object, dp_rank=3, local_dp_rank=2
    )


def _original_entrypoint(namespace):
    """Execute the installed upstream function with a CPU-only fake EngineCore.

    Read its AST to avoid importing GPU model/worker modules. This is upstream's
    actual control flow, not a hand-written approximation of the signal bug.
    """
    import vllm

    source = Path(vllm.__file__).parent / "v1" / "engine" / "core.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    engine = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "EngineCoreProc"
    )
    function = next(
        n
        for n in engine.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_engine_core"
    )
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["run_engine_core"]


def _signal_child(connection, patched, phase):
    """Run a real process, pausing deterministically inside cleanup."""
    os.setsid()

    def cleanup_barrier():
        connection.send("cleanup")
        if not connection.poll(10):
            raise RuntimeError("parent did not release cleanup barrier")
        assert connection.recv() == "finish"

    class Engine:
        def __init__(self, *args, **kwargs):
            self.input_queue = SimpleNamespace(put_nowait=lambda value: None)
            self.shutdown_state = None

        def run_busy_loop(self):
            connection.send("ready")
            deadline = time.monotonic() + 10
            while self.shutdown_state != "requested":
                if time.monotonic() >= deadline:
                    raise RuntimeError("shutdown signal was not delivered")
                time.sleep(0.01)

        def shutdown(self):
            if phase == "worker":
                cleanup_barrier()
            connection.send("reclaimed")

        def _send_engine_dead(self):
            connection.send("unexpected-engine-failure")

    class Callback:
        def __init__(self, callback):
            self.callback = callback

        def trigger(self):
            self.callback()

        def stop(self):
            if phase == "callback":
                cleanup_barrier()

    class Executor:
        _ensure_worker_termination = staticmethod(lambda processes: None)

        def _init_executor(self):
            pass

        def shutdown(self):
            pass

    core = SimpleNamespace(
        EngineCoreProc=Engine,
        SignalCallback=Callback,
        EngineCoreRequestType=SimpleNamespace(WAKEUP="wakeup"),
        EngineShutdownState=SimpleNamespace(REQUESTED="requested"),
        maybe_register_config_serialize_by_value=lambda: None,
        set_process_title=lambda title: None,
        maybe_init_worker_tracer=lambda *args: None,
        decorate_logs=lambda: None,
        logger=logging.getLogger("shutdown-test"),
    )
    modules = {
        "vllm.v1.engine.core": core,
        "vllm.v1.executor.multiproc_executor": SimpleNamespace(
            MultiprocExecutor=Executor
        ),
    }
    shutdown.import_module = modules.__getitem__
    if patched:
        # The spawn case imports and unpickles the actual public entrypoint.
        target = pickle.loads(pickle.dumps(shutdown.run_engine_core))
    else:
        target = _original_entrypoint({**vars(core), "signal": signal})
    target(vllm_config=_config(), executor_class=Executor)
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
    assert signal.getsignal(signal.SIGINT) == signal.SIG_DFL
    connection.send("finished")
    connection.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process signal semantics")
@pytest.mark.parametrize("phase", ["callback", "worker"])
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("patched", [False, True], ids=["upstream", "patched"])
def test_repeated_signal_during_cleanup(patched, signum, phase):
    _check_signal_case(patched, signum, phase, start_method="fork")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process signal semantics")
def test_spawn_entrypoint_handles_process_group_ctrl_c():
    _check_signal_case(True, signal.SIGINT, "worker", start_method="spawn")


def _check_signal_case(patched, signum, phase, *, start_method):
    context = multiprocessing.get_context(start_method)
    parent, child = context.Pipe()
    process = context.Process(target=_signal_child, args=(child, patched, phase))
    process.start()
    child.close()
    try:
        assert parent.poll(30), "EngineCore did not start"
        assert parent.recv() == "ready"
        # SIGINT to the private foreground group models terminal Ctrl+C.
        send = (
            (lambda: os.killpg(process.pid, signum))
            if signum == signal.SIGINT
            else (lambda: os.kill(process.pid, signum))
        )
        send()
        assert parent.poll(10), "EngineCore never entered cleanup"
        assert parent.recv() == "cleanup"
        send()
        if patched:
            parent.send("finish")
            assert parent.poll(10)
            assert parent.recv() == "reclaimed"
            assert parent.poll(10)
            assert parent.recv() == "finished"
        process.join(10)
        assert not process.is_alive()
        assert process.exitcode == (0 if patched else -signum)
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)
        parent.close()
        process.close()


def _stubborn_worker(connection):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    connection.send("ready")
    connection.recv()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process signal semantics")
def test_owned_workers_are_killed_reaped_and_unrelated_process_is_untouched(
    monkeypatch,
):
    context = multiprocessing.get_context("fork")
    processes = []
    connections = []
    monkeypatch.setattr(shutdown, "WORKER_GRACE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(shutdown, "WORKER_TERMINATE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(shutdown, "WORKER_KILL_TIMEOUT_S", 2.0)
    try:
        for _ in range(3):
            parent, child = context.Pipe()
            process = context.Process(target=_stubborn_worker, args=(child,))
            process.start()
            child.close()
            processes.append(process)
            connections.append(parent)
            assert parent.poll(5)
            assert parent.recv() == "ready"
        owned, unrelated = processes[:2], processes[2]
        shutdown.ensure_worker_termination(owned)
        for process in owned:
            assert process.exitcode == -signal.SIGKILL
            assert not process.is_alive()
            assert not psutil.pid_exists(process.pid), "worker was not reaped"
        assert unrelated.is_alive(), "another server's process must not be signalled"
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(5)
            process.close()
        for connection in connections:
            connection.close()


def test_unstarted_worker_is_safe():
    process = multiprocessing.get_context("spawn").Process()
    shutdown.ensure_worker_termination([process])
    assert process.pid is None
    process.close()


def test_already_exited_worker_is_still_joined():
    joins = []
    worker = SimpleNamespace(
        pid=123,
        join=lambda timeout: joins.append(timeout),
        is_alive=lambda: False,
    )
    shutdown.ensure_worker_termination([worker])
    assert len(joins) == 1


def test_worker_wait_uses_one_deadline(monkeypatch):
    clock = [0.0]
    joins = []

    class Worker:
        pid = 123

        def join(self, timeout):
            joins.append(timeout)
            clock[0] += timeout

        def is_alive(self):
            return True

    monkeypatch.setattr(shutdown.time, "monotonic", lambda: clock[0])
    assert not shutdown._join_workers([Worker(), Worker(), Worker()], 4.0)
    assert joins == [4.0, 0.0, 0.0]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process signal semantics")
def test_serving_helper_fails_before_its_own_recovery_cleanup():
    from tests.e2e_tests.serving.server_helper import VllmServer

    command = [
        sys.executable,
        "-u",
        "-c",
        (
            "import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); "
            "print('ready'); time.sleep(60)"
        ),
    ]
    processes = []
    try:
        for _ in range(2):
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            processes.append(process)
            assert process.stdout.readline().strip() == "ready"
        owned, unrelated = processes
        server = VllmServer(model="unused", shutdown_timeout=0.05)
        server._process = owned
        with pytest.raises(pytest.fail.Exception, match="before test cleanup"):
            server.stop()
        assert owned.poll() == -signal.SIGKILL
        assert unrelated.poll() is None
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            process.stdout.close()


def test_serving_helper_reaps_api_that_exits_after_poll(monkeypatch):
    from tests.e2e_tests.serving.server_helper import VllmServer

    class ExitingAPI:
        pid = 123
        returncode = None

        def __init__(self):
            self.waits = []

        def poll(self):
            # The API exits immediately after this snapshot; the subsequent
            # process scan sees a zombie and correctly reports no live process.
            return None

        def wait(self, timeout):
            self.waits.append(timeout)
            self.returncode = 0
            return 0

    server = VllmServer(model="unused")
    process = ExitingAPI()
    server._process = process
    monkeypatch.setattr(server, "_record_processes", lambda: None)
    monkeypatch.setattr(server, "_live_processes", lambda: [])
    server.stop()
    assert process.waits == [1]
    assert process.returncode == 0
    assert server._process is None


@pytest.mark.parametrize("patched", [False, True], ids=["upstream", "patched"])
@pytest.mark.parametrize("owner_raises", [False, True], ids=["success", "exception"])
def test_monitor_first_shutdown_waits_for_cleanup_owner(patched, owner_raises):
    entered = threading.Event()
    release = threading.Event()
    waiter_entered = threading.Event()
    waiter_returned = threading.Event()
    errors = []

    class Executor:
        _ensure_worker_termination = staticmethod(lambda processes: None)

        def __init__(self):
            self._init_executor()

        def _init_executor(self):
            self.shutting_down = False
            self.cleanup_calls = 0

        def shutdown(self):
            # The upstream boolean is set before its blocking worker join.
            if not self.shutting_down:
                self.shutting_down = True
                self.cleanup_calls += 1
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("cleanup owner was not released")
                if owner_raises:
                    raise RuntimeError("cleanup owner failed")

    if patched:
        shutdown.patch_multiproc_worker_termination(
            SimpleNamespace(MultiprocExecutor=Executor)
        )
    executor = Executor()

    def call_shutdown(*, waiter=False):
        if waiter:
            waiter_entered.set()
        try:
            executor.shutdown()
        except Exception as exc:
            errors.append(exc)
        finally:
            if waiter:
                waiter_returned.set()

    monitor = threading.Thread(target=call_shutdown, daemon=True)
    waiter = threading.Thread(
        target=call_shutdown, kwargs={"waiter": True}, daemon=True
    )
    try:
        monitor.start()
        assert entered.wait(2)
        waiter.start()
        assert waiter_entered.wait(2)
        if patched:
            assert not waiter_returned.wait(0.1), (
                "EngineCore skipped in-progress cleanup"
            )
        else:
            assert waiter_returned.wait(2), "baseline race was not reproduced"
    finally:
        release.set()
        monitor.join(2)
        if waiter.ident is not None:
            waiter.join(2)
    assert not monitor.is_alive()
    assert not waiter.is_alive()
    assert waiter_returned.is_set()
    assert executor.cleanup_calls == 1
    expected_errors = ["cleanup owner failed"] if owner_raises else []
    if owner_raises and patched:
        expected_errors.append("Previous worker shutdown failed")
        propagated = next(error for error in errors if error.__cause__ is not None)
        assert str(propagated.__cause__) == "cleanup owner failed"
    assert sorted(str(error) for error in errors) == sorted(expected_errors)


def test_executor_shutdown_is_reentrant_on_the_owner_thread():
    class Executor:
        _ensure_worker_termination = staticmethod(lambda processes: None)

        def __init__(self):
            self._init_executor()

        def _init_executor(self):
            self.shutting_down = False
            self.cleanup_calls = 0

        def shutdown(self):
            if not self.shutting_down:
                self.shutting_down = True
                self.cleanup_calls += 1
                self.shutdown()

    shutdown.patch_multiproc_worker_termination(
        SimpleNamespace(MultiprocExecutor=Executor)
    )
    executor = Executor()
    thread = threading.Thread(target=executor.shutdown, daemon=True)
    thread.start()
    thread.join(2)
    assert not thread.is_alive(), "same-thread shutdown reentry deadlocked"
    assert executor.cleanup_calls == 1
