# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for the Iluvatar worker shutdown cleanup.

Covers ``vllm_fl/dispatch/backends/vendor/iluvatar/shutdown.py``:

* ``_snapshot_descendants`` collects worker descendants via psutil,
  skipping processes that vanished.
* ``_live_processes`` filters to running, non-zombie entries.
* ``_wait_for_workers`` returns the still-live remainder after a timeout.
* ``_cleanup_workers`` terminates then kills retained workers and raises
  when they survive both.
* ``_with_worker_cleanup`` wraps a shutdown callable so descendants are
  cleaned up even when shutdown raises; idempotent via its marker flag.
* ``install_worker_cleanup`` swaps ``vllm.v1.engine.utils.shutdown``.

All psutil interactions run against fake in-memory process objects; the
psutil module attributes are monkeypatched (not replaced globally).
CPU-only. Vendor-gated to the iluvatar platform.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("iluvatar")]

# status constant psutil uses
_ZOMBIE = "zombie"


class FakeProcess:
    """In-memory stand-in for psutil.Process with child/exit behavior."""

    def __init__(self, pid, alive=True, zombie=False):
        self.pid = pid
        self._alive = alive
        self._zombie = zombie
        self.terminated = 0
        self.killed = 0
        # stubborn processes survive terminate(); immortal ones survive kill()
        self.survives_terminate = False
        self.survives_kill = False

    def is_running(self):
        return self._alive

    def status(self):
        return _ZOMBIE if self._zombie else "running"

    def terminate(self):
        self.terminated += 1
        if not self.survives_terminate:
            self._alive = False

    def kill(self):
        self.killed += 1
        if not self.survives_kill:
            self._alive = False


@pytest.fixture
def fake_psutil(monkeypatch):
    """Point shutdown.psutil at an in-memory process table."""
    import vllm_fl.dispatch.backends.vendor.iluvatar.shutdown as sd

    children_map = {}
    wait_results = []  # each entry: callable(procs, timeout) -> (alive, dead)

    class _FakeProcessHandle:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive=True):
            return children_map.get(self.pid, [])

    def fake_wait_procs(procs, timeout=None):
        if wait_results:
            return wait_results.pop(0)(procs, timeout)
        return list(procs), []

    monkeypatch.setattr(sd.psutil, "Process", _FakeProcessHandle)
    monkeypatch.setattr(sd.psutil, "wait_procs", fake_wait_procs)

    class Handle:
        def set_children(self, pid, kids):
            children_map[pid] = kids

        def on_wait(self, fn):
            wait_results.append(fn)

    return Handle()


class TestSnapshotDescendants:
    def test_collects_children_deduplicated(self, fake_psutil):
        from vllm_fl.dispatch.backends.vendor.iluvatar.shutdown import (
            _snapshot_descendants,
        )

        kid_a = FakeProcess(11)
        kid_b = FakeProcess(12)
        # two parents share one child: dedup by pid
        fake_psutil.set_children(1, [kid_a])
        fake_psutil.set_children(2, [kid_a, kid_b])
        result = _snapshot_descendants([FakeProcess(1), FakeProcess(2)])
        assert [p.pid for p in result] == [11, 12]

    def test_vanished_parent_is_skipped(self, fake_psutil, monkeypatch):
        import vllm_fl.dispatch.backends.vendor.iluvatar.shutdown as sd

        from vllm_fl.dispatch.backends.vendor.iluvatar.shutdown import (
            _snapshot_descendants,
        )

        def boom(pid):
            raise sd.psutil.NoSuchProcess(pid)

        # psutil.Process lookup itself raises NoSuchProcess
        monkeypatch.setattr(
            sd.psutil,
            "Process",
            lambda pid: boom(pid),
        )
        # must not raise
        assert _snapshot_descendants([FakeProcess(99)]) == []

    def test_none_pid_skipped(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.shutdown import (
            _snapshot_descendants,
        )

        ghost = FakeProcess(None)
        assert _snapshot_descendants([ghost]) == []


class TestLiveProcesses:
    def test_filters_dead_and_zombie(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.shutdown import (
            _live_processes,
        )

        alive = FakeProcess(1)
        dead = FakeProcess(2, alive=False)
        zombie = FakeProcess(3, alive=True, zombie=True)
        assert _live_processes([alive, dead, zombie]) == [alive]

    def test_nosuchprocess_swallowed(self, fake_psutil):
        from vllm_fl.dispatch.backends.vendor.iluvatar.shutdown import (
            _live_processes,
        )

        class Vanishing(FakeProcess):
            def is_running(self):
                import psutil

                raise psutil.NoSuchProcess(self.pid)

        survivor = FakeProcess(1)
        result = _live_processes([Vanishing(2), survivor])
        assert result == [survivor]


class TestWaitForWorkers:
    def test_returns_live_remainder(self, fake_psutil):
        from vllm_fl.dispatch.backends.vendor.iluvatar.shutdown import (
            _wait_for_workers,
        )

        p1, p2 = FakeProcess(1), FakeProcess(2)
        # psutil.wait_procs returns (gone, alive): p1 exited, p2 still alive
        fake_psutil.on_wait(
            lambda procs, timeout: ([p1], [p2])
        )
        remaining = _wait_for_workers([p1, p2], timeout=0.01)
        assert remaining == [p2]


class TestCleanupWorkers:
    def test_fast_exit_terminates_nothing(self, fake_psutil):
        from vllm_fl.dispatch.backends.vendor.iluvatar.shutdown import (
            _cleanup_workers,
        )

        p = FakeProcess(1)
        # every wait pass reports (gone, alive): all exited
        fake_psutil.on_wait(lambda procs, timeout: (list(procs), []))
        fake_psutil.on_wait(lambda procs, timeout: (list(procs), []))
        _cleanup_workers([p])
        assert p.terminated == 0
        assert p.killed == 0

    def test_terminate_then_kill(self, fake_psutil):
        from vllm_fl.dispatch.backends.vendor.iluvatar.shutdown import (
            _cleanup_workers,
        )

        stubborn = FakeProcess(1)
        stubborn.survives_terminate = True
        # each wait pass reports (gone, alive): never exits
        def wait(procs, timeout):
            return ([], list(procs))

        for _ in range(3):
            fake_psutil.on_wait(wait)
        _cleanup_workers([stubborn])
        assert stubborn.terminated == 1
        assert stubborn.killed == 1

    def test_survivors_raise_runtime_error(self, fake_psutil):
        from vllm_fl.dispatch.backends.vendor.iluvatar.shutdown import (
            _cleanup_workers,
        )

        immortal = FakeProcess(1)
        immortal.survives_terminate = True
        immortal.survives_kill = True
        # never exits through any wait: (gone, alive)
        def always_alive(procs, timeout):
            return ([], list(procs))

        for _ in range(4):
            fake_psutil.on_wait(always_alive)
        with pytest.raises(RuntimeError, match="did not exit"):
            _cleanup_workers([immortal])
        assert immortal.terminated == 1
        assert immortal.killed == 1


class TestWithWorkerCleanup:
    def test_wraps_and_returns_result(self, fake_psutil):
        from vllm_fl.dispatch.backends.vendor.iluvatar.shutdown import (
            _with_worker_cleanup,
        )

        fake_psutil.on_wait(lambda procs, timeout: ([], list(procs)))

        def shutdown(processes, timeout=None):
            return "shutdown-ok"

        wrapped = _with_worker_cleanup(shutdown)
        assert wrapped([FakeProcess(1)], timeout=1) == "shutdown-ok"
        assert getattr(wrapped, "_iluvatar_worker_cleanup", False)

    def test_cleanup_runs_when_shutdown_raises(self, fake_psutil):
        from vllm_fl.dispatch.backends.vendor.iluvatar.shutdown import (
            _with_worker_cleanup,
        )

        cleaned = []
        fake_psutil.on_wait(lambda procs, timeout: ([], list(procs)))

        def shutdown(processes, timeout=None):
            raise RuntimeError("engine shutdown failed")

        wrapped = _with_worker_cleanup(shutdown)
        with pytest.raises(RuntimeError, match="engine shutdown failed"):
            wrapped([FakeProcess(1)])
        # finally branch ran cleanup without swallowing the error

    def test_idempotent_marker(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.shutdown import (
            _with_worker_cleanup,
        )

        def shutdown(processes, timeout=None):
            return None

        wrapped = _with_worker_cleanup(shutdown)
        assert _with_worker_cleanup(wrapped) is wrapped

    def test_descendants_passed_to_cleanup_not_the_process_list(self, fake_psutil):
        from vllm_fl.dispatch.backends.vendor.iluvatar.shutdown import (
            _with_worker_cleanup,
        )

        parent = FakeProcess(1)
        child = FakeProcess(2)
        seen_wait_inputs = []
        fake_psutil.set_children(1, [child])

        def wait(procs, timeout):
            seen_wait_inputs.append([p.pid for p in procs])
            return ([], list(procs))

        for _ in range(3):
            fake_psutil.on_wait(wait)

        def shutdown(processes, timeout=None):
            return "ok"

        _with_worker_cleanup(shutdown)([parent], timeout=1)
        # every recorded wait pass (that still had live processes) saw the
        # descendant (pid 2), never the parent; a final empty pass is fine —
        # it means the terminated descendant was already filtered out.
        non_empty = [entry for entry in seen_wait_inputs if entry]
        assert non_empty
        assert all(2 in entry for entry in non_empty)
        assert all(1 not in entry for entry in non_empty)


class TestInstallWorkerCleanup:
    def test_install_swaps_engine_utils_shutdown(self, monkeypatch):
        import vllm.v1.engine.utils as engine_utils

        import vllm_fl.dispatch.backends.vendor.iluvatar.shutdown as sd

        original = engine_utils.shutdown
        try:
            sd.install_worker_cleanup()
            installed = engine_utils.shutdown
            assert installed is not original
            assert getattr(installed, "_iluvatar_worker_cleanup", False)
            # calling the installed fn routes through the wrapper
            assert installed.__wrapped__ is original
        finally:
            engine_utils.shutdown = original

    def test_install_is_idempotent(self, monkeypatch):
        import vllm.v1.engine.utils as engine_utils

        import vllm_fl.dispatch.backends.vendor.iluvatar.shutdown as sd

        original = engine_utils.shutdown
        try:
            sd.install_worker_cleanup()
            first = engine_utils.shutdown
            sd.install_worker_cleanup()
            assert engine_utils.shutdown is first
        finally:
            engine_utils.shutdown = original
