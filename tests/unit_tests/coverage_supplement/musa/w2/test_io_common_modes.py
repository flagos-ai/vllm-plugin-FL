# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for io_common mode management and IO-active lifecycle.

Covers ModeManager (LIFO enter/exit/drain), managed_inference_mode and the
independent re-entrancy guards created by make_guard().
"""

import threading

import pytest
import torch

from vllm_fl.dispatch import io_common as c


class _FakeMode:
    """Minimal TorchDispatchMode/TorchFunctionMode stand-in."""

    def __init__(self, log=None, name="m"):
        self.entered = 0
        self.exited = 0
        self.exit_args = []
        self._log = log
        self._name = name

    def __enter__(self):
        self.entered += 1
        if self._log is not None:
            self._log.append(("enter", self._name))
        return self

    def __exit__(self, *exc):
        self.exited += 1
        self.exit_args.append(exc)
        if self._log is not None:
            self._log.append(("exit", self._name))
        return False


class TestIoActiveFlag:
    def test_default_is_false(self):
        c.set_io_active(False)
        assert c.is_io_active() is False

    def test_set_true_and_back(self):
        try:
            c.set_io_active(True)
            assert c.is_io_active() is True
            assert c._io_active is True
        finally:
            c.set_io_active(False)
        assert c.is_io_active() is False
        assert c._io_active is False


class TestModeManager:
    def test_initial_state(self):
        mgr = c.ModeManager()
        assert mgr.is_entered("x") is False
        assert mgr._stack == []
        assert mgr._pending == set()

    def test_enter_enters_and_records(self):
        mgr = c.ModeManager()
        mode = _FakeMode()
        mgr.enter("dump", mode)
        assert mode.entered == 1
        assert mgr.is_entered("dump") is True
        assert mgr._stack == [("dump", mode)]

    def test_enter_is_idempotent_by_name(self):
        mgr = c.ModeManager()
        first = _FakeMode()
        second = _FakeMode()
        mgr.enter("dump", first)
        mgr.enter("dump", second)
        assert first.entered == 1
        assert second.entered == 0
        assert len(mgr._stack) == 1
        assert mgr._stack[0][1] is first

    def test_enter_multiple_names(self):
        mgr = c.ModeManager()
        a, b = _FakeMode(name="a"), _FakeMode(name="b")
        mgr.enter("a", a)
        mgr.enter("b", b)
        assert [n for n, _ in mgr._stack] == ["a", "b"]
        assert a.entered == 1 and b.entered == 1

    def test_request_exit_unknown_name_is_noop(self):
        mgr = c.ModeManager()
        mgr.request_exit("missing")
        assert mgr._pending == set()
        assert mgr._stack == []

    def test_request_exit_top_exits_immediately(self):
        mgr = c.ModeManager()
        mode = _FakeMode()
        mgr.enter("a", mode)
        mgr.request_exit("a")
        assert mode.exited == 1
        assert mgr._stack == []
        assert mgr._pending == set()
        assert mgr.is_entered("a") is False

    def test_request_exit_non_top_stays_pending(self):
        mgr = c.ModeManager()
        bottom, top = _FakeMode(name="bottom"), _FakeMode(name="top")
        mgr.enter("bottom", bottom)
        mgr.enter("top", top)

        mgr.request_exit("bottom")
        assert mgr._pending == {"bottom"}
        assert bottom.exited == 0
        assert top.exited == 0
        assert mgr.is_entered("bottom") is True

    def test_request_exit_drains_from_top_once_unblocked(self):
        mgr = c.ModeManager()
        bottom, top = _FakeMode(name="bottom"), _FakeMode(name="top")
        mgr.enter("bottom", bottom)
        mgr.enter("top", top)

        mgr.request_exit("bottom")
        assert bottom.exited == 0

        mgr.request_exit("top")
        # Exiting top unblocks bottom, which also had a pending exit.
        assert top.exited == 1
        assert bottom.exited == 1
        assert mgr._stack == []
        assert mgr._pending == set()

    def test_request_exit_drains_chain_of_pending(self):
        mgr = c.ModeManager()
        modes = [_FakeMode(name=str(i)) for i in range(4)]
        for i, m in enumerate(modes):
            mgr.enter(str(i), m)
        # Pending exits for everything except the topmost.
        for i in range(3):
            mgr.request_exit(str(i))
        assert all(m.exited == 0 for m in modes)
        mgr.request_exit("3")
        assert [m.exited for m in modes] == [1, 1, 1, 1]
        assert mgr._stack == []

    def test_exit_all_reverse_lifo(self):
        log = []
        mgr = c.ModeManager()
        a, b, c_ = _FakeMode(log, "a"), _FakeMode(log, "b"), _FakeMode(log, "c")
        for name, m in (("a", a), ("b", b), ("c", c_)):
            mgr.enter(name, m)
        mgr.exit_all()
        assert log == [
            ("enter", "a"),
            ("enter", "b"),
            ("enter", "c"),
            ("exit", "c"),
            ("exit", "b"),
            ("exit", "a"),
        ]
        assert all(m.exited == 1 for m in (a, b, c_))
        assert mgr._stack == []
        assert mgr._pending == set()

    def test_exit_all_clears_pending_even_without_stack(self):
        mgr = c.ModeManager()
        mgr._pending.add("ghost")
        mgr.exit_all()
        assert mgr._pending == set()

    def test_exit_all_on_empty_is_noop(self):
        mgr = c.ModeManager()
        mgr.exit_all()
        assert mgr._stack == []

    def test_exit_passes_none_tuple(self):
        mgr = c.ModeManager()
        mode = _FakeMode()
        mgr.enter("a", mode)
        mgr.request_exit("a")
        assert mode.exit_args == [(None, None, None)]

    def test_module_level_managers_exist(self):
        assert isinstance(c.dispatch_mode_mgr, c.ModeManager)
        assert isinstance(c.func_mode_mgr, c.ModeManager)
        assert c.dispatch_mode_mgr is not c.func_mode_mgr

    def test_concurrent_enter_is_single(self):
        mgr = c.ModeManager()
        modes = [_FakeMode() for _ in range(8)]
        threads = [threading.Thread(target=mgr.enter, args=("shared", m)) for m in modes]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(mgr._stack) == 1
        assert sum(m.entered for m in modes) == 1


class TestManagedInferenceMode:
    def teardown_method(self):
        c.set_io_active(False)

    def test_without_io_uses_inference_mode(self):
        c.set_io_active(False)

        @c.managed_inference_mode()
        def probe():
            return torch.is_inference_mode_enabled(), torch.is_grad_enabled()

        inference, grad = probe()
        assert inference is True
        assert grad is False

    def test_with_io_uses_no_grad(self):
        c.set_io_active(True)

        @c.managed_inference_mode()
        def probe():
            return torch.is_inference_mode_enabled(), torch.is_grad_enabled()

        inference, grad = probe()
        assert inference is False
        assert grad is False

    def test_passes_through_args_kwargs_and_result(self):
        c.set_io_active(True)

        @c.managed_inference_mode()
        def add(a, b=0, *, scale=1):
            return (a + b) * scale

        assert add(1, 2, scale=3) == 9

    def test_preserves_function_metadata(self):
        @c.managed_inference_mode()
        def documented():
            """docstring marker"""

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "docstring marker"

    def test_propagates_exceptions(self):
        c.set_io_active(False)

        @c.managed_inference_mode()
        def boom():
            raise KeyError("bad")

        with pytest.raises(KeyError, match="bad"):
            boom()


class TestMakeGuard:
    def test_default_inactive(self):
        active, _set = c.make_guard()
        assert active() is False

    def test_set_and_reset(self):
        active, set_guard = c.make_guard()
        set_guard(True)
        assert active() is True
        set_guard(False)
        assert active() is False

    def test_guards_are_independent(self):
        active_a, set_a = c.make_guard()
        active_b, set_b = c.make_guard()
        set_a(True)
        assert active_a() is True
        assert active_b() is False
        set_b(True)
        set_a(False)
        assert active_a() is False
        assert active_b() is True

    def test_guard_is_thread_local(self):
        active, set_guard = c.make_guard()
        seen = {}

        def worker():
            seen["before"] = active()
            set_guard(True)
            seen["after"] = active()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert seen == {"before": False, "after": True}
        # Main thread state is untouched by the worker.
        assert active() is False
