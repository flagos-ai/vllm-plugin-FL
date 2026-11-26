# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for the remaining ``vllm_fl/dispatch/__init__.py`` lines.

``tests/unit_tests/dispatch/test_call_op.py`` (which must not be modified)
covers the convenience ``call_op``/``resolve_op`` functions, the basic
``CachedOp`` warmup + torch.compile path, the WeakSet registry semantics and
the AMD in-place-failure no-retry behaviour through the real manager.  Still
uncovered after that file plus the supplementary dispatch tests run:

* ``CachedOp.warmup`` without an explicit manager (line 199: default-manager
  resolution inside warmup);
* ``CachedOp.__call__``'s AMD compile-ready branch (212-220): the
  ``torch.compiler.is_compiling()`` early impl call, and the RuntimeError
  when the warmed impl was dropped;
* ``__call__``'s opt-out branches: ``VLLM_FL_OP_FAST_PATH=0`` (225) and
  IO-dump enabled (228);
* the post-failure ``_use_manager_call`` fast route (242);
* the failure retry tail (265-267): marking the impl failed and falling
  back to ``mgr.call`` for retryable ops;
* ``warmup_cached_ops`` without an explicit manager (273) and its
  ``amd_compile`` flagging plus the optional-op ``RuntimeError`` skip
  (281-286).

The real OpManager/registry stack is used (registered test impls), so every
assertion runs against the genuine dispatch pipeline; only the AMD compile
branch stubs ``torch.compiler.is_compiling`` because a real Dynamo trace
cannot run inside these tests on a CPU-only box.
"""

import os
import sys
from types import SimpleNamespace

import pytest

from vllm_fl.dispatch import (
    BackendImplKind,
    CachedOp,
    OpImpl,
    get_default_manager,
    io_dumper as dispatch_io_dumper,
    reset_default_manager,
    reset_global_policy,
    warmup_cached_ops,
)


@pytest.fixture(autouse=True)
def _reset_dispatch_state():
    """Reset the global manager/policy around every test (core pattern)."""
    reset_default_manager()
    reset_global_policy()
    yield
    reset_default_manager()
    reset_global_policy()


@pytest.fixture(autouse=True)
def _fast_path_default(monkeypatch):
    """Keep the fast path enabled and dumping disabled unless a test says so."""
    monkeypatch.delenv("VLLM_FL_OP_FAST_PATH", raising=False)
    monkeypatch.setattr(dispatch_io_dumper, "_enabled", False)
    yield


def _register_op(op_name, fn, impl_id="default.test"):
    """Register *fn* for *op_name* in the real default manager."""
    manager = get_default_manager()
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    impl = OpImpl(
        op_name=op_name,
        impl_id=impl_id,
        kind=BackendImplKind.DEFAULT,
        fn=fn,
    )
    manager.registry.register_impl(impl)
    return manager, impl


class TestCachedOpWarmupDefaultManager:
    """``CachedOp.warmup`` with no explicit manager argument (line 199)."""

    def test_warmup_without_manager_uses_default(self):
        """warmup() resolves through get_default_manager and caches the impl,
        so a later call goes straight to the impl function."""
        marker = SimpleNamespace()

        def impl_fn(x):
            marker.hits = getattr(marker, "hits", 0) + 1
            return x * 21

        manager, _ = _register_op("warmup_default_op", impl_fn)
        assert manager.policy_epoch >= 0

        cached = CachedOp("warmup_default_op")
        cached.warmup()  # no mgr argument: default-manager resolution
        assert cached._impl is not None
        assert cached._impl.fn is impl_fn

        assert cached(2) == 42
        # The cached call did not re-resolve: same impl object, one hit only
        # through the direct fn invocation.
        assert marker.hits == 1


class TestCachedOpAmdCompileBranch:
    """``CachedOp.__call__``'s ``_amd_compile_ready`` branch (212-220)."""

    def test_compiling_calls_impl_directly(self, monkeypatch):
        """With _amd_compile_ready and torch.compiler.is_compiling() True the
        call bypasses the manager and runs the warmed impl."""
        calls = []

        def impl_fn(x):
            calls.append(x)
            return x + 100

        manager, _ = _register_op("amd_compile_op", impl_fn)
        cached = CachedOp("amd_compile_op")
        warmup_cached_ops(manager, amd_compile=True)
        assert cached._amd_compile_ready is True

        torch_mod = sys.modules["torch"]
        monkeypatch.setattr(
            torch_mod.compiler, "is_compiling", lambda: True, raising=False
        )
        assert cached(5) == 105
        assert calls == [5]

    def test_compiling_without_impl_raises(self, monkeypatch):
        """A compile-ready op whose impl was dropped raises the RuntimeError
        instead of silently returning a wrong result."""
        manager, _ = _register_op("amd_lost_impl", lambda x: x)
        cached = CachedOp("amd_lost_impl")
        cached.warmup(manager)
        cached._amd_compile_ready = True
        cached._impl = None  # simulate the impl being dropped

        torch_mod = sys.modules["torch"]
        monkeypatch.setattr(
            torch_mod.compiler, "is_compiling", lambda: True, raising=False
        )
        with pytest.raises(RuntimeError, match="lost its warmed AMD"):
            cached(1)

    def test_not_compiling_uses_normal_path(self, monkeypatch):
        """_amd_compile_ready alone is not enough outside a compile region."""
        manager, _ = _register_op("amd_nocompile_op", lambda x: x * 2)
        cached = CachedOp("amd_nocompile_op")
        warmup_cached_ops(manager, amd_compile=True)
        torch_mod = sys.modules["torch"]
        monkeypatch.setattr(
            torch_mod.compiler, "is_compiling", lambda: False, raising=False
        )
        assert cached(3) == 6


class TestCachedOpOptOutBranches:
    """``__call__``'s fast-path opt-out branches (lines 225, 228)."""

    def test_fast_path_disabled_routes_through_manager(self, monkeypatch):
        """VLLM_FL_OP_FAST_PATH=0 routes every call through mgr.call."""
        dispatch = sys.modules["vllm_fl.dispatch"]
        manager = get_default_manager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        routed = []

        real_call = manager.call

        def tracking_call(op_name, *args, **kwargs):
            routed.append(op_name)
            return real_call(op_name, *args, **kwargs)

        monkeypatch.setattr(manager, "call", tracking_call)
        monkeypatch.setattr(dispatch, "_OP_FAST_PATH_ENABLED", False)

        def impl_fn(x):
            return x * 3

        manager.registry.register_impl(
            OpImpl(
                op_name="fastpath_off_op",
                impl_id="default.off",
                kind=BackendImplKind.DEFAULT,
                fn=impl_fn,
            )
        )
        cached = CachedOp("fastpath_off_op")
        cached.warmup(manager)
        assert cached(4) == 12
        assert routed == ["fastpath_off_op"]

    def test_dump_enabled_routes_through_manager(self, monkeypatch):
        """An enabled IO dump forces the manager route so dumping hooks run."""
        dispatch = sys.modules["vllm_fl.dispatch"]
        manager = get_default_manager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        routed = []

        real_call = manager.call

        def tracking_call(op_name, *args, **kwargs):
            routed.append(op_name)
            return real_call(op_name, *args, **kwargs)

        monkeypatch.setattr(manager, "call", tracking_call)
        # Fast path stays enabled; only the dump flag flips.
        monkeypatch.setattr(dispatch, "_OP_FAST_PATH_ENABLED", True)
        monkeypatch.setattr(dispatch_io_dumper, "_enabled", True)

        def impl_fn(x):
            return x + 1

        manager.registry.register_impl(
            OpImpl(
                op_name="dump_on_op",
                impl_id="default.dump",
                kind=BackendImplKind.DEFAULT,
                fn=impl_fn,
            )
        )
        cached = CachedOp("dump_on_op")
        cached.warmup(manager)
        assert cached(9) == 10
        assert routed == ["dump_on_op"]


class TestCachedOpFailureRetry:
    """The failure tail of ``CachedOp.__call__`` (lines 242, 265-267)."""

    def test_retryable_failure_marks_impl_and_falls_back(self):
        """A failing retryable impl is marked failed and the fallback runs
        through mgr.call in the same invocation."""
        attempts = []

        def failing_fn(x):
            attempts.append("primary")
            raise RuntimeError("primary exploded")

        def fallback_fn(x):
            attempts.append("fallback")
            return x * 5

        manager = get_default_manager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        manager.registry.register_impl(
            OpImpl(
                op_name="retry_me",
                impl_id="default.failing",
                kind=BackendImplKind.DEFAULT,
                fn=failing_fn,
            )
        )
        manager.registry.register_impl(
            OpImpl(
                op_name="retry_me",
                impl_id="reference.fallback",
                kind=BackendImplKind.REFERENCE,
                fn=fallback_fn,
                priority=100,
            )
        )

        cached = CachedOp("retry_me")
        cached.warmup(manager)
        assert cached._use_manager_call is False

        # First call: primary fails, retry is allowed -> fallback returns.
        assert cached(3) == 15
        assert attempts == ["primary", "fallback"]
        # The cached op flipped to the manager route after the failure.
        assert cached._use_manager_call is True
        assert cached._impl is None

        # Second call takes the _use_manager_call fast route (line 242).
        assert cached(3) == 15
        assert attempts == ["primary", "fallback", "fallback"]

    def test_strict_policy_failure_propagates(self):
        """A strict policy disables retrying: the failure propagates."""
        from vllm_fl.dispatch import SelectionPolicy, set_global_policy

        def failing_fn(x):
            raise RuntimeError("no retry")

        manager, _ = _register_op("strict_fail", failing_fn)
        set_global_policy(SelectionPolicy(strict=True))
        cached = CachedOp("strict_fail")
        cached.warmup(manager)
        with pytest.raises(RuntimeError, match="no retry"):
            cached(1)
        assert cached._use_manager_call is False


class TestWarmupCachedOpsBranches:
    """``warmup_cached_ops`` default manager + amd flag + skip (273-286)."""

    def test_default_manager_resolution_and_amd_flag(self, monkeypatch):
        """Without a manager argument the default manager initializes; with
        amd_compile the flag only sticks while the fast path is on and
        dumping is off."""
        dispatch = sys.modules["vllm_fl.dispatch"]

        def impl_fn(x):
            return x

        manager = get_default_manager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        manager.registry.register_impl(
            OpImpl(
                op_name="warmup_flag_op",
                impl_id="default.flag",
                kind=BackendImplKind.DEFAULT,
                fn=impl_fn,
            )
        )
        cached = CachedOp("warmup_flag_op")

        # amd_compile=True + fast path on + dump off -> flag armed (281-283).
        warmup_cached_ops(amd_compile=True)
        assert cached._impl is not None
        assert cached._amd_compile_ready is True

        # amd_compile with the fast path disabled leaves the flag off.
        cached2 = CachedOp("warmup_flag_op")
        monkeypatch.setattr(dispatch, "_OP_FAST_PATH_ENABLED", False)
        warmup_cached_ops(amd_compile=True)
        assert cached2._amd_compile_ready is False

    def test_optional_op_runtime_error_is_skipped(self, monkeypatch):
        """A CachedOp whose warmup raises RuntimeError is skipped without
        aborting the sweep (lines 284-286)."""
        import weakref

        dispatch = sys.modules["vllm_fl.dispatch"]
        cached_ops = weakref.WeakSet()
        monkeypatch.setattr(dispatch, "_CACHED_OPS", cached_ops)

        good_calls = []

        class _FakeManager:
            policy_epoch = 1

            def ensure_initialized(self):
                pass

            def _resolve_impl(self, op_name):
                if op_name == "unavailable_op":
                    raise RuntimeError("op unavailable on this platform")
                return SimpleNamespace(
                    op_name=op_name, impl_id=f"default.{op_name}", fn=None
                )

            def _record_first_use(self, op_name, impl):
                good_calls.append(op_name)

        good = CachedOp("good_op")
        bad = CachedOp("unavailable_op")
        warmup_cached_ops(_FakeManager())
        assert good_calls == ["good_op"]
        assert good._impl is not None
        # The failed op is skipped but stays registered for a later retry.
        assert bad._impl is None
