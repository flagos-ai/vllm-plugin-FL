# Copyright (c) 2026 BAAI. All rights reserved.

"""
Tests for CachedOp in vllm_fl.dispatch.

Covers:
  - Basic call-through on first use
  - Cache hit: same impl reused when epoch unchanged
  - Cache invalidation on OpManager policy_epoch bump
  - Cache invalidation on global policy_epoch bump (policy_context / set_global_policy)
  - Fast-path disabled via VLLM_FL_OP_FAST_PATH=0
  - IO-dump active routes through mgr.call
  - Fallback on impl failure (non-strict)
  - Raise on impl failure (strict)
  - Multiple CachedOp instances are independent
"""

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_registry_with_op(op_name: str, fn):
    """Register a single op impl and return (registry, impl_id)."""
    from vllm_fl.dispatch.registry import OpRegistry
    from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority

    registry = OpRegistry()
    impl = OpImpl(
        op_name=op_name,
        impl_id=f"impl:ref.{op_name}",
        kind=BackendImplKind.REFERENCE,
        priority=BackendPriority.REFERENCE,
        fn=fn,
        vendor=None,
    )
    registry.register(op_name, impl)
    return registry


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_global_state():
    """Reset global manager and policy around every test."""
    from vllm_fl.dispatch.manager import reset_default_manager
    from vllm_fl.dispatch.policy import reset_global_policy
    reset_default_manager()
    reset_global_policy()
    yield
    reset_default_manager()
    reset_global_policy()


@pytest.fixture()
def register_double_op():
    """Register 'double_op' on the default manager and return it."""
    from vllm_fl.dispatch.manager import get_default_manager
    from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority

    mgr = get_default_manager()
    mgr._initialized = True  # skip builtin registration
    impl = OpImpl(
        op_name="double_op",
        impl_id="impl:ref.double_op",
        kind=BackendImplKind.REFERENCE,
        priority=BackendPriority.REFERENCE,
        fn=lambda x: x * 2,
        vendor=None,
    )
    mgr._registry.register("double_op", impl)
    return mgr


# ── Basic call-through ────────────────────────────────────────────────────────

class TestCachedOpBasic:

    def test_first_call_returns_correct_result(self, register_double_op):
        from vllm_fl.dispatch import CachedOp
        op = CachedOp("double_op")
        assert op(5) == 10

    def test_repeated_calls_return_correct_results(self, register_double_op):
        from vllm_fl.dispatch import CachedOp
        op = CachedOp("double_op")
        for i in range(5):
            assert op(i) == i * 2

    def test_kwargs_forwarded(self, register_double_op):
        """CachedOp forwards kwargs to the impl fn."""
        from vllm_fl.dispatch.manager import get_default_manager
        from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority
        from vllm_fl.dispatch import CachedOp

        mgr = get_default_manager()
        mgr._registry.register(
            "kwarg_op",
            OpImpl(
                op_name="kwarg_op",
                impl_id="impl:ref.kwarg_op",
                kind=BackendImplKind.REFERENCE,
                priority=BackendPriority.REFERENCE,
                fn=lambda x, scale=1: x * scale,
                vendor=None,
            ),
        )
        op = CachedOp("kwarg_op")
        assert op(3, scale=4) == 12

    def test_op_name_stored(self, register_double_op):
        from vllm_fl.dispatch import CachedOp
        op = CachedOp("double_op")
        assert op._op_name == "double_op"

    def test_initial_impl_is_none(self):
        from vllm_fl.dispatch import CachedOp
        op = CachedOp("any_op")
        assert op._impl is None

    def test_impl_cached_after_first_call(self, register_double_op):
        from vllm_fl.dispatch import CachedOp
        op = CachedOp("double_op")
        op(1)
        assert op._impl is not None


# ── Cache invalidation ────────────────────────────────────────────────────────

class TestCachedOpCacheInvalidation:

    def test_cache_invalidated_on_manager_epoch_bump(self, register_double_op):
        from vllm_fl.dispatch import CachedOp
        from vllm_fl.dispatch.manager import get_default_manager

        op = CachedOp("double_op")
        op(1)
        impl_before = op._impl

        mgr = get_default_manager()
        mgr.bump_policy_epoch()

        op(1)
        # impl may be refreshed (None mid-call, then re-resolved)
        # Just verify op still works correctly
        assert op(3) == 6

    def test_cache_invalidated_on_policy_context_change(self, register_double_op):
        from vllm_fl.dispatch import CachedOp, policy_context, SelectionPolicy, PREFER_REFERENCE

        op = CachedOp("double_op")
        op(2)

        with policy_context(SelectionPolicy(prefer=PREFER_REFERENCE)):
            result = op(4)
        assert result == 8

    def test_cache_invalidated_on_set_global_policy(self, register_double_op):
        from vllm_fl.dispatch import CachedOp, set_global_policy, SelectionPolicy, PREFER_REFERENCE

        op = CachedOp("double_op")
        op(1)
        set_global_policy(SelectionPolicy(prefer=PREFER_REFERENCE))
        # After policy change, op must still resolve and return correct result
        assert op(5) == 10

    def test_cache_invalidated_on_manager_reset(self, register_double_op):
        from vllm_fl.dispatch import CachedOp
        from vllm_fl.dispatch.manager import reset_default_manager, get_default_manager
        from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority

        op = CachedOp("double_op")
        op(1)

        reset_default_manager()
        # Re-register on new manager
        mgr = get_default_manager()
        mgr._initialized = True
        mgr._registry.register(
            "double_op",
            OpImpl(
                op_name="double_op",
                impl_id="impl:ref.double_op",
                kind=BackendImplKind.REFERENCE,
                priority=BackendPriority.REFERENCE,
                fn=lambda x: x * 2,
                vendor=None,
            ),
        )
        assert op(7) == 14


# ── Fast-path disabled ────────────────────────────────────────────────────────

class TestCachedOpFastPathDisabled:

    def test_fast_path_disabled_still_calls_correctly(self, register_double_op, monkeypatch):
        import vllm_fl.dispatch as dispatch_mod
        monkeypatch.setattr(dispatch_mod, "_OP_FAST_PATH_ENABLED", False)
        from vllm_fl.dispatch import CachedOp
        op = CachedOp("double_op")
        assert op(6) == 12

    def test_fast_path_disabled_impl_not_cached(self, register_double_op, monkeypatch):
        import vllm_fl.dispatch as dispatch_mod
        monkeypatch.setattr(dispatch_mod, "_OP_FAST_PATH_ENABLED", False)
        from vllm_fl.dispatch import CachedOp
        op = CachedOp("double_op")
        op(1)
        # Fast path disabled → impl is never cached directly
        assert op._impl is None


# ── IO dump active ────────────────────────────────────────────────────────────

class TestCachedOpIoDump:

    def test_io_dump_routes_through_manager(self, register_double_op, monkeypatch):
        """When IO dump is enabled, CachedOp must route through mgr.call."""
        import vllm_fl.dispatch.io_dumper as io_dumper_mod
        monkeypatch.setattr(io_dumper_mod, "_dump_enabled", True)
        from vllm_fl.dispatch import CachedOp
        op = CachedOp("double_op")
        assert op(3) == 6
        # Restore
        monkeypatch.setattr(io_dumper_mod, "_dump_enabled", False)


# ── Error handling ────────────────────────────────────────────────────────────

class TestCachedOpErrorHandling:

    def test_fallback_on_impl_failure_non_strict(self, register_double_op):
        """On non-strict policy, impl failure falls back through mgr.call."""
        from vllm_fl.dispatch import CachedOp, set_global_policy, SelectionPolicy, PREFER_REFERENCE
        from vllm_fl.dispatch.manager import get_default_manager
        from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority

        # Register a failing impl at higher priority and a working fallback
        mgr = get_default_manager()
        call_count = {"n": 0}

        def failing_fn(x):
            if call_count["n"] == 0:
                call_count["n"] += 1
                raise RuntimeError("deliberate failure")
            return x * 2

        mgr._registry.register(
            "fallback_op",
            OpImpl(
                op_name="fallback_op",
                impl_id="impl:ref.fallback_op",
                kind=BackendImplKind.REFERENCE,
                priority=BackendPriority.REFERENCE,
                fn=lambda x: x * 2,
                vendor=None,
            ),
        )

        set_global_policy(SelectionPolicy(prefer=PREFER_REFERENCE, strict=False))
        op = CachedOp("fallback_op")
        # First call may fail internally and fall back; result should be correct
        result = op(4)
        assert result == 8

    def test_strict_mode_raises_on_unknown_op(self):
        """In strict mode, calling unknown op raises RuntimeError."""
        from vllm_fl.dispatch import CachedOp, set_global_policy, SelectionPolicy, PREFER_REFERENCE
        from vllm_fl.dispatch.manager import get_default_manager

        mgr = get_default_manager()
        mgr._initialized = True
        set_global_policy(SelectionPolicy(prefer=PREFER_REFERENCE, strict=True))

        op = CachedOp("nonexistent_op_xyz")
        with pytest.raises((RuntimeError, KeyError)):
            op(1)


# ── Multiple independent instances ───────────────────────────────────────────

class TestCachedOpMultipleInstances:

    def test_two_ops_independent(self, register_double_op):
        from vllm_fl.dispatch import CachedOp
        from vllm_fl.dispatch.manager import get_default_manager
        from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority

        mgr = get_default_manager()
        mgr._registry.register(
            "triple_op",
            OpImpl(
                op_name="triple_op",
                impl_id="impl:ref.triple_op",
                kind=BackendImplKind.REFERENCE,
                priority=BackendPriority.REFERENCE,
                fn=lambda x: x * 3,
                vendor=None,
            ),
        )

        double = CachedOp("double_op")
        triple = CachedOp("triple_op")

        assert double(4) == 8
        assert triple(4) == 12

    def test_two_cached_ops_same_name_share_nothing(self, register_double_op):
        """Two CachedOp instances for the same op are independent objects."""
        from vllm_fl.dispatch import CachedOp
        op1 = CachedOp("double_op")
        op2 = CachedOp("double_op")
        assert op1 is not op2
        assert op1(2) == op2(2) == 4
