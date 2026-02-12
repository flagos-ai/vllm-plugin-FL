# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for dispatch call_op and resolve_op convenience functions.

This module tests the high-level dispatch API exposed through the
dispatch module's __init__.py, ensuring the full dispatch pipeline
works correctly from call_op -> manager -> registry -> implementation.
"""

import os
import pytest
from unittest.mock import patch

from vllm_fl.dispatch import (
    call_op,
    resolve_op,
    get_default_manager,
    reset_default_manager,
    OpImpl,
    BackendImplKind,
    BackendPriority,
    SelectionPolicy,
    set_global_policy,
    reset_global_policy,
    with_preference,
    PREFER_VENDOR,
    PREFER_REFERENCE,
)


class TestCallOp:
    """Test call_op convenience function."""

    @pytest.fixture(autouse=True)
    def reset_all(self):
        """Reset global state before and after each test."""
        reset_default_manager()
        reset_global_policy()
        yield
        reset_default_manager()
        reset_global_policy()

    @pytest.fixture
    def setup_test_op(self):
        """Setup a test operator in the registry."""
        manager = get_default_manager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()

        def impl_fn(x, multiplier=2):
            return x * multiplier

        impl = OpImpl(
            op_name="test_call_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=impl_fn,
        )
        manager.registry.register_impl(impl)
        return impl_fn

    def test_call_op_basic(self, setup_test_op):
        result = call_op("test_call_op", 5)
        assert result == 10

    def test_call_op_with_kwargs(self, setup_test_op):
        result = call_op("test_call_op", 5, multiplier=3)
        assert result == 15

    def test_call_op_nonexistent_raises(self):
        manager = get_default_manager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()

        with pytest.raises(RuntimeError, match="No available implementation"):
            call_op("nonexistent_op", 1)

    def test_call_op_uses_default_manager(self):
        manager = get_default_manager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()

        call_tracker = {"called": False}

        def tracking_fn(x):
            call_tracker["called"] = True
            return x

        manager.registry.register_impl(OpImpl(
            op_name="track_op",
            impl_id="default.track",
            kind=BackendImplKind.DEFAULT,
            fn=tracking_fn,
        ))

        call_op("track_op", 1)
        assert call_tracker["called"] is True


class TestResolveOp:
    """Test resolve_op convenience function."""

    @pytest.fixture(autouse=True)
    def reset_all(self):
        reset_default_manager()
        reset_global_policy()
        yield
        reset_default_manager()
        reset_global_policy()

    @pytest.fixture
    def setup_test_op(self):
        manager = get_default_manager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()

        def impl_fn(x):
            return x * 2

        impl = OpImpl(
            op_name="test_resolve_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=impl_fn,
        )
        manager.registry.register_impl(impl)
        return impl_fn

    def test_resolve_op_returns_function(self, setup_test_op):
        fn = resolve_op("test_resolve_op")
        assert callable(fn)
        assert fn is setup_test_op

    def test_resolve_op_can_be_called(self, setup_test_op):
        fn = resolve_op("test_resolve_op")
        result = fn(5)
        assert result == 10

    def test_resolve_op_nonexistent_raises(self):
        manager = get_default_manager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()

        with pytest.raises(RuntimeError, match="No available implementation"):
            resolve_op("nonexistent_op")


class TestCallOpWithPolicy:
    """Test call_op behavior with different policies."""

    @pytest.fixture(autouse=True)
    def reset_all(self):
        reset_default_manager()
        reset_global_policy()
        yield
        reset_default_manager()
        reset_global_policy()

    @pytest.fixture
    def setup_multi_impl_op(self):
        """Setup an operator with multiple implementations."""
        manager = get_default_manager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()

        results = {"default": 0, "vendor": 0, "reference": 0}

        def default_fn(x):
            results["default"] += 1
            return x * 2

        def vendor_fn(x):
            results["vendor"] += 1
            return x * 3

        def reference_fn(x):
            results["reference"] += 1
            return x * 4

        manager.registry.register_impl(OpImpl(
            op_name="policy_op",
            impl_id="default.impl",
            kind=BackendImplKind.DEFAULT,
            fn=default_fn,
            priority=BackendPriority.DEFAULT,
        ))
        manager.registry.register_impl(OpImpl(
            op_name="policy_op",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=vendor_fn,
            priority=BackendPriority.VENDOR,
            vendor="CUDA",
        ))
        manager.registry.register_impl(OpImpl(
            op_name="policy_op",
            impl_id="reference.pytorch",
            kind=BackendImplKind.REFERENCE,
            fn=reference_fn,
            priority=BackendPriority.REFERENCE,
        ))

        return results

    def test_call_op_default_policy_uses_default(self, setup_multi_impl_op):
        results = setup_multi_impl_op

        result = call_op("policy_op", 5)

        assert result == 10  # default_fn: x * 2
        assert results["default"] == 1
        assert results["vendor"] == 0
        assert results["reference"] == 0

    def test_call_op_vendor_policy(self, setup_multi_impl_op):
        results = setup_multi_impl_op

        set_global_policy(SelectionPolicy(prefer=PREFER_VENDOR))
        # Clear cache after policy change
        get_default_manager().bump_policy_epoch()

        result = call_op("policy_op", 5)

        assert result == 15  # vendor_fn: x * 3
        assert results["vendor"] == 1

    def test_call_op_reference_policy(self, setup_multi_impl_op):
        results = setup_multi_impl_op

        set_global_policy(SelectionPolicy(prefer=PREFER_REFERENCE))
        get_default_manager().bump_policy_epoch()

        result = call_op("policy_op", 5)

        assert result == 20  # reference_fn: x * 4
        assert results["reference"] == 1

    def test_call_op_with_policy_context(self, setup_multi_impl_op):
        results = setup_multi_impl_op

        # Default call
        result1 = call_op("policy_op", 5)
        assert result1 == 10
        assert results["default"] == 1

        # With vendor preference context
        with with_preference("vendor"):
            get_default_manager().bump_policy_epoch()
            result2 = call_op("policy_op", 5)
            assert result2 == 15
            assert results["vendor"] == 1

        # Back to default after context
        get_default_manager().bump_policy_epoch()
        result3 = call_op("policy_op", 5)
        assert result3 == 10
        assert results["default"] == 2


class TestCallOpIntegration:
    """Integration tests for the full dispatch pipeline."""

    @pytest.fixture(autouse=True)
    def reset_all(self):
        reset_default_manager()
        reset_global_policy()
        yield
        reset_default_manager()
        reset_global_policy()

    def test_full_pipeline_with_multiple_ops(self):
        """Test calling multiple different operators."""
        manager = get_default_manager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()

        # Register multiple operators
        manager.registry.register_impl(OpImpl(
            op_name="add_op",
            impl_id="default.add",
            kind=BackendImplKind.DEFAULT,
            fn=lambda x, y: x + y,
        ))
        manager.registry.register_impl(OpImpl(
            op_name="mul_op",
            impl_id="default.mul",
            kind=BackendImplKind.DEFAULT,
            fn=lambda x, y: x * y,
        ))
        manager.registry.register_impl(OpImpl(
            op_name="sub_op",
            impl_id="default.sub",
            kind=BackendImplKind.DEFAULT,
            fn=lambda x, y: x - y,
        ))

        # Call each operator
        assert call_op("add_op", 2, 3) == 5
        assert call_op("mul_op", 2, 3) == 6
        assert call_op("sub_op", 5, 3) == 2

    def test_resolve_and_call_consistency(self):
        """Test that resolve_op and call_op give consistent results."""
        manager = get_default_manager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()

        manager.registry.register_impl(OpImpl(
            op_name="consistent_op",
            impl_id="default.impl",
            kind=BackendImplKind.DEFAULT,
            fn=lambda x: x * 10,
        ))

        # Both should give same result
        fn = resolve_op("consistent_op")
        result1 = fn(5)
        result2 = call_op("consistent_op", 5)

        assert result1 == result2 == 50

    @patch.dict(os.environ, {"VLLM_FL_STRICT": "1"})
    def test_fallback_chain(self):
        """Test fallback from failed impl to successful one."""
        manager = get_default_manager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()

        call_sequence = []

        def failing_impl(x):
            call_sequence.append("failing")
            raise RuntimeError("Intentional failure")

        def success_impl(x):
            call_sequence.append("success")
            return x * 2

        manager.registry.register_impl(OpImpl(
            op_name="fallback_chain_op",
            impl_id="default.failing",
            kind=BackendImplKind.DEFAULT,
            fn=failing_impl,
            priority=200,
        ))
        manager.registry.register_impl(OpImpl(
            op_name="fallback_chain_op",
            impl_id="reference.success",
            kind=BackendImplKind.REFERENCE,
            fn=success_impl,
            priority=100,
        ))

        result = call_op("fallback_chain_op", 5)

        assert result == 10
        assert call_sequence == ["failing", "success"]

    def test_vendor_filtering(self):
        """Test that vendor filtering works through call_op."""
        manager = get_default_manager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()

        manager.registry.register_impl(OpImpl(
            op_name="vendor_filter_op",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=lambda x: x * 2,
            vendor="CUDA",
        ))
        manager.registry.register_impl(OpImpl(
            op_name="vendor_filter_op",
            impl_id="vendor.amd",
            kind=BackendImplKind.VENDOR,
            fn=lambda x: x * 3,
            vendor="AMD",
        ))
        manager.registry.register_impl(OpImpl(
            op_name="vendor_filter_op",
            impl_id="reference.pytorch",
            kind=BackendImplKind.REFERENCE,
            fn=lambda x: x * 4,
        ))

        # Deny AMD, prefer vendor -> should use CUDA
        set_global_policy(SelectionPolicy(
            prefer=PREFER_VENDOR,
            deny_vendors=frozenset({"AMD"})
        ))
        manager.bump_policy_epoch()

        result = call_op("vendor_filter_op", 5)
        assert result == 10  # CUDA: x * 2
