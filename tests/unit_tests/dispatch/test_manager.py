# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for dispatch manager module.

This module tests the core OpManager class which handles:
- Operator resolution and selection
- Dispatch caching with policy epoch invalidation
- Fallback mechanisms for failed implementations
- Multi-process safety (fork handling)
"""

import os
import pytest
import threading
from unittest.mock import patch

from vllm_fl.dispatch.manager import (
    OpManager,
    get_default_manager,
    reset_default_manager,
    _OpManagerState,
)
from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority
from vllm_fl.dispatch.policy import (
    SelectionPolicy,
    set_global_policy,
    reset_global_policy,
    PREFER_VENDOR,
    PREFER_REFERENCE,
)


class TestOpManagerState:
    """Test _OpManagerState dataclass."""

    def test_default_values(self):
        state = _OpManagerState()
        assert state.init_pid == -1
        assert state.initialized is False
        assert state.policy_epoch == 0


class TestOpManagerBasic:
    """Test basic OpManager functionality."""

    @pytest.fixture
    def registry(self):
        return OpRegistry()

    @pytest.fixture
    def manager(self, registry):
        return OpManager(registry=registry)

    @pytest.fixture
    def sample_impl(self):
        return OpImpl(
            op_name="test_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=lambda x: x * 2,
            priority=BackendPriority.DEFAULT,
        )

    @pytest.fixture
    def reference_impl(self):
        return OpImpl(
            op_name="test_op",
            impl_id="reference.test",
            kind=BackendImplKind.REFERENCE,
            fn=lambda x: x * 2,
            priority=BackendPriority.REFERENCE,
        )

    @pytest.fixture
    def vendor_impl(self):
        return OpImpl(
            op_name="test_op",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=lambda x: x * 3,
            priority=BackendPriority.VENDOR,
            vendor="CUDA",
        )

    def test_init_with_custom_registry(self, registry):
        manager = OpManager(registry=registry)
        assert manager.registry is registry

    def test_init_creates_default_registry(self):
        manager = OpManager()
        assert manager.registry is not None
        assert isinstance(manager.registry, OpRegistry)

    def test_registry_property(self, manager, registry):
        assert manager.registry is registry


class TestOpManagerPolicyEpoch:
    """Test policy epoch and cache invalidation."""

    @pytest.fixture(autouse=True)
    def reset_policy(self):
        reset_global_policy()
        yield
        reset_global_policy()

    @pytest.fixture
    def manager(self):
        return OpManager()

    def test_bump_policy_epoch(self, manager):
        initial_epoch = manager._state.policy_epoch
        manager.bump_policy_epoch()
        assert manager._state.policy_epoch == initial_epoch + 1

    def test_bump_policy_epoch_clears_cache(self, manager):
        # Add something to cache
        manager._dispatch_cache[("test", "fp", 0)] = lambda x: x
        assert len(manager._dispatch_cache) == 1

        manager.bump_policy_epoch()
        assert len(manager._dispatch_cache) == 0

    def test_bump_policy_epoch_clears_failed_impls(self, manager):
        manager._failed_impls["test_op"] = {"impl1", "impl2"}
        manager.bump_policy_epoch()
        assert len(manager._failed_impls) == 0


class TestOpManagerFailedImpls:
    """Test failed implementation tracking."""

    @pytest.fixture
    def manager(self):
        return OpManager()

    def test_clear_failed_impls_all(self, manager):
        manager._failed_impls["op1"] = {"impl1"}
        manager._failed_impls["op2"] = {"impl2"}

        manager.clear_failed_impls()
        assert manager._failed_impls == {}

    def test_clear_failed_impls_specific_op(self, manager):
        manager._failed_impls["op1"] = {"impl1"}
        manager._failed_impls["op2"] = {"impl2"}

        manager.clear_failed_impls("op1")
        assert "op1" not in manager._failed_impls
        assert "op2" in manager._failed_impls

    def test_clear_failed_impls_nonexistent_op(self, manager):
        manager._failed_impls["op1"] = {"impl1"}
        manager.clear_failed_impls("nonexistent")
        assert "op1" in manager._failed_impls

    def test_get_failed_impls_all(self, manager):
        manager._failed_impls["op1"] = {"impl1", "impl2"}
        manager._failed_impls["op2"] = {"impl3"}

        result = manager.get_failed_impls()
        assert result == {"op1": {"impl1", "impl2"}, "op2": {"impl3"}}

    def test_get_failed_impls_specific_op(self, manager):
        manager._failed_impls["op1"] = {"impl1"}
        manager._failed_impls["op2"] = {"impl2"}

        result = manager.get_failed_impls("op1")
        assert result == {"op1": {"impl1"}}

    def test_get_failed_impls_returns_copy(self, manager):
        manager._failed_impls["op1"] = {"impl1"}
        result = manager.get_failed_impls()

        # Modify the result
        result["op1"].add("impl2")

        # Original should not be modified
        assert manager._failed_impls["op1"] == {"impl1"}


class TestOpManagerResolve:
    """Test operator resolution logic."""

    @pytest.fixture(autouse=True)
    def reset_policy(self):
        reset_global_policy()
        yield
        reset_global_policy()

    @pytest.fixture
    def registry(self):
        return OpRegistry()

    @pytest.fixture
    def manager(self, registry):
        mgr = OpManager(registry=registry)
        # Mark as initialized to skip builtin registration
        mgr._state.initialized = True
        mgr._state.init_pid = os.getpid()
        return mgr

    def test_resolve_single_impl(self, manager, registry):
        impl = OpImpl(
            op_name="single_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=lambda x: x * 2,
        )
        registry.register_impl(impl)

        fn = manager.resolve("single_op")
        assert fn is impl.fn

    def test_resolve_prefers_default_by_policy(self, manager, registry):
        default_fn = lambda x: x * 2
        reference_fn = lambda x: x * 3

        registry.register_impl(OpImpl(
            op_name="multi_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=default_fn,
            priority=BackendPriority.DEFAULT,
        ))
        registry.register_impl(OpImpl(
            op_name="multi_op",
            impl_id="reference.test",
            kind=BackendImplKind.REFERENCE,
            fn=reference_fn,
            priority=BackendPriority.REFERENCE,
        ))

        # Default policy prefers "flagos" (DEFAULT)
        fn = manager.resolve("multi_op")
        assert fn is default_fn

    def test_resolve_prefers_vendor_with_policy(self, manager, registry):
        default_fn = lambda x: x * 2
        vendor_fn = lambda x: x * 3

        registry.register_impl(OpImpl(
            op_name="vendor_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=default_fn,
        ))
        registry.register_impl(OpImpl(
            op_name="vendor_op",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=vendor_fn,
            vendor="CUDA",
        ))

        # Set policy to prefer vendor
        set_global_policy(SelectionPolicy(prefer=PREFER_VENDOR))

        fn = manager.resolve("vendor_op")
        assert fn is vendor_fn

    def test_resolve_prefers_reference_with_policy(self, manager, registry):
        default_fn = lambda x: x * 2
        reference_fn = lambda x: x * 3

        registry.register_impl(OpImpl(
            op_name="ref_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=default_fn,
        ))
        registry.register_impl(OpImpl(
            op_name="ref_op",
            impl_id="reference.test",
            kind=BackendImplKind.REFERENCE,
            fn=reference_fn,
        ))

        # Set policy to prefer reference
        set_global_policy(SelectionPolicy(prefer=PREFER_REFERENCE))

        fn = manager.resolve("ref_op")
        assert fn is reference_fn

    def test_resolve_filters_denied_vendors(self, manager, registry):
        default_fn = lambda x: x * 2
        vendor_fn = lambda x: x * 3

        registry.register_impl(OpImpl(
            op_name="deny_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=default_fn,
        ))
        registry.register_impl(OpImpl(
            op_name="deny_op",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=vendor_fn,
            vendor="CUDA",
        ))

        # Deny CUDA vendor, prefer vendor
        set_global_policy(SelectionPolicy(
            prefer=PREFER_VENDOR,
            deny_vendors=frozenset({"CUDA"})
        ))

        # Should fall back to default since CUDA is denied
        fn = manager.resolve("deny_op")
        assert fn is default_fn

    def test_resolve_filters_by_allow_vendors(self, manager, registry):
        vendor_cuda_fn = lambda x: x * 2
        vendor_amd_fn = lambda x: x * 3

        registry.register_impl(OpImpl(
            op_name="allow_op",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            fn=vendor_cuda_fn,
            vendor="CUDA",
        ))
        registry.register_impl(OpImpl(
            op_name="allow_op",
            impl_id="vendor.amd",
            kind=BackendImplKind.VENDOR,
            fn=vendor_amd_fn,
            vendor="AMD",
        ))

        # Only allow CUDA vendor
        set_global_policy(SelectionPolicy(
            prefer=PREFER_VENDOR,
            allow_vendors=frozenset({"CUDA"})
        ))

        fn = manager.resolve("allow_op")
        assert fn is vendor_cuda_fn

    def test_resolve_no_impl_raises(self, manager):
        with pytest.raises(RuntimeError, match="No available implementation"):
            manager.resolve("nonexistent_op")

    def test_resolve_caches_result(self, manager, registry):
        impl = OpImpl(
            op_name="cache_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=lambda x: x,
        )
        registry.register_impl(impl)

        # First call
        fn1 = manager.resolve("cache_op")

        # Second call should return cached result
        fn2 = manager.resolve("cache_op")

        assert fn1 is fn2
        assert len(manager._dispatch_cache) == 1

    def test_resolve_cache_invalidated_by_policy_change(self, manager, registry):
        impl = OpImpl(
            op_name="epoch_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=lambda x: x,
        )
        registry.register_impl(impl)

        # First resolve
        manager.resolve("epoch_op")
        assert len(manager._dispatch_cache) == 1

        # Bump epoch (simulates policy change)
        manager.bump_policy_epoch()
        assert len(manager._dispatch_cache) == 0

    def test_resolve_filters_unavailable_impls(self, manager, registry):
        available_fn = lambda x: x * 2
        unavailable_fn = lambda x: x * 3

        # Create an unavailable implementation
        def check_available():
            return False
        unavailable_fn._is_available = check_available

        registry.register_impl(OpImpl(
            op_name="avail_op",
            impl_id="default.unavailable",
            kind=BackendImplKind.DEFAULT,
            fn=unavailable_fn,
            priority=200,  # Higher priority but unavailable
        ))
        registry.register_impl(OpImpl(
            op_name="avail_op",
            impl_id="default.available",
            kind=BackendImplKind.DEFAULT,
            fn=available_fn,
            priority=100,
        ))

        fn = manager.resolve("avail_op")
        assert fn is available_fn


class TestOpManagerCall:
    """Test operator call with fallback support."""

    @pytest.fixture(autouse=True)
    def reset_policy(self):
        reset_global_policy()
        yield
        reset_global_policy()

    @pytest.fixture
    def registry(self):
        return OpRegistry()

    @pytest.fixture
    def manager(self, registry):
        mgr = OpManager(registry=registry)
        mgr._state.initialized = True
        mgr._state.init_pid = os.getpid()
        return mgr

    def test_call_invokes_implementation(self, manager, registry):
        result_value = [0]

        def impl_fn(x):
            result_value[0] = x * 2
            return result_value[0]

        registry.register_impl(OpImpl(
            op_name="call_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=impl_fn,
        ))

        result = manager.call("call_op", 5)
        assert result == 10
        assert result_value[0] == 10

    def test_call_passes_args_and_kwargs(self, manager, registry):
        def impl_fn(a, b, c=10):
            return a + b + c

        registry.register_impl(OpImpl(
            op_name="args_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=impl_fn,
        ))

        result = manager.call("args_op", 1, 2, c=3)
        assert result == 6

    @patch.dict(os.environ, {"VLLM_FL_STRICT": "1"})
    def test_call_fallback_on_failure(self, manager, registry):
        call_order = []

        def failing_fn(x):
            call_order.append("failing")
            raise RuntimeError("Primary failed")

        def fallback_fn(x):
            call_order.append("fallback")
            return x * 2

        registry.register_impl(OpImpl(
            op_name="fallback_op",
            impl_id="default.primary",
            kind=BackendImplKind.DEFAULT,
            fn=failing_fn,
            priority=200,
        ))
        registry.register_impl(OpImpl(
            op_name="fallback_op",
            impl_id="reference.fallback",
            kind=BackendImplKind.REFERENCE,
            fn=fallback_fn,
            priority=100,
        ))

        result = manager.call("fallback_op", 5)
        assert result == 10
        assert call_order == ["failing", "fallback"]

    @patch.dict(os.environ, {"VLLM_FL_STRICT": "1"})
    def test_call_tracks_failed_impls(self, manager, registry):
        def failing_fn(x):
            raise RuntimeError("Failed")

        def success_fn(x):
            return x

        registry.register_impl(OpImpl(
            op_name="track_op",
            impl_id="default.failing",
            kind=BackendImplKind.DEFAULT,
            fn=failing_fn,
            priority=200,
        ))
        registry.register_impl(OpImpl(
            op_name="track_op",
            impl_id="reference.success",
            kind=BackendImplKind.REFERENCE,
            fn=success_fn,
            priority=100,
        ))

        manager.call("track_op", 1)

        # Check that failed impl is tracked
        failed = manager.get_failed_impls("track_op")
        assert "default.failing" in failed["track_op"]

    @patch.dict(os.environ, {"VLLM_FL_STRICT": "1"})
    def test_call_all_impls_fail_raises(self, manager, registry):
        def failing_fn1(x):
            raise RuntimeError("Failed 1")

        def failing_fn2(x):
            raise RuntimeError("Failed 2")

        registry.register_impl(OpImpl(
            op_name="allfail_op",
            impl_id="default.fail1",
            kind=BackendImplKind.DEFAULT,
            fn=failing_fn1,
        ))
        registry.register_impl(OpImpl(
            op_name="allfail_op",
            impl_id="reference.fail2",
            kind=BackendImplKind.REFERENCE,
            fn=failing_fn2,
        ))

        with pytest.raises(RuntimeError, match="implementation.*failed"):
            manager.call("allfail_op", 1)

    @patch.dict(os.environ, {"VLLM_FL_STRICT": "0"})
    def test_call_no_fallback_when_disabled(self, manager, registry):
        def failing_fn(x):
            raise RuntimeError("Primary failed")

        def fallback_fn(x):
            return x * 2

        registry.register_impl(OpImpl(
            op_name="nofallback_op",
            impl_id="default.primary",
            kind=BackendImplKind.DEFAULT,
            fn=failing_fn,
            priority=200,
        ))
        registry.register_impl(OpImpl(
            op_name="nofallback_op",
            impl_id="reference.fallback",
            kind=BackendImplKind.REFERENCE,
            fn=fallback_fn,
            priority=100,
        ))

        # Should raise immediately without trying fallback
        with pytest.raises(RuntimeError, match="Primary failed"):
            manager.call("nofallback_op", 5)


class TestOpManagerResolveCandidates:
    """Test resolve_candidates method."""

    @pytest.fixture(autouse=True)
    def reset_policy(self):
        reset_global_policy()
        yield
        reset_global_policy()

    @pytest.fixture
    def registry(self):
        return OpRegistry()

    @pytest.fixture
    def manager(self, registry):
        mgr = OpManager(registry=registry)
        mgr._state.initialized = True
        mgr._state.init_pid = os.getpid()
        return mgr

    def test_resolve_candidates_returns_sorted_list(self, manager, registry):
        fn1 = lambda x: x
        fn2 = lambda x: x
        fn3 = lambda x: x

        registry.register_impl(OpImpl(
            op_name="multi_op",
            impl_id="default.impl",
            kind=BackendImplKind.DEFAULT,
            fn=fn1,
            priority=BackendPriority.DEFAULT,
        ))
        registry.register_impl(OpImpl(
            op_name="multi_op",
            impl_id="vendor.impl",
            kind=BackendImplKind.VENDOR,
            fn=fn2,
            priority=BackendPriority.VENDOR,
            vendor="CUDA",
        ))
        registry.register_impl(OpImpl(
            op_name="multi_op",
            impl_id="reference.impl",
            kind=BackendImplKind.REFERENCE,
            fn=fn3,
            priority=BackendPriority.REFERENCE,
        ))

        candidates = manager.resolve_candidates("multi_op")

        # Default policy: flagos > vendor > reference
        assert len(candidates) == 3
        assert candidates[0].impl_id == "default.impl"
        assert candidates[1].impl_id == "vendor.impl"
        assert candidates[2].impl_id == "reference.impl"

    def test_resolve_candidates_respects_policy_order(self, manager, registry):
        fn1 = lambda x: x
        fn2 = lambda x: x

        registry.register_impl(OpImpl(
            op_name="order_op",
            impl_id="default.impl",
            kind=BackendImplKind.DEFAULT,
            fn=fn1,
        ))
        registry.register_impl(OpImpl(
            op_name="order_op",
            impl_id="reference.impl",
            kind=BackendImplKind.REFERENCE,
            fn=fn2,
        ))

        # Set policy to prefer reference
        set_global_policy(SelectionPolicy(prefer=PREFER_REFERENCE))

        candidates = manager.resolve_candidates("order_op")

        # Reference should come first
        assert candidates[0].impl_id == "reference.impl"
        assert candidates[1].impl_id == "default.impl"

    def test_resolve_candidates_no_impl_raises(self, manager):
        with pytest.raises(RuntimeError, match="No available implementation"):
            manager.resolve_candidates("nonexistent")


class TestOpManagerGetSelectedImplId:
    """Test get_selected_impl_id method."""

    @pytest.fixture(autouse=True)
    def reset_policy(self):
        reset_global_policy()
        yield
        reset_global_policy()

    @pytest.fixture
    def registry(self):
        return OpRegistry()

    @pytest.fixture
    def manager(self, registry):
        mgr = OpManager(registry=registry)
        mgr._state.initialized = True
        mgr._state.init_pid = os.getpid()
        return mgr

    def test_get_selected_impl_id(self, manager, registry):
        fn = lambda x: x

        registry.register_impl(OpImpl(
            op_name="id_op",
            impl_id="default.test",
            kind=BackendImplKind.DEFAULT,
            fn=fn,
        ))

        impl_id = manager.get_selected_impl_id("id_op")
        assert impl_id == "default.test"


class TestOpManagerThreadSafety:
    """Test thread safety of OpManager."""

    @pytest.fixture
    def registry(self):
        return OpRegistry()

    @pytest.fixture
    def manager(self, registry):
        mgr = OpManager(registry=registry)
        mgr._state.initialized = True
        mgr._state.init_pid = os.getpid()
        return mgr

    def test_concurrent_resolve(self, manager, registry):
        # Register multiple implementations
        for i in range(5):
            registry.register_impl(OpImpl(
                op_name=f"thread_op_{i}",
                impl_id=f"default.impl_{i}",
                kind=BackendImplKind.DEFAULT,
                fn=lambda x, i=i: x + i,
            ))

        errors = []
        results = []

        def resolve_op(op_idx):
            try:
                fn = manager.resolve(f"thread_op_{op_idx}")
                results.append((op_idx, fn(10)))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=resolve_op, args=(i % 5,))
            for i in range(20)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20

    def test_concurrent_bump_policy_epoch(self, manager, registry):
        registry.register_impl(OpImpl(
            op_name="epoch_test",
            impl_id="default.impl",
            kind=BackendImplKind.DEFAULT,
            fn=lambda x: x,
        ))

        errors = []

        def bump_and_resolve():
            try:
                manager.bump_policy_epoch()
                manager.resolve("epoch_test")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=bump_and_resolve)
            for _ in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestGlobalDefaultManager:
    """Test global default manager functions."""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        reset_default_manager()
        yield
        reset_default_manager()

    def test_get_default_manager_singleton(self):
        manager1 = get_default_manager()
        manager2 = get_default_manager()
        assert manager1 is manager2

    def test_reset_default_manager(self):
        manager1 = get_default_manager()
        reset_default_manager()
        manager2 = get_default_manager()
        assert manager1 is not manager2

    def test_get_default_manager_creates_instance(self):
        manager = get_default_manager()
        assert isinstance(manager, OpManager)
