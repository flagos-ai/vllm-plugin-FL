# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for dispatch registry.
"""

import pytest
from vllm_fl.dispatch.registry import OpRegistry, OpRegistrySnapshot
from vllm_fl.dispatch.types import BackendImplKind, OpImpl


class TestOpRegistry:
    @pytest.fixture
    def registry(self):
        return OpRegistry()

    @pytest.fixture
    def sample_impl(self):
        return OpImpl(
            op_name="silu",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=lambda x: x,
            priority=100,
        )

    @pytest.fixture
    def another_impl(self):
        return OpImpl(
            op_name="silu",
            impl_id="reference.pytorch",
            kind=BackendImplKind.REFERENCE,
            fn=lambda x: x,
            priority=50,
        )

    def test_register_impl(self, registry, sample_impl):
        registry.register_impl(sample_impl)
        result = registry.get_implementation("silu", "default.flagos")
        assert result == sample_impl

    def test_register_impl_duplicate_raises(self, registry, sample_impl):
        registry.register_impl(sample_impl)
        with pytest.raises(ValueError, match="Duplicate impl_id"):
            registry.register_impl(sample_impl)

    def test_register_many(self, registry, sample_impl, another_impl):
        registry.register_many([sample_impl, another_impl])
        assert registry.get_implementation("silu", "default.flagos") == sample_impl
        assert registry.get_implementation("silu", "reference.pytorch") == another_impl

    def test_get_implementations(self, registry, sample_impl, another_impl):
        registry.register_many([sample_impl, another_impl])
        impls = registry.get_implementations("silu")
        assert len(impls) == 2
        assert sample_impl in impls
        assert another_impl in impls

    def test_get_implementations_empty(self, registry):
        impls = registry.get_implementations("nonexistent")
        assert impls == []

    def test_get_implementation_not_found(self, registry):
        result = registry.get_implementation("nonexistent", "any_id")
        assert result is None

    def test_list_operators(self, registry, sample_impl, another_impl):
        rms_impl = OpImpl(
            op_name="rms_norm",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=lambda x: x,
        )
        registry.register_many([sample_impl, another_impl, rms_impl])
        ops = registry.list_operators()
        assert set(ops) == {"silu", "rms_norm"}

    def test_list_operators_empty(self, registry):
        assert registry.list_operators() == []

    def test_clear(self, registry, sample_impl):
        registry.register_impl(sample_impl)
        registry.clear()
        assert registry.list_operators() == []
        assert registry.get_implementation("silu", "default.flagos") is None

    def test_snapshot(self, registry, sample_impl, another_impl):
        registry.register_many([sample_impl, another_impl])
        snapshot = registry.snapshot()

        assert isinstance(snapshot, OpRegistrySnapshot)
        assert "silu" in snapshot.impls_by_op
        assert len(snapshot.impls_by_op["silu"]) == 2

    def test_snapshot_is_immutable_copy(self, registry, sample_impl):
        registry.register_impl(sample_impl)
        snapshot = registry.snapshot()

        # Register more after snapshot
        new_impl = OpImpl(
            op_name="rms_norm",
            impl_id="default.flagos",
            kind=BackendImplKind.DEFAULT,
            fn=lambda x: x,
        )
        registry.register_impl(new_impl)

        # Snapshot should not contain the new impl
        assert "rms_norm" not in snapshot.impls_by_op

    def test_thread_safety(self, registry):
        """Test basic thread safety of registry operations."""
        import threading

        errors = []

        def register_impl(i):
            try:
                impl = OpImpl(
                    op_name=f"op_{i}",
                    impl_id=f"impl_{i}",
                    kind=BackendImplKind.DEFAULT,
                    fn=lambda x: x,
                )
                registry.register_impl(impl)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_impl, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry.list_operators()) == 10
