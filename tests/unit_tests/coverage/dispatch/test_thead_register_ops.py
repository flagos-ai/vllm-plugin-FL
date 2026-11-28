# Copyright 2025 BAAI. All rights reserved.

"""register_ops contract for the T-Head (PPU) dispatch backend.

``register_builtins`` must contribute exactly one VENDOR-kind OpImpl — the
attention backend bridge — wrapped by ``_bind_is_available`` so the dispatch
policy can probe hardware availability through the impl's ``fn`` without
importing the backend eagerly.

All tests are CPU-only and use a fresh registry per test.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.supplementary]


@pytest.fixture(autouse=True)
def reset_availability_memo():
    """Reset the class-level availability memo around every test."""
    from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

    TheadBackend._available = None
    yield
    TheadBackend._available = None


class TestRegisterBuiltins:
    """register_builtins installs a single vendor attention_backend impl."""

    @staticmethod
    def _fresh_registry():
        from vllm_fl.dispatch.registry import OpRegistry

        return OpRegistry()

    @pytest.mark.platforms("thead")
    def test_registers_exactly_one_vendor_impl(self):
        from vllm_fl.dispatch.backends.vendor.thead import register_ops
        from vllm_fl.dispatch.types import BackendImplKind, BackendPriority

        registry = self._fresh_registry()
        register_ops.register_builtins(registry)

        snapshot = registry.snapshot()
        assert list(snapshot.impls_by_op.keys()) == ["attention_backend"], (
            "thead must register only the attention_backend op"
        )
        impls = snapshot.impls_by_op["attention_backend"]
        assert len(impls) == 1, "expected exactly one attention_backend impl"
        impl = impls[0]

        assert impl.op_name == "attention_backend"
        assert impl.impl_id == "vendor.thead"
        assert impl.kind is BackendImplKind.VENDOR
        assert impl.kind == BackendImplKind.VENDOR  # str-enum comparison
        assert impl.vendor == "thead"
        assert impl.priority == BackendPriority.VENDOR
        assert impl.priority == 100

    @pytest.mark.platforms("thead")
    def test_registry_lookup_by_op_and_impl_id(self):
        from vllm_fl.dispatch.backends.vendor.thead import register_ops

        registry = self._fresh_registry()
        register_ops.register_builtins(registry)

        impl = registry.get_implementation("attention_backend", "vendor.thead")
        assert impl is not None
        assert impl.fn is not None and callable(impl.fn)
        # The registry is keyed per op; nothing leaks into unrelated ops.
        assert registry.get_implementations("attention_backend")
        assert registry.get_implementations("silu_and_mul") == []
        assert registry.list_operators() == ["attention_backend"]

    @pytest.mark.platforms("thead")
    def test_registered_fn_matches_direct_backend_call(self):
        """The impl's fn is TheadBackend.attention_backend, bound and usable."""
        from vllm_fl.dispatch.backends.vendor.thead import register_ops
        from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

        registry = self._fresh_registry()
        register_ops.register_builtins(registry)
        impl = registry.get_implementation("attention_backend", "vendor.thead")

        backend = TheadBackend()
        assert impl.fn() == backend.attention_backend()
        assert impl.fn(use_mla=False) == backend.attention_backend(
            use_mla=False, use_sparse=False
        )

    @pytest.mark.platforms("thead")
    def test_impl_fn_carries_is_available_binding(self):
        """_bind_is_available attaches the backend probe to the impl's fn."""
        from vllm_fl.dispatch.backends.vendor.thead import register_ops
        from vllm_fl.dispatch.backends.vendor.thead.thead import TheadBackend

        registry = self._fresh_registry()
        register_ops.register_builtins(registry)
        impl = registry.get_implementation("attention_backend", "vendor.thead")

        probe = getattr(impl.fn, "_is_available", None)
        assert callable(probe), "impl fn must carry a _is_available probe"
        # Same bound method target as the backend's probe.
        assert probe.__self__ is not None and isinstance(probe.__self__, TheadBackend)
        assert probe.__func__ is TheadBackend.is_available
        # OpImpl.is_available delegates to the bound probe.
        assert impl.is_available() == probe()


class TestBindIsAvailable:
    """_bind_is_available wraps while preserving and delegating."""

    @pytest.mark.platforms("thead")
    def test_sets_is_available_attribute_and_delegates(self):
        from vllm_fl.dispatch.backends.vendor.thead.register_ops import (
            _bind_is_available,
        )

        recorded = []

        def target(*args, **kwargs):
            recorded.append((args, kwargs))
            return "result"

        probe = lambda: "probe-result"  # noqa: E731
        wrapped = _bind_is_available(target, probe)

        assert wrapped._is_available is probe
        assert wrapped(1, 2, key="value") == "result"
        assert recorded == [((1, 2), {"key": "value"})]
        assert wrapped() == "result"  # zero-arg delegation
        assert recorded[-1] == ((), {})

    @pytest.mark.platforms("thead")
    def test_wraps_preserves_function_metadata(self):
        from vllm_fl.dispatch.backends.vendor.thead.register_ops import (
            _bind_is_available,
        )

        def target_with_docs(positional, keyword=None):
            """Target docstring."""

        wrapped = _bind_is_available(target_with_docs, lambda: True)
        assert wrapped.__name__ == "target_with_docs"
        assert wrapped.__doc__ == "Target docstring."
