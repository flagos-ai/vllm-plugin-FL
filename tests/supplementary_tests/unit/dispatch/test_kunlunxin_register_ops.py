# Copyright 2025 BAAI. All rights reserved.

"""CPU contracts for Kunlunxin's ``register_ops`` dispatch registrations.

``register_builtins`` builds ``OpImpl`` records for every operator the
Kunlunxin vendor backend ships.  These tests pin:

* the registered set: operator names, ``impl_id`` prefix, ``kind`` = VENDOR,
  ``vendor`` name and the VENDOR priority (values read from
  ``vllm_fl.dispatch.types`` rather than hard-coded, so a renumbered
  priority still fails only when the intent actually changes);
* ``_bind_is_available``'s wrapper semantics: delegation, ``__name__``
  preservation, and the ``_is_available`` attribute used by
  ``OpImpl.is_available()``;
* ``OpImpl.is_available()`` semantics for True / False / raising probes.

Everything runs on CPU against a fresh ``OpRegistry``; no vendor runtime is
imported because the bound functions are only constructed, never called.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = [pytest.mark.supplementary]

from vllm_fl.dispatch.backends.vendor.kunlunxin.register_ops import (
    _bind_is_available,
    register_builtins,
)
from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

#: Operator names register_builtins must produce, pinned from the source.
EXPECTED_OPS = {
    "silu_and_mul",
    "rms_norm",
    "rotary_embedding",
    "attention_backend",
    "chunk_gated_delta_rule_fwd",
    "fused_recurrent_gated_delta_rule_fwd",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "fused_gdn_gating",
    "fused_experts",
    "topk_softmax",
    "grouped_topk",
}


@pytest.fixture
def registered_impls():
    """Fresh registry with kunlunxin builtins registered; list of all impls."""
    registry = OpRegistry()
    register_builtins(registry)
    snapshot = registry.snapshot()
    impls = [
        impl
        for impls in snapshot.impls_by_op.values()
        for impl in impls
    ]
    return registry, impls


@pytest.mark.platforms("kunlunxin")
def test_registers_every_builtin_operator(registered_impls):
    _, impls = registered_impls
    assert {impl.op_name for impl in impls} == EXPECTED_OPS
    assert len(impls) == len(EXPECTED_OPS)


@pytest.mark.platforms("kunlunxin")
def test_all_impls_carry_vendor_metadata(registered_impls):
    _, impls = registered_impls
    for impl in impls:
        assert impl.impl_id == "vendor.kunlunxin"
        assert impl.kind is BackendImplKind.VENDOR
        assert impl.vendor == "kunlunxin"
        assert impl.priority == BackendPriority.VENDOR


@pytest.mark.platforms("kunlunxin")
def test_registration_survives_a_second_registry(registered_impls):
    # register_builtins must be repeatable against a fresh registry (the
    # dispatch manager constructs one per process) without duplicate-id
    # errors leaking state between them.
    registry = OpRegistry()
    register_builtins(registry)
    snapshot = registry.snapshot()
    impls = [i for v in snapshot.impls_by_op.values() for i in v]
    assert {impl.op_name for impl in impls} == EXPECTED_OPS


@pytest.mark.platforms("kunlunxin")
def test_every_impl_fn_is_the_bound_backend_method(registered_impls):
    # Each OpImpl wraps the KunlunxinBackend instance built inside
    # register_builtins; functools.wraps keeps the original method name.
    registry, impls = registered_impls
    for impl in impls:
        assert impl.fn.__name__ == impl.op_name, (
            f"{impl.op_name}: wrapper lost the method name"
        )
        # The wrapper must be callable with arbitrary forwarding (delegation
        # itself is covered by the backend-class tests), and carry the probe.
        assert hasattr(impl.fn, "_is_available")


@pytest.mark.platforms("kunlunxin")
def test_registered_impl_is_resolvable_by_id(registered_impls):
    registry, _ = registered_impls
    impl = registry.get_implementation("silu_and_mul", "vendor.kunlunxin")
    assert impl is not None
    assert impl.op_name == "silu_and_mul"
    assert impl in registry.get_implementations("silu_and_mul")
    # and a lookup for another vendor id resolves to nothing
    assert registry.get_implementation("silu_and_mul", "vendor.cuda") is None


class TestBindIsAvailable:
    """_bind_is_available(fn, probe) -> wrapper with fn's behavior + probe."""

    def _target(self):
        def recorded(a, b=0, *, c=None):
            """docstring that must survive"""
            return (a, b, c)

        return recorded

    def test_wrapper_delegates_arguments_and_result(self):
        calls = []

        def target(*args, **kwargs):
            calls.append((args, kwargs))
            return "result"

        wrapped = _bind_is_available(target, lambda: True)
        assert wrapped(1, 2, c=3) == "result"
        assert calls == [((1, 2), {"c": 3})]

    def test_wraps_metadata(self):
        target = self._target()
        wrapped = _bind_is_available(target, lambda: False)
        assert wrapped.__name__ == target.__name__
        assert wrapped.__doc__ == target.__doc__

    def test_original_function_is_left_untouched(self):
        target = self._target()
        _bind_is_available(target, lambda: True)
        assert not hasattr(target, "_is_available")
        assert target(1) == (1, 0, None)

    def test_binds_the_probe(self):
        probe = lambda: True  # noqa: E731
        wrapped = _bind_is_available(self._target(), probe)
        assert wrapped._is_available is probe


class TestOpImplIsAvailable:
    """OpImpl.is_available() honors the _is_available attribute on fn."""

    def _impl_for(self, fn):
        return OpImpl(
            op_name="op",
            impl_id="vendor.kunlunxin",
            kind=BackendImplKind.VENDOR,
            fn=fn,
            vendor="kunlunxin",
            priority=BackendPriority.VENDOR,
        )

    def test_true_when_probe_returns_true(self):
        impl = self._impl_for(_bind_is_available(lambda *a: None, lambda: True))
        assert impl.is_available() is True

    def test_false_when_probe_returns_false(self):
        impl = self._impl_for(_bind_is_available(lambda *a: None, lambda: False))
        assert impl.is_available() is False

    def test_false_when_probe_raises(self):
        def exploding():
            raise RuntimeError("hardware probe failed")

        impl = self._impl_for(_bind_is_available(lambda *a: None, exploding))
        assert impl.is_available() is False

    def test_truthy_probe_results_are_coerced(self):
        # bool(avail_fn()) is applied, so non-bool truthy results count.
        impl = self._impl_for(_bind_is_available(lambda *a: None, lambda: 1))
        assert impl.is_available() is True

    def test_defaults_to_available_without_probe(self):
        # The OpImpl contract: a fn without _is_available is unconditionally
        # available (the default flag every other kind of impl relies on).
        impl = self._impl_for(lambda *a: None)
        assert impl.is_available() is True


class TestRegisterOpsSourceContract:
    """register_builtins binds every op through _bind_is_available (AST check).

    Guards against a future edit registering a bare method: the resulting
    impl would then claim to be available on every platform, including hosts
    without Kunlunxin hardware.
    """

    @pytest.mark.platforms("kunlunxin")
    def test_every_registration_is_bound(self):
        source = inspect.getsource(register_builtins)
        # one OpImpl( per expected op
        assert source.count("OpImpl(") == len(EXPECTED_OPS)
        assert source.count("_bind_is_available(") == len(EXPECTED_OPS)
        # every binding uses the same backend availability probe
        assert source.count("is_avail)") == len(EXPECTED_OPS)
