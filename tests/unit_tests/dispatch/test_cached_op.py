# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for CachedOp fast-path dispatch behavior."""

import os

import pytest

import vllm_fl.dispatch as dispatch
import vllm_fl.dispatch.manager as manager_mod
from vllm_fl.dispatch import CachedOp, get_default_manager, reset_default_manager
from vllm_fl.dispatch.policy import SelectionPolicy, reset_global_policy, set_global_policy
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


@pytest.fixture(autouse=True)
def reset_dispatch_state():
    reset_default_manager()
    reset_global_policy()
    set_global_policy(SelectionPolicy())
    yield
    reset_default_manager()
    reset_global_policy()


def _register_impl(op_name, impl_id, kind, fn, *, priority, vendor=None):
    impl = OpImpl(
        op_name=op_name,
        impl_id=impl_id,
        kind=kind,
        fn=fn,
        priority=priority,
        vendor=vendor,
    )
    manager = get_default_manager()
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    manager.registry.register_impl(impl)
    return impl


def test_cached_op_routes_through_manager_when_dump_enabled(monkeypatch):
    op_name = "cached_dump_op"
    _register_impl(
        op_name,
        "default.cached_dump",
        BackendImplKind.DEFAULT,
        lambda x: x + 1,
        priority=BackendPriority.DEFAULT,
    )

    mgr = get_default_manager()
    original_call = mgr.call
    calls = []

    def call_spy(name, *args, **kwargs):
        calls.append(name)
        return original_call(name, *args, **kwargs)

    monkeypatch.setattr(dispatch, "is_dump_enabled", lambda: True)
    monkeypatch.setattr(mgr, "call", call_spy)

    assert CachedOp(op_name)(3) == 4
    assert calls == [op_name]


def test_cached_op_records_first_use_and_flagos_oplist(monkeypatch):
    op_name = "cached_first_use_op"
    _register_impl(
        op_name,
        "default.cached_first_use",
        BackendImplKind.DEFAULT,
        lambda x: x * 2,
        priority=BackendPriority.DEFAULT,
    )

    recorded = []
    monkeypatch.setattr(
        manager_mod,
        "_record_default_flagos_op",
        lambda name, impl: recorded.append((name, impl.impl_id)),
    )

    op = CachedOp(op_name)
    assert op(2) == 4
    assert op(3) == 6

    mgr = get_default_manager()
    assert mgr._called_ops[op_name] == "default.cached_first_use"
    assert recorded == [(op_name, "default.cached_first_use")]


def test_cached_op_uses_current_default_manager_after_reset():
    op_name = "cached_reset_op"
    _register_impl(
        op_name,
        "default.cached_reset.first",
        BackendImplKind.DEFAULT,
        lambda x: x + 1,
        priority=BackendPriority.DEFAULT,
    )

    op = CachedOp(op_name)
    assert op(1) == 2

    reset_default_manager()
    _register_impl(
        op_name,
        "default.cached_reset.second",
        BackendImplKind.DEFAULT,
        lambda x: x + 10,
        priority=BackendPriority.DEFAULT,
    )

    assert op(1) == 11


def test_cached_op_failure_sticks_to_manager_fallback_until_epoch_changes():
    op_name = "cached_fallback_op"
    calls = {"bad": 0, "good": 0}

    def bad_impl(x):
        calls["bad"] += 1
        raise RuntimeError("boom")

    def good_impl(x):
        calls["good"] += 1
        return x + 100

    _register_impl(
        op_name,
        "default.cached_fallback.bad",
        BackendImplKind.DEFAULT,
        bad_impl,
        priority=BackendPriority.DEFAULT,
    )
    _register_impl(
        op_name,
        "reference.cached_fallback.good",
        BackendImplKind.REFERENCE,
        good_impl,
        priority=BackendPriority.REFERENCE,
    )

    op = CachedOp(op_name)

    assert op(1) == 101
    assert op(2) == 102
    assert calls == {"bad": 1, "good": 2}
