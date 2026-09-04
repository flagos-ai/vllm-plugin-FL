# Copyright 2026 FlagOS Contributors

import os

import pytest

from vllm_fl.dispatch import (
    BackendImplKind,
    CachedOp,
    OpImpl,
    SelectionPolicy,
    freeze_dispatch,
    get_default_manager,
    get_frozen_dispatch_manifest,
    is_dispatch_frozen,
    reset_default_manager,
    reset_global_policy,
    set_global_policy,
)


@pytest.fixture(autouse=True)
def _reset_dispatch_state():
    reset_default_manager()
    reset_global_policy()
    yield
    reset_default_manager()
    reset_global_policy()


def _initialized_manager():
    manager = get_default_manager()
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    return manager


def test_frozen_cached_op_never_reenters_manager(monkeypatch):
    manager = _initialized_manager()
    manager.registry.register_impl(
        OpImpl(
            op_name="unit_freeze",
            impl_id="default.unit",
            kind=BackendImplKind.DEFAULT,
            fn=lambda value: value + 1,
        )
    )
    cached_op = CachedOp("unit_freeze")

    manifest = freeze_dispatch(op_names={"unit_freeze"})
    assert is_dispatch_frozen()
    assert cached_op.frozen_impl_id == "default.unit"
    assert manifest.fingerprint
    assert manifest.selections[0].op_name == "unit_freeze"

    def fail_if_resolved_again(*args, **kwargs):
        raise AssertionError("frozen execution re-entered OpManager")

    monkeypatch.setattr(manager, "_resolve_impl", fail_if_resolved_again)
    assert cached_op(4) == 5


def test_policy_is_immutable_while_frozen():
    manager = _initialized_manager()
    manager.registry.register_impl(
        OpImpl(
            op_name="unit_policy_freeze",
            impl_id="default.unit",
            kind=BackendImplKind.DEFAULT,
            fn=lambda value: value,
        )
    )
    CachedOp("unit_policy_freeze")
    freeze_dispatch(op_names={"unit_policy_freeze"})

    with pytest.raises(RuntimeError, match="dispatch is frozen"):
        set_global_policy(SelectionPolicy(prefer="vendor"))


def test_unresolved_optional_op_fails_deterministically():
    cached_op = CachedOp("missing_optional_op")
    manifest = freeze_dispatch(
        op_names={"missing_optional_op"},
        strict=False,
    )

    assert get_frozen_dispatch_manifest() is manifest
    assert manifest.unresolved[0][0] == "missing_optional_op"
    with pytest.raises(RuntimeError, match="was not bound"):
        cached_op(1)
