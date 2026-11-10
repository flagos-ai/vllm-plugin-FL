# Copyright (c) 2026 BAAI. All rights reserved.

import os

import pytest

import vllm_fl.dispatch as dispatch
from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import reset_global_policy
from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, OpImpl


@pytest.fixture(autouse=True)
def isolated_dispatch(monkeypatch):
    registry = OpRegistry()
    manager = OpManager(registry=registry)
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()

    reset_global_policy()
    monkeypatch.setattr(dispatch, "_CACHED_OPS", [])
    monkeypatch.setattr(dispatch, "_OP_FAST_PATH_ENABLED", True)
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
    yield manager, registry
    reset_global_policy()


def register_impl(registry, op_name="test_op", fn=lambda value: value * 2):
    impl = OpImpl(
        op_name=op_name,
        impl_id=f"default.{op_name}",
        kind=BackendImplKind.DEFAULT,
        fn=fn,
    )
    registry.register_impl(impl)
    return impl


def test_prewarm_resolves_and_freezes_cached_op(isolated_dispatch):
    manager, registry = isolated_dispatch
    impl = register_impl(registry)
    cached = dispatch.CachedOp("test_op")

    assert dispatch.prewarm_cached_ops() == 1
    assert cached._impl is impl
    assert cached._frozen is True
    assert cached._manager_id == id(manager)
    assert cached._manager_epoch == manager.policy_epoch


def test_frozen_cached_op_exposes_only_impl_while_compiling(
    isolated_dispatch, monkeypatch
):
    manager, registry = isolated_dispatch
    register_impl(registry)
    cached = dispatch.CachedOp("test_op")
    dispatch.prewarm_cached_ops()

    monkeypatch.setattr(dispatch.torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        manager,
        "_resolve_impl",
        lambda *_args, **_kwargs: pytest.fail("manager entered during tracing"),
    )

    assert cached(3) == 6


def test_policy_epoch_change_unfreezes_runtime_cache(isolated_dispatch):
    manager, registry = isolated_dispatch
    register_impl(registry)
    cached = dispatch.CachedOp("test_op")
    dispatch.prewarm_cached_ops()

    manager.bump_policy_epoch()

    assert cached(4) == 8
    assert cached._frozen is False
    assert cached._manager_epoch == manager.policy_epoch


def test_prewarm_skips_unavailable_optional_ops(isolated_dispatch):
    cached = dispatch.CachedOp("missing_op")

    assert dispatch.prewarm_cached_ops() == 0
    assert cached._impl is None
    assert cached._frozen is False


def test_prewarm_does_not_freeze_when_io_dump_is_enabled(
    isolated_dispatch, monkeypatch
):
    _, registry = isolated_dispatch
    register_impl(registry)
    cached = dispatch.CachedOp("test_op")
    monkeypatch.setattr(dispatch, "is_dump_enabled", lambda: True)

    assert dispatch.prewarm_cached_ops() == 1
    assert cached._frozen is False
