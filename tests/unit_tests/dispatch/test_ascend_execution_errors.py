# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

from pathlib import Path

import pytest
import torch

from vllm_fl.dispatch import (
    BackendImplKind,
    CachedOp,
    OpImpl,
    OpManager,
    OpRegistry,
    policy_context,
    policy_from_config,
)


@pytest.mark.parametrize("cached", [False, True])
def test_ascend_execution_error_does_not_retry_after_state_write(monkeypatch, cached):
    import vllm_fl.dispatch as dispatch

    config = Path(dispatch.__file__).parent / "config" / "ascend.yaml"
    policy = policy_from_config(str(config))
    assert policy.strict
    registry = OpRegistry()
    state = torch.zeros(1)
    calls = []

    def update_then_fail(value):
        calls.append("primary")
        value.add_(1)
        raise RuntimeError("state update failed")

    def alternative(value):
        calls.append("alternative")
        return value.add_(1)

    for name, kind, fn in (
        ("flagos.primary", BackendImplKind.DEFAULT, update_then_fail),
        ("reference.alternative", BackendImplKind.REFERENCE, alternative),
    ):
        registry.register_impl(
            OpImpl(op_name="state_update", impl_id=name, kind=kind, fn=fn)
        )
    manager = OpManager(registry=registry)
    monkeypatch.setattr(manager, "ensure_initialized", lambda: None)
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
    call = (
        CachedOp("state_update")
        if cached
        else lambda value: manager.call("state_update", value)
    )
    with (
        policy_context(policy),
        pytest.raises(RuntimeError, match="state update failed"),
    ):
        call(state)
    assert calls == ["primary"]
    torch.testing.assert_close(state, torch.ones(1))
