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
)
from vllm_fl.dispatch.policy import policy_from_env


def _use_platform_config(monkeypatch, platform):
    import vllm_fl.dispatch.config as config

    path = Path(config.__file__).parent / f"{platform}.yaml"
    monkeypatch.setattr(config, "get_config_path", lambda: path)
    monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)


@pytest.mark.parametrize("override", [None, "", "0", "1"])
@pytest.mark.parametrize("platform", ["ascend", "nvidia"])
def test_strict_platform_default_and_environment_override(
    monkeypatch, platform, override
):
    _use_platform_config(monkeypatch, platform)
    if override is None:
        monkeypatch.delenv("VLLM_FL_STRICT", raising=False)
    else:
        monkeypatch.setenv("VLLM_FL_STRICT", override)
    expected = override == "1" if override else platform == "ascend"
    assert policy_from_env().strict is expected


@pytest.mark.parametrize("cached", [False, True])
def test_ascend_execution_error_does_not_retry_after_state_write(monkeypatch, cached):
    import vllm_fl.dispatch as dispatch

    _use_platform_config(monkeypatch, "ascend")
    monkeypatch.delenv("VLLM_FL_STRICT", raising=False)
    policy = policy_from_env()
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
