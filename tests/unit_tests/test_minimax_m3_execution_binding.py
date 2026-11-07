# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Exercise real Torch boundaries; mocks only inject provider behavior."""

import pytest
import torch

import vllm_fl.dispatch as dispatch
from vllm_fl.dispatch import CachedOp
from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import SelectionPolicy, policy_context
from vllm_fl.dispatch.types import BackendImplKind, OpImpl
from vllm_fl.ops.minimax_m3 import torch_ops


@pytest.mark.gpu
@pytest.mark.parametrize("fast", [False, True])
def test_started_operation_error_does_not_retry(monkeypatch, fast):
    monkeypatch.setattr(dispatch, "_OP_FAST_PATH_ENABLED", fast)
    calls = []

    def broken(x, limit, alpha, beta, out=None):
        calls.append("broken")
        out.fill_(7)
        raise RuntimeError("injected post-write error")

    def alternate(x, limit, alpha, beta, out=None):
        calls.append("alternate")
        out.fill_(9)

    manager = OpManager(register_builtins=False)
    manager.registry.register_many(
        [
            OpImpl("m3_swiglu", "test.broken", BackendImplKind.DEFAULT, broken),
            OpImpl("m3_swiglu", "test.alternate", BackendImplKind.DEFAULT, alternate),
        ]
    )
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
    binding = CachedOp("m3_swiglu", retry_on_error=False)
    monkeypatch.setattr(torch_ops, "_swiglu", binding)
    x = torch.ones((1, 8), device="cuda", dtype=torch.bfloat16)
    out = torch.zeros((1, 4), device="cuda", dtype=x.dtype)
    with (
        policy_context(
            SelectionPolicy.from_dict(
                strict=False,
                per_op_order={"m3_swiglu": ["impl:test.broken", "impl:test.alternate"]},
            )
        ),
        pytest.raises(RuntimeError, match="post-write"),
    ):
        torch_ops.swiglu(x, 7.0, 1.702, 1.0, out=out)
    torch.cuda.synchronize()
    assert calls == ["broken"]
    assert (out == 7).all()
    assert manager.get_failed_impls() == {}


@pytest.mark.gpu
def test_torch_boundary_obeys_explicit_provider_selection(monkeypatch):
    def one(x, limit, alpha, beta, out=None):
        out.fill_(1)

    def two(x, limit, alpha, beta, out=None):
        out.fill_(2)

    manager = OpManager(register_builtins=False)
    manager.registry.register_many(
        [
            OpImpl("m3_swiglu", "test.one", BackendImplKind.DEFAULT, one),
            OpImpl("m3_swiglu", "test.two", BackendImplKind.DEFAULT, two),
        ]
    )
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
    binding = CachedOp("m3_swiglu", retry_on_error=False)
    monkeypatch.setattr(torch_ops, "_swiglu", binding)
    x = torch.ones((1, 8), device="cuda", dtype=torch.bfloat16)
    out = torch.zeros((1, 4), device="cuda", dtype=x.dtype)
    for name, value in [("test.one", 1), ("test.two", 2), ("test.one", 1)]:
        with policy_context(
            SelectionPolicy.from_dict(per_op_order={"m3_swiglu": ["impl:" + name]})
        ):
            torch_ops.swiglu(x, 7.0, 1.702, 1.0, out=out)
            assert binding.inspect_binding().impl_id == name and (out == value).all()
