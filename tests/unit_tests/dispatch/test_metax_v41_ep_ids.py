# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native EP extra-token transport must match hidden/logit row order."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.metax.patches.deepseek_v41 import (
    ep_router_ids as patch,
)


@pytest.mark.parametrize("mode", [patch.CUDAGraphMode.NONE, patch.CUDAGraphMode.FULL])
def test_image_ids_follow_dispatched_rows(monkeypatch, mode):
    calls = []
    local_ids = torch.tensor([129264, 42])
    other_ids = torch.tensor([17, 129264, 99])
    hidden = torch.arange(6).view(2, 3).float()
    other = torch.arange(9).view(3, 3).float() + 20

    def dispatch(x, logits, sequence_parallel, extra_tensors):
        calls.append("dispatch")
        torch.testing.assert_close(extra_tensors[0], local_ids)
        return (
            torch.cat([x, other]),
            torch.cat([logits, other * 2]),
            [torch.cat([local_ids, other_ids])],
        )

    monkeypatch.setattr(
        patch,
        "get_ep_group",
        lambda: SimpleNamespace(
            dispatch_router_logits=dispatch, barrier=lambda: calls.append("fence")
        ),
    )
    monkeypatch.setattr(
        patch,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=mode),
    )
    method = type("MetaXCompressedTensorsW4A8Method", (), {})()

    def quant(**kwargs):
        calls.append("quant")
        torch.testing.assert_close(
            kwargs["input_ids"], torch.tensor([129264, 42, 17, 129264, 99])
        )
        assert kwargs["hidden_states"].shape[0] == kwargs["input_ids"].numel()
        assert (kwargs["input_ids"] == 129264).nonzero().flatten().tolist() == [0, 3]
        return None, kwargs["hidden_states"]

    runner = SimpleNamespace(
        _quant_method=method,
        do_naive_dispatch_combine=True,
        router=SimpleNamespace(
            bias_vl=torch.ones(4), image_sentinel_lo=129264, global_num_experts=384
        ),
        moe_config=SimpleNamespace(pcp_size=1, is_sequence_parallel=False),
        _fse_fuse_gate=False,
        routed_experts=SimpleNamespace(
            _ensure_moe_quant_config_init=lambda: calls.append("config")
        ),
        _shared_experts=None,
        gate=lambda x: (x * 2, None),
        _sequence_parallel_context=nullcontext,
        _apply_quant_method=quant,
        _maybe_combine=lambda shared, x: x,
    )
    wrapped = patch._wrap_forward_impl(lambda *args: pytest.fail("wrong fallback"))
    output = wrapped(runner, hidden, hidden, None, local_ids)
    torch.testing.assert_close(output, torch.cat([hidden, other]))
    expected = ["config", "dispatch", "quant"]
    if mode == patch.CUDAGraphMode.NONE:
        expected = ["config", "fence", "dispatch", "quant", "fence"]
    assert calls == expected
    runner.router.image_sentinel_lo = 0
    assert (
        patch._wrap_forward_impl(lambda *args: "original")(
            runner, hidden, hidden, None, local_ids
        )
        == "original"
    )


def test_unsupported_scope_fails_before_collective(monkeypatch):
    runner = SimpleNamespace(
        _quant_method=type("MetaXCompressedTensorsW4A8Method", (), {})(),
        do_naive_dispatch_combine=True,
        router=SimpleNamespace(bias_vl=torch.ones(4), image_sentinel_lo=129264),
        moe_config=SimpleNamespace(pcp_size=2, is_sequence_parallel=False),
        _fse_fuse_gate=False,
    )
    wrapped = patch._wrap_forward_impl(lambda *args: None)
    with pytest.raises(NotImplementedError, match="PCP1"):
        wrapped(
            runner,
            torch.zeros(1, 3),
            torch.zeros(1, 4),
            None,
            torch.zeros(1, dtype=torch.int64),
        )


@pytest.mark.parametrize("mode", [patch.CUDAGraphMode.NONE, patch.CUDAGraphMode.FULL])
def test_draft_preparation_fences_preserve_native_id_free_dispatch(monkeypatch, mode):
    calls = []
    monkeypatch.setattr(
        patch,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=mode),
    )
    monkeypatch.setattr(
        patch,
        "get_ep_group",
        lambda: SimpleNamespace(barrier=lambda: calls.append("fence")),
    )

    def dispatch(x, logits):
        calls.append("dispatch")
        return x, logits

    def compute(**kwargs):
        calls.append("quant")
        assert kwargs["input_ids"] is None
        return None, kwargs["hidden_states"]

    runner = SimpleNamespace(
        _quant_method=type("MetaXCompressedTensorsW4A8Method", (), {})(),
        do_naive_dispatch_combine=True,
        router=SimpleNamespace(
            bias_vl=torch.ones(128), image_sentinel_lo=0, global_num_experts=128
        ),
        moe_config=SimpleNamespace(
            hidden_dim=5120, pcp_size=1, is_sequence_parallel=False
        ),
        _fse_fuse_gate=False,
        routed_experts=SimpleNamespace(_ensure_moe_quant_config_init=lambda: None),
        _shared_experts=None,
        gate=None,
        _sequence_parallel_context=nullcontext,
        _maybe_dispatch=dispatch,
        _apply_quant_method=compute,
        _maybe_combine=lambda shared, x: x,
    )
    x = torch.randn(2, 3)
    output = patch._wrap_forward_impl(lambda *args: pytest.fail("wrong fallback"))(
        runner, x, x, None
    )
    torch.testing.assert_close(output, x)
    assert calls == (
        ["fence", "dispatch", "quant", "fence"]
        if mode == patch.CUDAGraphMode.NONE
        else ["dispatch", "quant"]
    )
