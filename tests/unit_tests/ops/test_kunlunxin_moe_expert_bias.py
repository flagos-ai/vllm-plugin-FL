# Copyright (c) 2026 BAAI. All rights reserved.

"""Nonzero expert biases must survive the Kunlunxin fused-MoE fast path.

``w1_bias``/``w2_bias`` are optional, so a fast path that forgets to forward
them produces output that is silently identical to the bias-free result -- no
shape or dtype assertion can see it. The down-projection identity below is
exact, so it pins the values rather than only their presence.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


def _kunlunxin_available() -> bool:
    try:
        import xtorch_ops  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


requires_kunlunxin = pytest.mark.skipif(
    not _kunlunxin_available(),
    reason="Kunlunxin device or xtorch_ops not available",
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
E, TOPK, H, I, T = 4, 2, 64, 32, 8


def _setup():
    hidden_states = torch.randn(T, H, dtype=DTYPE, device=DEVICE)
    w1 = torch.randn(E, 2 * I, H, dtype=DTYPE, device=DEVICE) * 0.05
    w2 = torch.randn(E, H, I, dtype=DTYPE, device=DEVICE) * 0.05
    topk_weights = torch.rand(T, TOPK, dtype=torch.float32, device=DEVICE)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).to(DTYPE)
    topk_ids = torch.stack([torch.randperm(E, device=DEVICE)[:TOPK] for _ in range(T)])
    return hidden_states, w1, w2, topk_weights, topk_ids


def _run(hidden_states, w1, w2, topk_weights, topk_ids, **biases):
    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fused_moe.fused_moe import (
        fused_experts_impl,
    )

    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation="silu",
        global_num_experts=E,
        **biases,
    )


@requires_kunlunxin
def test_w2_bias_shifts_output_by_router_weighted_bias():
    """The down-projection bias sits outside the activation, so its effect is exact."""
    hidden_states, w1, w2, topk_weights, topk_ids = _setup()
    w2_bias = torch.randn(E, H, dtype=DTYPE, device=DEVICE) * 0.5

    without = _run(hidden_states, w1, w2, topk_weights, topk_ids)
    biased = _run(hidden_states, w1, w2, topk_weights, topk_ids, w2_bias=w2_bias)

    expected = (topk_weights.float().unsqueeze(-1) * w2_bias.float()[topk_ids]).sum(1)
    observed = biased.float() - without.float()

    assert torch.allclose(observed, expected, atol=5e-2, rtol=5e-2)


@requires_kunlunxin
def test_w1_bias_is_consumed():
    """A dropped gate/up bias leaves the output bit-identical; a used one moves it."""
    hidden_states, w1, w2, topk_weights, topk_ids = _setup()

    baseline = _run(hidden_states, w1, w2, topk_weights, topk_ids)
    zero_bias = _run(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_bias=torch.zeros(E, 2 * I, dtype=DTYPE, device=DEVICE),
    )
    assert torch.allclose(baseline.float(), zero_bias.float(), atol=1e-3)

    w1_bias = torch.randn(E, 2 * I, dtype=DTYPE, device=DEVICE) * 0.5
    biased = _run(hidden_states, w1, w2, topk_weights, topk_ids, w1_bias=w1_bias)

    assert not torch.allclose(baseline.float(), biased.float(), atol=1e-2)


@pytest.mark.parametrize(
    "w1_bias, w2_bias", [(True, True), (True, False), (False, True)]
)
def test_apply_forwards_expert_biases_to_the_kunlunxin_kernel(w1_bias, w2_bias):
    """Hardware-free: the fast path must hand both biases to fused_experts_impl."""
    from vllm_fl.ops.fused_moe import fused_moe_utils

    experts = object.__new__(fused_moe_utils.TritonExpertsFL)
    experts._lora_context = None
    experts.quant_config = MagicMock()
    experts.per_act_token_quant = False
    experts.w1_scale = experts.w2_scale = None
    experts.block_shape = None
    experts.w1_bias = torch.zeros(E, 2 * I) if w1_bias else None
    experts.w2_bias = torch.zeros(E, H) if w2_bias else None

    hidden_states = torch.zeros(T, H)
    output = torch.zeros(T, H)
    sentinel = torch.ones(T, H)
    spy = MagicMock(return_value=sentinel)

    with (
        patch.object(fused_moe_utils, "get_platform_name", return_value="kunlunxin"),
        patch("vllm_fl.ops.fused_moe.fused_moe.fused_experts_impl", spy),
    ):
        experts.apply(
            output=output,
            hidden_states=hidden_states,
            w1=torch.zeros(E, 2 * I, H),
            w2=torch.zeros(E, H, I),
            topk_weights=torch.zeros(T, TOPK),
            topk_ids=torch.zeros(T, TOPK, dtype=torch.long),
            activation=MagicMock(value="silu"),
            global_num_experts=E,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )

    assert spy.call_args.kwargs["w1_bias"] is experts.w1_bias
    assert spy.call_args.kwargs["w2_bias"] is experts.w2_bias
