# SPDX-License-Identifier: Apache-2.0
"""Small NPU-only checks for sqrtsoftplus/hash routing without a model."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(current_platform.device_type != "npu", reason="Requires Ascend"),
]


@pytest.fixture(autouse=True)
def strict_reference():
    from vllm_fl.dispatch import get_default_manager, with_strict_mode

    with with_strict_mode():
        manager = get_default_manager()
        assert manager.get_selected_impl_id("topk_softplus_sqrt") == "reference.torch"
        yield


def _inputs(dtype):
    logits = torch.tensor(
        [[-2.0, 0.3, 1.5, -0.4], [1.2, -1.0, 0.5, 2.3], [0.1, 1.7, -2.4, 0.8]],
        device="npu",
        dtype=dtype,
    )
    hidden = torch.zeros(3, 8, device="npu", dtype=dtype)
    bias = torch.tensor([2.5, 0.1, -0.2, 0.6], device="npu", dtype=torch.float32)
    tokens = torch.tensor([-1, 2, 1], device="npu", dtype=torch.int64)
    table = torch.tensor([[3, 1], [0, 2], [2, 3]], device="npu", dtype=torch.int32)
    return hidden, logits, bias, tokens, table


def _oracle(logits, bias, renormalize, tokens=None, table=None):
    # Independently form softplus with logaddexp, avoiding provider helpers.
    scores = torch.logaddexp(logits.float(), torch.zeros_like(logits.float())).sqrt()
    if table is None:
        selection = scores if bias is None else scores + bias.float()
        ids = selection.argsort(dim=-1, descending=True)[:, :2]
    else:
        safe_tokens = torch.where(tokens == -1, 0, tokens).long()
        ids = table[safe_tokens].long()
    weights = scores.gather(1, ids)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return weights, ids


@pytest.mark.parametrize(
    "dtype,index_dtype,hashed",
    [
        (torch.bfloat16, torch.int32, False),
        (torch.float16, torch.int64, False),
        (torch.bfloat16, torch.int64, True),
        (torch.float16, torch.int32, True),
    ],
)
@pytest.mark.parametrize("renormalize", [False, True])
def test_biased_and_hash_weights(dtype, index_dtype, hashed, renormalize):
    from vllm_fl.dispatch import call_op

    hidden, logits, bias, tokens, table = _inputs(dtype)
    token_arg, table_arg = (tokens, table) if hashed else (None, None)
    weights, ids = call_op(
        "topk_softplus_sqrt",
        hidden,
        logits,
        2,
        renormalize,
        e_score_correction_bias=bias,
        indices_type=index_dtype,
        input_tokens=token_arg,
        hash_indices_table=table_arg,
    )
    expected, expected_ids = _oracle(logits, bias, renormalize, token_arg, table_arg)
    assert weights.dtype == torch.float32
    assert ids.dtype == index_dtype
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(ids.long(), expected_ids)
    torch.testing.assert_close(weights, expected, rtol=1e-6, atol=1e-6)
    if renormalize:
        torch.testing.assert_close(weights.sum(-1), torch.ones(3, device="npu"))
    if hashed:
        torch.testing.assert_close(ids[0].long(), table[0].long())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hashed", [False, True])
def test_actual_router_scales_once(dtype, hashed):
    from vllm_fl.ops.fused_moe.router import FusedTopKBiasRouterFL

    hidden, logits, bias, tokens, table = _inputs(dtype)
    router = FusedTopKBiasRouterFL(
        top_k=2,
        global_num_experts=4,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=2.5,
        scoring_func="sqrtsoftplus",
        hash_indices_table=table if hashed else None,
    )
    token_arg = tokens if hashed else None
    weights, ids = router._compute_routing(
        hidden, logits, torch.int64, input_ids=token_arg
    )
    expected, expected_ids = _oracle(
        logits, bias, True, token_arg, table if hashed else None
    )
    torch.testing.assert_close(weights, expected * 2.5, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ids.long(), expected_ids)
    torch.testing.assert_close(weights.sum(-1), torch.full((3,), 2.5, device="npu"))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_changed_logits_and_tokens_graph(dtype):
    import torch_npu  # noqa: F401

    from vllm_fl.ops.fused_moe.router import FusedTopKBiasRouterFL

    hidden, logits, bias, tokens, table = _inputs(dtype)
    router = FusedTopKBiasRouterFL(
        top_k=2,
        global_num_experts=4,
        e_score_correction_bias=bias,
        renormalize=True,
        routed_scaling_factor=2.5,
        scoring_func="sqrtsoftplus",
        hash_indices_table=table,
    )

    def run():
        return router._compute_routing(hidden, logits, torch.int32, input_ids=tokens)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        weights, ids = run()
    for step in range(2):
        logits.add_(torch.tensor([0.4, -0.2, 0.1, -0.3], device="npu", dtype=dtype))
        tokens.copy_(torch.tensor([step + 1, 1 - step, -1], device="npu"))
        expected, expected_ids = _oracle(logits, bias, True, tokens, table)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(weights, expected * 2.5, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(ids.long(), expected_ids)


def test_empty_rows():
    from vllm_fl.dispatch import call_op

    weights, ids = call_op(
        "topk_softplus_sqrt",
        torch.empty(0, 8, device="npu"),
        torch.empty(0, 4, device="npu", dtype=torch.bfloat16),
        2,
        True,
        indices_type=torch.int64,
    )
    assert weights.shape == ids.shape == (0, 2)
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int64


@pytest.mark.parametrize("missing", ["tokens", "table"])
def test_hash_arguments_require_pair(missing):
    from vllm_fl.dispatch import call_op

    hidden, logits, bias, tokens, table = _inputs(torch.bfloat16)
    with pytest.raises(ValueError):
        call_op(
            "topk_softplus_sqrt",
            hidden,
            logits,
            2,
            True,
            e_score_correction_bias=bias,
            input_tokens=None if missing == "tokens" else tokens,
            hash_indices_table=None if missing == "table" else table,
        )
