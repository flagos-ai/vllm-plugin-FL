"""Regression tests for EMPTY-vLLM 0.32 routing semantics."""

import importlib

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.iluvatar.moe_router import (
    patch_moe_router,
    topk_softmax,
)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_router_padding_bias_and_return(monkeypatch, bias, normalize):
    router = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.router.fused_topk_router"
    )
    mask = torch.tensor([False, True, False])
    monkeypatch.setattr(router, "_get_padding_mask", lambda n: mask)
    logits = torch.tensor(
        [[1.0, 3.0, 2.0, 0.0], [0.0, 1.0, 3.0, 2.0], [3.0, 2.0, 0.0, 1.0]]
    )
    correction = torch.tensor([0.0, 0.0, 0.0, 2.0]) if bias else None
    scores = logits.softmax(-1)
    expected_ids = (scores + (correction if bias else 0)).topk(2).indices
    expected_weights = scores.gather(1, expected_ids)
    if normalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)

    def dispatch(name, weights, ids, source, gating, renormalize):
        assert name == "topk_softmax"
        values, indices = gating.softmax(-1).topk(2)
        ids.copy_(indices)
        source.copy_(torch.arange(3)[:, None] + torch.arange(2) * 3)
        return (values / values.sum(-1, keepdim=True) if renormalize else values), ids

    monkeypatch.setattr("vllm_fl.dispatch.call_op", dispatch)
    weights = torch.empty(3, 2)
    ids = torch.empty(3, 2, dtype=torch.int32)
    source = torch.empty_like(ids)
    result = topk_softmax(weights, ids, source, logits, normalize, correction)
    assert result[0] is weights and result[1] is ids
    torch.testing.assert_close(weights, expected_weights)
    expected_ids[1] = -1
    torch.testing.assert_close(ids.long(), expected_ids)
    torch.testing.assert_close(
        source.long(), torch.arange(3)[:, None] + torch.arange(2) * 3
    )


def test_patch_idempotent():
    patch_moe_router()
    patch_moe_router()
    for name in ("fused_topk_router", "fused_topk_bias_router"):
        module = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.router." + name
        )
        assert module.vllm_topk_softmax is topk_softmax
