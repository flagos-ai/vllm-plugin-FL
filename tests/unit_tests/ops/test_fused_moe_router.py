# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import torch
import torch.nn.functional as F


def test_bias_router_preserves_v028_hash_scaling_and_shared_experts(monkeypatch):
    import vllm_fl.ops.fused_moe.router as router_mod

    captured = {}
    routed_weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    routed_ids = torch.tensor([[2, 4]], dtype=torch.int32)

    def fake_fused_topk_bias(**kwargs):
        captured.update(kwargs)
        return routed_weights.clone(), routed_ids.clone()

    monkeypatch.setattr(router_mod, "fused_topk_bias", fake_fused_topk_bias)
    monkeypatch.setattr(router_mod, "_has_dsv4_topk_op", lambda: True)

    hash_table = torch.tensor([[2, 4]], dtype=torch.int32)
    input_ids = torch.tensor([0], dtype=torch.int32)
    bias = torch.nn.Parameter(torch.zeros(6))
    router = SimpleNamespace(
        e_score_correction_bias=bias,
        top_k=2,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=0.5,
        _hash_indices_table=hash_table,
        num_fused_shared_experts=2,
        shared_expert_weight=1.25,
        global_num_experts=6,
    )

    weights, ids = router_mod.FusedTopKBiasRouterFL._compute_routing(
        router,
        torch.zeros((1, 4)),
        torch.zeros((1, 6)),
        torch.int32,
        input_ids=input_ids,
    )

    assert captured["input_tokens"] is input_ids
    assert captured["hash_indices_table"] is hash_table
    assert captured["routed_scaling_factor"] == 0.5
    assert torch.equal(ids, torch.tensor([[2, 4, 6, 7]], dtype=torch.int32))
    assert torch.equal(
        weights,
        torch.tensor([[0.25, 0.75, 1.25, 1.25]], dtype=torch.float32),
    )


def test_sqrtsoftplus_fallback_matches_dsv4_formula():
    from vllm_fl.ops.fused_moe.router import _sqrtsoftplus_topk

    logits = torch.tensor([[0.0, 1.0, -1.0, 3.0]], dtype=torch.float32)
    bias = torch.tensor([0.4, 0.0, 0.8, -0.2], dtype=torch.float32)
    weights, ids = _sqrtsoftplus_topk(
        gating_output=logits,
        e_score_correction_bias=bias,
        topk=2,
        renormalize=True,
        indices_type=torch.int64,
        input_tokens=None,
        hash_indices_table=None,
        routed_scaling_factor=0.5,
    )

    scores = torch.sqrt(F.softplus(logits))
    expected_ids = torch.topk(scores + bias, k=2, dim=-1, sorted=False).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(-1, keepdim=True) * 0.5
    assert torch.equal(ids, expected_ids)
    assert torch.allclose(weights, expected_weights)


def test_sqrtsoftplus_fallback_honors_hash_table():
    from vllm_fl.ops.fused_moe.router import _sqrtsoftplus_topk

    weights, ids = _sqrtsoftplus_topk(
        gating_output=torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32),
        e_score_correction_bias=torch.zeros(3),
        topk=2,
        renormalize=False,
        indices_type=torch.int32,
        input_tokens=torch.tensor([1], dtype=torch.int64),
        hash_indices_table=torch.tensor([[0, 1], [2, 0]], dtype=torch.int32),
        routed_scaling_factor=2.0,
    )

    expected = torch.sqrt(F.softplus(torch.tensor([2.0, 0.0]))) * 2.0
    assert torch.equal(ids, torch.tensor([[2, 0]], dtype=torch.int32))
    assert torch.allclose(weights[0], expected)
