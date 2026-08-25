# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import torch


def test_bias_router_preserves_v028_hash_scaling_and_shared_experts(monkeypatch):
    import vllm_fl.ops.fused_moe.router as router_mod

    captured = {}
    routed_weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    routed_ids = torch.tensor([[2, 4]], dtype=torch.int32)

    def fake_fused_topk_bias(**kwargs):
        captured.update(kwargs)
        return routed_weights.clone(), routed_ids.clone()

    monkeypatch.setattr(router_mod, "fused_topk_bias", fake_fused_topk_bias)

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
