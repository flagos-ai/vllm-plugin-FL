# Copyright (c) 2026 BAAI. All rights reserved.

import inspect
from types import SimpleNamespace

import pytest
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
    assert "bias_vl" not in captured
    assert "image_sentinel_lo" not in captured
    assert torch.equal(ids, torch.tensor([[2, 4, 6, 7]], dtype=torch.int32))
    assert torch.equal(
        weights,
        torch.tensor([[0.25, 0.75, 1.25, 1.25]], dtype=torch.float32),
    )


def test_sqrtsoftplus_fallback_matches_dsv4_formula():
    import vllm_fl.ops.fused_moe.router as router_mod

    logits = torch.tensor([[0.0, 1.0, -1.0, 3.0]], dtype=torch.float32)
    bias = torch.tensor([0.4, 0.0, 0.8, -0.2], dtype=torch.float32)
    weights, ids = router_mod._sqrtsoftplus_topk(
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


def test_ascend_sqrtsoftplus_uses_torch_even_when_cuda_op_is_present(monkeypatch):
    import vllm_fl.ops.fused_moe.router as router_mod

    monkeypatch.setattr(router_mod, "_is_ascend_npu_tensor", lambda value: True)
    monkeypatch.setattr(
        router_mod,
        "fused_topk_bias",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Ascend must not call the CUDA router")
        ),
    )
    router = SimpleNamespace(
        e_score_correction_bias=None,
        top_k=1,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        _hash_indices_table=None,
        num_fused_shared_experts=0,
    )

    weights, ids = router_mod.FusedTopKBiasRouterFL._compute_routing(
        router,
        torch.zeros((1, 4)),
        torch.tensor([[0.0, 2.0]], dtype=torch.float32),
        torch.int32,
    )

    assert torch.equal(ids, torch.tensor([[1]], dtype=torch.int32))
    assert torch.equal(weights, torch.ones((1, 1), dtype=torch.float32))


def test_sqrtsoftplus_fallback_honors_hash_table():
    import vllm_fl.ops.fused_moe.router as router_mod

    weights, ids = router_mod._sqrtsoftplus_topk(
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


def test_bias_router_preserves_v032_image_routing_through_upstream(monkeypatch):
    import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as native

    import vllm_fl.ops.fused_moe.router as router_mod

    if "bias_vl" not in inspect.signature(native.fused_topk_bias).parameters:
        pytest.skip("Image-token router bias was introduced in vLLM 0.32")
    captured = {}

    def capture_kernel(*args, **kwargs):
        captured.update(kwargs)
        # The real upstream function passes its allocated outputs first.
        return args[0].fill_(0.5), args[1].fill_(2)

    monkeypatch.setattr(native.rocm_aiter_ops, "is_fused_moe_enabled", lambda: False)
    monkeypatch.setattr(native, "can_use_dsv4_topk", lambda *args: False)
    monkeypatch.setattr(native, "vllm_topk_softplus_sqrt", capture_kernel)
    monkeypatch.setattr(router_mod, "_is_ascend_npu_tensor", lambda value: False)
    image_bias = torch.nn.Parameter(torch.tensor([0.0, 0.0, 10.0]))
    router = SimpleNamespace(
        e_score_correction_bias=torch.nn.Parameter(torch.zeros(3)),
        bias_vl=image_bias,
        image_sentinel_lo=10,
        top_k=1,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=0.5,
        _hash_indices_table=None,
        num_fused_shared_experts=0,
    )
    weights, ids = router_mod.FusedTopKBiasRouterFL._compute_routing(
        router,
        torch.zeros((2, 4)),
        torch.zeros((2, 3)),
        torch.int32,
        input_ids=torch.tensor([10, 20]),
    )
    assert captured["image_sentinel_lo"] == 10
    assert captured["bias_vl"].data_ptr() == image_bias.data_ptr()
    torch.testing.assert_close(weights, torch.full((2, 1), 0.5))
    torch.testing.assert_close(ids, torch.full((2, 1), 2, dtype=torch.int32))
