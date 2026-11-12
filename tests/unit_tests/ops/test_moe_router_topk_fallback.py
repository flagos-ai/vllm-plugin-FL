# Copyright (c) 2026 BAAI. All rights reserved.

"""Regression tests for the no-_moe_C top-k fallbacks.

vLLM's FusedTopKBiasRouter calls torch.ops._moe_C.topk_sigmoid/topk_softmax.
Builds without the compiled extension (several Ascend images ship
vllm 0.28.0+empty) raise AttributeError during MoE init. The FL router must
fall back to pure torch for BOTH scoring functions.
"""

import torch

from vllm_fl.ops.fused_moe import router


def _reference_topk(gating, bias, topk, renormalize, scoring):
    if scoring == "sigmoid":
        scores = torch.sigmoid(gating.float())
    else:
        scores = torch.softmax(gating.float(), dim=-1)
    choice = scores
    if bias is not None:
        choice = choice + bias.float()
    choice = torch.nan_to_num(choice, nan=-1e30)
    ids = torch.topk(choice, k=topk, dim=-1, sorted=False).indices
    weights = scores.gather(1, ids)
    if renormalize:
        denominator = weights.sum(dim=-1, keepdim=True)
        weights = weights / torch.where(
            denominator > 0, denominator, torch.ones_like(denominator)
        )
    return weights.float(), ids.to(torch.int32)


def test_softmax_fallback_matches_kernel_semantics():
    gating = torch.randn(8, 16)
    bias = torch.randn(16) * 0.1

    weights, ids = router.fused_topk_bias_softmax_torch(gating, bias, 4, True)
    ref_weights, ref_ids = _reference_topk(gating, bias, 4, True, "softmax")

    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(weights, ref_weights)


def test_sigmoid_fallback_matches_kernel_semantics():
    gating = torch.randn(8, 16)
    bias = torch.randn(16) * 0.1

    weights, ids = router.fused_topk_bias_sigmoid_torch(gating, bias, 4, False)
    ref_weights, ref_ids = _reference_topk(gating, bias, 4, False, "sigmoid")

    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(weights, ref_weights)


def test_moe_c_detection_returns_bool():
    assert isinstance(router.has_moe_c_topk_sigmoid(), bool)
    assert isinstance(router.has_moe_c_topk_softmax(), bool)


class _FakeBiasRouter:
    def __init__(self, scoring_func):
        self.scoring_func = scoring_func
        self.e_score_correction_bias = None
        self.top_k = 4
        self.renormalize = True
        self.routed_scaling_factor = 1.0


def test_bias_router_softmax_uses_fallback_without_moe_c(monkeypatch):
    monkeypatch.setattr(router, "has_moe_c_topk_softmax", lambda: False)
    monkeypatch.setattr(
        router,
        "fused_topk_bias",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("upstream called")),
    )
    gating = torch.randn(8, 16)

    weights, ids = router.FusedTopKBiasRouterFL._compute_routing(
        _FakeBiasRouter("softmax"), None, gating, None
    )
    ref_weights, ref_ids = _reference_topk(gating, None, 4, True, "softmax")

    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(weights, ref_weights)


def test_bias_router_sigmoid_uses_fallback_without_moe_c(monkeypatch):
    monkeypatch.setattr(router, "has_moe_c_topk_sigmoid", lambda: False)
    monkeypatch.setattr(
        router,
        "fused_topk_bias",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("upstream called")),
    )
    gating = torch.randn(8, 16)

    weights, ids = router.FusedTopKBiasRouterFL._compute_routing(
        _FakeBiasRouter("sigmoid"), None, gating, None
    )
    ref_weights, ref_ids = _reference_topk(gating, None, 4, True, "sigmoid")

    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(weights, ref_weights)


def test_bias_router_keeps_upstream_when_moe_c_available(monkeypatch):
    monkeypatch.setattr(router, "has_moe_c_topk_softmax", lambda: True)
    sentinel = (torch.zeros(8, 4), torch.zeros(8, 4, dtype=torch.int32))
    monkeypatch.setattr(router, "fused_topk_bias", lambda **kwargs: sentinel)

    result = router.FusedTopKBiasRouterFL._compute_routing(
        _FakeBiasRouter("softmax"), None, torch.randn(8, 16), None
    )
    assert result is sentinel
