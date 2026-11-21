# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import torch
import torch.nn.functional as F


def test_fused_topk_zeroes_padded_rows_without_native_moe_kernel(monkeypatch):
    """Padded rows must not reach the experts when ``_moe_C`` is absent."""
    import vllm_fl.ops.fused_moe.router as router_mod

    def fake_topk(weights, ids, token_expert_indices, gating_output, renormalize):
        weights.fill_(1.0)
        ids.fill_(0)
        return weights, ids

    monkeypatch.setattr(router_mod, "_topk_softmax", fake_topk)
    monkeypatch.setattr(router_mod, "_has_native_moe_topk", lambda: False)
    monkeypatch.setattr(
        router_mod, "_get_padding_mask", lambda n: torch.tensor([False, True])[:n]
    )
    monkeypatch.setattr(
        router_mod,
        "upstream_fused_topk",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("the CUDA-only upstream kernel must not be required")
        ),
    )

    weights, ids, token_expert_indices = router_mod.fused_topk(
        hidden_states=torch.zeros(2, 4),
        gating_output=torch.zeros(2, 3),
        topk=2,
        renormalize=False,
    )

    assert torch.equal(weights, torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    assert ids.shape == (2, 2)
    assert token_expert_indices.shape == (2, 2)


def test_fused_topk_prefers_upstream_when_native_kernel_takes_the_mask(monkeypatch):
    import vllm_fl.ops.fused_moe.router as router_mod

    captured = {}

    def upstream(**kwargs):
        captured.update(kwargs)
        return (
            torch.zeros(2, 2),
            torch.zeros(2, 2, dtype=torch.int32),
            torch.zeros(2, 2),
        )

    monkeypatch.setattr(router_mod, "_has_native_moe_topk", lambda: True)
    monkeypatch.setattr(
        router_mod, "_get_padding_mask", lambda n: torch.tensor([False, True])[:n]
    )
    monkeypatch.setattr(router_mod, "upstream_fused_topk", upstream)

    router_mod.fused_topk(
        hidden_states=torch.zeros(2, 4),
        gating_output=torch.zeros(2, 3),
        topk=2,
        renormalize=False,
    )

    assert captured["scoring_func"] == "softmax"


def test_bias_router_preserves_hash_scaling_and_shared_experts(monkeypatch):
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
        bias_vl=None,
        image_sentinel_lo=0,
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
    assert captured["bias_vl"] is None
    assert captured["image_sentinel_lo"] == 0
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


def _padding_inputs():
    hidden_states = torch.zeros((4, 16))
    gating_output = torch.zeros((4, 8))
    padding_mask = torch.tensor([False, False, True, True])
    return hidden_states, gating_output, padding_mask


def test_fused_topk_padded_delegates_when_native_op_exists(monkeypatch):
    """Vendors with the native padded kernel keep the upstream path."""
    import vllm_fl.ops.fused_moe.router as router_mod

    hidden_states, gating_output, padding_mask = _padding_inputs()
    sentinel = (
        torch.full((4, 2), 0.25),
        torch.full((4, 2), 3, dtype=torch.int32),
        torch.zeros((4, 2), dtype=torch.int32),
    )
    calls = {"upstream": 0, "kernel": 0}

    def fake_upstream(**kwargs):
        calls["upstream"] += 1
        return sentinel

    def fail_kernel(*args, **kwargs):
        calls["kernel"] += 1
        raise AssertionError("FL kernel must not run when delegating upstream")

    monkeypatch.setattr(
        router_mod, "_get_padding_mask", lambda num_tokens: padding_mask
    )
    monkeypatch.setattr(router_mod, "upstream_fused_topk", fake_upstream)
    monkeypatch.setattr(router_mod, "_topk_softmax", fail_kernel)
    monkeypatch.setattr(torch.ops, "_moe_C", SimpleNamespace(topk_softmax=object()))

    result = router_mod.fused_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=2,
        renormalize=True,
    )

    assert calls == {"upstream": 1, "kernel": 0}
    assert all(
        torch.equal(actual, expected) for actual, expected in zip(result, sentinel)
    )


def test_fused_topk_padded_zeroes_weights_without_native_op(monkeypatch):
    """Empty builds zero the padding rows after running the FL kernel."""
    import vllm_fl.ops.fused_moe.router as router_mod

    hidden_states, gating_output, padding_mask = _padding_inputs()

    def fake_upstream(**kwargs):
        raise AssertionError("upstream path requires the missing native operator")

    def fake_kernel(weights, ids, token_expert_indices, gating_output, renormalize):
        weights.fill_(1.0)
        ids.fill_(2)
        return weights, ids

    monkeypatch.setattr(
        router_mod, "_get_padding_mask", lambda num_tokens: padding_mask
    )
    monkeypatch.setattr(router_mod, "upstream_fused_topk", fake_upstream)
    monkeypatch.setattr(router_mod, "_topk_softmax", fake_kernel)
    # No topk_softmax on the namespace -> hasattr() is False on every vendor.
    monkeypatch.setattr(torch.ops, "_moe_C", SimpleNamespace())

    weights, ids, token_expert_indices = router_mod.fused_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=2,
        renormalize=True,
    )

    torch.testing.assert_close(weights[:2], torch.ones((2, 2)))
    torch.testing.assert_close(weights[2:], torch.zeros((2, 2)))
    torch.testing.assert_close(ids, torch.full((4, 2), 2, dtype=torch.int32))
