# Copyright (c) 2026 BAAI. All rights reserved.

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_fl.ops.fused_moe import layer, router as router_module


def _runner_with_quant_method(quant_method):
    runner = MagicMock()
    runner._quant_method = quant_method
    runner.moe_config = MagicMock()
    return runner


def test_fused_moe_fl_replaces_unquantized_method():
    quant_method = MagicMock(spec=layer.UnquantizedFusedMoEMethod)
    runner = _runner_with_quant_method(quant_method)
    replacement = MagicMock()

    with (
        patch.object(layer, "_OrigFusedMoE", return_value=runner),
        patch.object(
            layer,
            "UnquantizedFusedMoEMethodFL",
            return_value=replacement,
        ) as replacement_cls,
        patch.object(layer, "replace_router_with_fl") as replace_router,
    ):
        result = layer.FusedMoEFL(test_arg=True)

    assert result is runner
    replacement_cls.assert_called_once_with(runner.moe_config)
    runner._replace_quant_method.assert_called_once_with(replacement)
    replace_router.assert_called_once_with()


def test_fused_moe_fl_preserves_quantized_method():
    quant_method = object()
    runner = _runner_with_quant_method(quant_method)

    with (
        patch.object(layer, "_OrigFusedMoE", return_value=runner),
        patch.object(layer, "UnquantizedFusedMoEMethodFL") as replacement_cls,
        patch.object(layer, "replace_router_with_fl") as replace_router,
        patch.object(layer.logger, "info_once") as info_once,
    ):
        result = layer.FusedMoEFL()

    assert result is runner
    replacement_cls.assert_not_called()
    runner._replace_quant_method.assert_not_called()
    replace_router.assert_called_once_with()
    info_once.assert_called_once_with(
        "Preserving upstream quantized MoE method %s in FusedMoEFL.",
        "object",
    )


def _bias_router(bias, scale=1.25):
    return SimpleNamespace(
        e_score_correction_bias=SimpleNamespace(data=bias),
        top_k=4,
        renormalize=True,
        scoring_func="sigmoid",
        routed_scaling_factor=scale,
    )


def _bias_reference(logits, bias, ids, scale):
    scores = torch.sigmoid(logits.float())
    expected_ids = (
        (scores + bias.float()).topk(ids.shape[-1], dim=-1, sorted=False).indices
    )
    assert all(
        (
            set(actual) == set(expected)
            for actual, expected in zip(ids.cpu().tolist(), expected_ids.cpu().tolist())
        )
    )
    weights = scores.gather(1, ids.long())
    return weights / weights.sum(dim=-1, keepdim=True) * scale


def _invoke_bias_router(router, hidden_states, logits):
    return router_module.FusedTopKBiasRouterFL._compute_routing(
        router, hidden_states, logits, indices_type=None
    )


def test_sigmoid_bias_routes_through_cached_topk_sigmoid(monkeypatch):
    expected_weights = torch.ones(2, 4, dtype=torch.float32)
    expected_ids = torch.zeros(2, 4, dtype=torch.int32)
    calls = []

    def fake_topk_sigmoid(**kwargs):
        calls.append(kwargs)
        return (expected_weights, expected_ids)

    monkeypatch.setattr(router_module, "_topk_sigmoid", fake_topk_sigmoid)
    hidden_states = torch.zeros(2, 64, dtype=torch.bfloat16)
    logits = torch.zeros(2, 128, dtype=torch.float32)
    bias = torch.zeros(128, dtype=torch.float32)
    route = _bias_router(bias, scale=1.0)
    weights, ids = _invoke_bias_router(route, hidden_states, logits)
    assert calls and calls[0]["e_score_correction_bias"] is bias
    assert calls[0]["topk"] == 4
    assert calls[0]["renormalize"] is True
    assert weights is expected_weights
    assert ids is expected_ids


@pytest.mark.parametrize("indices_type", [torch.int32, torch.int64])
def test_sigmoid_bias_preserves_requested_indices_type(monkeypatch, indices_type):
    monkeypatch.setattr(
        router_module,
        "_topk_sigmoid",
        lambda **kwargs: (torch.ones(2, 4), torch.zeros(2, 4, dtype=torch.int32)),
    )
    _, ids = router_module.FusedTopKBiasRouterFL._compute_routing(
        _bias_router(torch.zeros(128)),
        torch.zeros(2, 64),
        torch.zeros(2, 128),
        indices_type=indices_type,
    )
    assert ids.dtype == indices_type


def test_softmax_bias_retains_native_routing_contract(monkeypatch):
    calls = []
    expected = (torch.ones(2, 4), torch.zeros(2, 4, dtype=torch.int64))

    def native(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(router_module, "fused_topk_bias", native)
    route = _bias_router(torch.zeros(128), scale=1.0)
    route.scoring_func = "softmax"
    result = router_module.FusedTopKBiasRouterFL._compute_routing(
        route, torch.zeros(2, 64), torch.zeros(2, 128), indices_type=torch.int64
    )
    assert result == expected
    assert calls[0]["indices_type"] is torch.int64


@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("indices_type", [None, torch.int32, torch.int64])
def test_sigmoid_bias_weights_scale_and_dtype(monkeypatch, renormalize, indices_type):
    def sigmoid_route(**kwargs):
        scores = torch.sigmoid(kwargs["gating_output"].float())
        ids = (
            (scores + kwargs["e_score_correction_bias"])
            .topk(kwargs["topk"], dim=-1)
            .indices
        )
        weights = scores.gather(1, ids)
        if kwargs["renormalize"]:
            weights = weights / weights.sum(-1, keepdim=True)
        return weights, ids.to(kwargs["indices_type"] or torch.int32)

    monkeypatch.setattr(router_module, "_topk_sigmoid", sigmoid_route)
    monkeypatch.setattr(
        router_module,
        "_fl_grouped_topk",
        lambda **kwargs: pytest.fail("ordinary routing must not call grouped top-k"),
    )
    logits = torch.tensor([[0.2, -1.3, 2.4, 0.8, -0.5, 1.1]])
    bias = torch.tensor([0.1, 0.5, -0.9, 0.2, 0.3, -0.1])
    route = _bias_router(bias, scale=1.7)
    route.renormalize = renormalize
    weights, ids = router_module.FusedTopKBiasRouterFL._compute_routing(
        route, torch.zeros(1, 64), logits, indices_type=indices_type
    )
    expected = torch.sigmoid(logits.float()).gather(1, ids.long())
    if renormalize:
        expected = expected / expected.sum(-1, keepdim=True)
    torch.testing.assert_close(weights, expected * 1.7)
    assert ids.dtype == (indices_type or torch.int32)


def test_true_grouped_sigmoid_keeps_grouped_dispatch(monkeypatch):
    calls = []
    expected = (torch.ones(2, 4), torch.zeros(2, 4, dtype=torch.int32))

    def grouped(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(router_module, "_grouped_topk", grouped)
    monkeypatch.setattr(
        router_module,
        "_topk_sigmoid",
        lambda **kwargs: pytest.fail("grouped routing must not call ordinary top-k"),
    )
    logits = torch.zeros(2, 128)
    bias = torch.zeros(128)
    actual = router_module._fl_grouped_topk(
        torch.zeros(2, 64),
        logits,
        4,
        True,
        num_expert_group=4,
        topk_group=2,
        scoring_func="sigmoid",
        e_score_correction_bias=bias,
    )
    assert actual[0] is expected[0] and actual[1] is expected[1]
    assert calls[0][1:4] == (4, 2, 4)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or "PPU_SDK" not in os.environ,
    reason="T-Head PPU is not available",
)
class TestTheadMoERouting:
    @pytest.fixture(autouse=True)
    def flaggems_dispatch(self, monkeypatch):
        import vllm_fl.dispatch as dispatch
        from vllm_fl.dispatch.manager import OpManager
        from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

        # Exercise the optional library explicitly, with a fresh cache and
        # strict selection so a native fallback cannot pass this kernel test.
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
        monkeypatch.setenv("USE_FLAGGEMS", "1")
        monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "topk_sigmoid")
        manager = OpManager()
        monkeypatch.setattr(dispatch, "_CACHED_OPS", [])
        monkeypatch.setattr(dispatch, "get_default_manager", lambda: manager)
        monkeypatch.setattr(
            router_module, "_topk_sigmoid", dispatch.CachedOp("topk_sigmoid")
        )
        policy = SelectionPolicy.from_dict(
            strict=True, per_op_order={"topk_sigmoid": ["flagos"]}
        )
        with policy_context(policy):
            assert manager._resolve_impl("topk_sigmoid").impl_id == "default.flagos"
            yield

    def test_sigmoid_bias_matches_reference(self):
        torch.manual_seed(21)
        hidden_states = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16)
        logits = torch.randn(17, 128, device="cuda", dtype=torch.float32)
        bias = torch.randn(128, device="cuda", dtype=torch.float32) * 0.1
        route = _bias_router(bias)
        weights, ids = _invoke_bias_router(route, hidden_states, logits)
        torch.testing.assert_close(
            weights,
            _bias_reference(logits, bias, ids, route.routed_scaling_factor),
            rtol=2e-05,
            atol=2e-05,
        )
        assert weights.dtype == torch.float32
        assert ids.dtype == torch.int32

    def test_flaggems_sigmoid_bias_cuda_graph_changed_input(self):
        torch.manual_seed(22)
        hidden_states = torch.randn(9, 64, device="cuda", dtype=torch.bfloat16)
        logits = torch.randn(9, 128, device="cuda", dtype=torch.float32)
        bias = torch.randn(128, device="cuda", dtype=torch.float32) * 0.1
        route = _bias_router(bias)
        _invoke_bias_router(route, hidden_states, logits)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            weights, ids = _invoke_bias_router(route, hidden_states, logits)
        torch.cuda.synchronize()
        replay_logits = torch.randn_like(logits)
        logits.copy_(replay_logits)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            weights,
            _bias_reference(replay_logits, bias, ids, route.routed_scaling_factor),
            rtol=2e-05,
            atol=2e-05,
        )
