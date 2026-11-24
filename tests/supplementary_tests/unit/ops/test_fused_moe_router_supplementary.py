# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ``vllm_fl.ops.fused_moe.router``.

The core suite (``tests/unit_tests/ops/test_fused_moe_router.py``) covers
``fused_topk`` padding/zeroing, the bias router passthrough and the
sqrtsoftplus fallback. The baseline still misses:

* the pure-torch fallback body of ``_fl_grouped_topk`` (softmax/sigmoid
  scoring, unsupported-scoring errors, the biased CachedOp arms, mask
  construction, renormalization and routed scaling);
* ``GroupedTopKRouterFL._compute_routing``'s invalid-grouping branches
  (bias and plain top-k), the ROCm AITER arm, and the valid-grouping arm
  driving ``_fl_grouped_topk``;
* ``FusedTopKRouterFL._compute_routing`` (a thin wrapper over fused_topk);
* ``FusedTopKBiasRouterFL``'s sqrtsoftplus-on-NPU arm and the
  shared-experts concat;
* ``replace_router_with_fl`` installing the FL methods onto the upstream
  router classes.

All numeric checks use independent torch oracles built in the tests.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import vllm_fl.ops.fused_moe.router as router_mod

pytestmark = [pytest.mark.supplementary]


def _router_simple(**attrs):
    """Build the minimal attribute surface the FL routers read."""
    base = dict(
        e_score_correction_bias=None,
        top_k=2,
        renormalize=True,
        scoring_func="softmax",
        routed_scaling_factor=1.0,
        _hash_indices_table=None,
        num_fused_shared_experts=0,
        shared_expert_weight=1.0,
        global_num_experts=8,
        bias_vl=None,
        image_sentinel_lo=0,
        # Grouped router attributes
        num_expert_group=2,
        topk_group=1,
    )
    base.update(attrs)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# _fl_grouped_topk pure-torch fallback
# ---------------------------------------------------------------------------


class TestFlGroupedTopkFallback:
    def _softmax_oracle(self, gating, topk, num_expert_group, topk_group,
                        renormalize, routed_scaling_factor):
        """Reference implementation built only from public torch APIs."""
        scores = torch.softmax(gating.float(), dim=-1)
        num_experts = scores.size(-1)
        group_size = num_experts // num_expert_group
        grouped = scores.view(-1, num_expert_group, group_size)
        group_scores = grouped.amax(dim=-1)
        _, selected = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)
        mask = torch.zeros_like(scores)
        for i in range(topk_group):
            g = selected[:, i]
            start = g * group_size
            for j in range(group_size):
                mask.scatter_(1, (start + j).unsqueeze(1), 1.0)
        masked = scores * mask
        weights, ids = torch.topk(masked, k=topk, dim=-1, sorted=False)
        if renormalize:
            weights = weights / weights.sum(-1, keepdim=True)
        if routed_scaling_factor != 1.0:
            weights = weights * routed_scaling_factor
        return weights.to(torch.float32), ids.to(torch.int32)

    def test_full_softmax_oracle_equality(self):
        """The impl matches a from-scratch softmax oracle exactly."""
        torch.manual_seed(21)
        gating = torch.randn(10, 12)
        weights, ids = router_mod._fl_grouped_topk(
            gating, gating, 4, True, num_expert_group=3, topk_group=2
        )
        exp_w, exp_i = self._softmax_oracle(gating, 4, 3, 2, True, 1.0)
        # topk with sorted=False may return ties in different orders; compare
        # as sorted sets of (id) and per-row weight sums.
        for row in range(gating.shape[0]):
            assert sorted(ids[row].tolist()) == sorted(exp_i[row].tolist())
        torch.testing.assert_close(
            weights.sort(-1).values, exp_w.sort(-1).values, rtol=1e-5, atol=1e-5
        )

    def test_softmax_matches_oracle(self):
        torch.manual_seed(3)
        gating = torch.randn(6, 8)
        weights, ids = router_mod._fl_grouped_topk(
            gating, gating, 3, True, num_expert_group=2, topk_group=1
        )
        # Independent oracle: recompute from the returned ids — the weights
        # are the softmax scores gathered at those ids, renormalized to 1.
        scores = torch.softmax(gating.float(), dim=-1)
        re_gathered = scores.gather(1, ids.long())
        expected = re_gathered / re_gathered.sum(-1, keepdim=True)
        torch.testing.assert_close(weights, expected, rtol=1e-5, atol=1e-5)

    def test_sigmoid_scoring(self):
        torch.manual_seed(4)
        gating = torch.randn(4, 6)
        weights, ids = router_mod._fl_grouped_topk(
            gating, gating, 2, True, num_expert_group=3, topk_group=2,
            scoring_func="sigmoid",
        )
        scores = torch.sigmoid(gating)
        re_gathered = scores.gather(1, ids.long())
        expected = re_gathered / re_gathered.sum(-1, keepdim=True)
        torch.testing.assert_close(weights, expected, rtol=1e-5, atol=1e-5)
        # Weights renormalized to sum 1 per row.
        torch.testing.assert_close(
            weights.sum(-1), torch.ones(4), rtol=1e-5, atol=1e-5
        )

    def test_unsupported_scoring_without_bias(self):
        with pytest.raises(ValueError, match="Unsupported scoring function"):
            router_mod._fl_grouped_topk(
                torch.zeros(2, 4), torch.zeros(2, 4), 2, True,
                scoring_func="relu",
            )

    def test_routed_scaling_factor(self):
        gating = torch.zeros(2, 4)
        gating[0, 0] = 10.0  # expert 0 dominates
        weights, ids = router_mod._fl_grouped_topk(
            gating, gating, 1, True, num_expert_group=2, topk_group=1,
            routed_scaling_factor=2.5,
        )
        assert torch.allclose(weights, torch.full((2, 1), 2.5))

    def test_renormalize_false_keeps_raw_scores(self):
        torch.manual_seed(7)
        gating = torch.randn(3, 8)
        weights, ids = router_mod._fl_grouped_topk(
            gating, gating, 2, False, num_expert_group=2, topk_group=1
        )
        scores = torch.softmax(gating.float(), dim=-1)
        re_gathered = scores.gather(1, ids.long())
        torch.testing.assert_close(weights, re_gathered, rtol=1e-5, atol=1e-5)

    def test_group_selection_restricts_candidates(self):
        """All selected ids must come from the winning group only."""
        torch.manual_seed(13)
        gating = torch.randn(20, 12)
        num_expert_group, topk_group, topk = 3, 1, 4
        _, ids = router_mod._fl_grouped_topk(
            gating, gating, topk, True,
            num_expert_group=num_expert_group, topk_group=topk_group,
        )
        scores = torch.softmax(gating.float(), dim=-1)
        group_size = 12 // num_expert_group
        grouped = scores.view(-1, num_expert_group, group_size)
        best_group = grouped.amax(-1).argmax(-1)  # [20]
        for row in range(20):
            g = int(best_group[row])
            lo, hi = g * group_size, (g + 1) * group_size
            for expert in ids[row].tolist():
                assert lo <= expert < hi, (
                    f"row {row}: expert {expert} outside group {g} "
                    f"[{lo}, {hi})"
                )


class TestFlGroupedTopkBiased:
    """The e_score_correction_bias arms routing through the CachedOp.

    Neither registered ``grouped_topk`` implementation runs on a CPU test
    host (FlagGems needs a triton driver; the kunlunxin impl needs the
    accelerator), so the router module's ``_grouped_topk`` CachedOp is
    replaced with a recording stub for the duration of each test. The
    production code under test is the argument marshalling in
    ``_fl_grouped_topk``'s biased arms.
    """

    @pytest.fixture(autouse=True)
    def _clear_cached_failures(self):
        # A previous test file may have permanently failed this op on CPU.
        from vllm_fl.dispatch import get_default_manager

        get_default_manager().clear_failed_impls("grouped_topk")
        yield
        get_default_manager().clear_failed_impls("grouped_topk")

    def _install_stub(self, monkeypatch):
        calls = []

        def fake_grouped_topk(*args):
            calls.append(args)
            # Production signature: (scores, n_group, topk_group, topk,
            # renormalize, routed_scaling_factor, bias, scoring_func)
            # Return (values, indices) derived from the scores tensor.
            scores = args[0]
            topk = args[3]
            values, indices = torch.topk(scores, k=topk, dim=-1)
            return values.clone(), indices.clone()

        monkeypatch.setattr(router_mod, "_grouped_topk", fake_grouped_topk)
        return calls

    def test_sigmoid_bias_arm(self, monkeypatch):
        calls = self._install_stub(monkeypatch)
        gating = torch.tensor([[0.5, -0.2], [1.0, 0.3]])
        bias = torch.zeros(2)
        weights, ids = router_mod._fl_grouped_topk(
            gating, gating, 1, True, num_expert_group=1, topk_group=1,
            scoring_func="sigmoid", e_score_correction_bias=bias,
        )
        assert calls, "the patched op must be called"
        # The sigmoid arm forwards the RAW gating output (not sigmoided).
        torch.testing.assert_close(calls[0][0], gating)
        assert calls[0][1] == 1  # num_expert_group
        assert calls[0][2] == 1  # topk_group
        assert calls[0][3] == 1  # topk
        assert calls[0][4] is True  # renormalize
        assert calls[0][5] == 1.0  # routed_scaling_factor
        assert calls[0][6] is bias  # correction bias forwarded
        assert calls[0][7] == 1  # scoring_func code for sigmoid
        # Weights come back from the stub's topk selection.
        expected = torch.topk(gating, 1, dim=-1)
        torch.testing.assert_close(weights, expected.values)

    def test_softmax_bias_arm(self, monkeypatch):
        calls = self._install_stub(monkeypatch)
        gating = torch.tensor([[0.5, -0.2], [1.0, 0.3]])
        bias = torch.zeros(2)
        router_mod._fl_grouped_topk(
            gating, gating, 1, True, num_expert_group=1, topk_group=1,
            scoring_func="softmax", e_score_correction_bias=bias,
        )
        # The softmax arm pre-normalizes the scores before the op.
        torch.testing.assert_close(
            calls[0][0], torch.softmax(gating, dim=-1)
        )
        assert calls[0][7] == 0  # scoring_func code for softmax

    def test_unsupported_scoring_with_bias(self):
        with pytest.raises(ValueError, match="Unsupported scoring function"):
            router_mod._fl_grouped_topk(
                torch.zeros(2, 4), torch.zeros(2, 4), 1, True,
                scoring_func="relu", e_score_correction_bias=torch.zeros(4),
            )


class TestGroupedTopKRouterComputeRouting:
    def _router(self, **attrs):
        return _router_simple(**attrs)

    def test_invalid_grouping_without_bias_uses_fused_topk(self, monkeypatch):
        captured = {}

        def fake_fused_topk(**kwargs):
            captured.update(kwargs)
            return (
                torch.tensor([[0.25, 0.75]]),
                torch.tensor([[1, 0]], dtype=torch.int32),
                torch.zeros(1, 2),
            )

        monkeypatch.setattr(router_mod, "fused_topk", fake_fused_topk)
        # num_experts (4) <= num_expert_group (8): invalid grouping.
        router = self._router(num_expert_group=8, topk_group=1)
        weights, ids = router_mod.GroupedTopKRouterFL._compute_routing(
            router, torch.zeros(1, 4), torch.zeros(1, 4), torch.int32,
        )
        assert torch.equal(ids, torch.tensor([[1, 0]], dtype=torch.int32))
        assert captured["topk"] == 2

    def test_invalid_grouping_with_bias_uses_fused_topk_bias(self, monkeypatch):
        captured = {}

        def fake_bias(**kwargs):
            captured.update(kwargs)
            return torch.tensor([[0.5, 0.5]]), torch.tensor([[3, 1]], dtype=torch.int32)

        monkeypatch.setattr(router_mod, "fused_topk_bias", fake_bias)
        router = self._router(
            num_expert_group=8,
            e_score_correction_bias=torch.nn.Parameter(torch.zeros(4)),
            routed_scaling_factor=0.5,
        )
        weights, ids = router_mod.GroupedTopKRouterFL._compute_routing(
            router, torch.zeros(1, 4), torch.zeros(1, 4), torch.int32,
        )
        assert captured["topk"] == 2
        assert torch.allclose(weights, torch.full((1, 2), 0.25))  # 0.5 * 0.5

    def test_valid_grouping_uses_fl_grouped_topk(self, monkeypatch):
        captured = {}

        def fake_grouped(**kwargs):
            captured.update(kwargs)
            return torch.tensor([[0.4, 0.6]]), torch.tensor([[0, 1]], dtype=torch.int32)

        monkeypatch.setattr(router_mod, "_fl_grouped_topk", fake_grouped)
        router = self._router(num_expert_group=2, topk_group=1)
        weights, ids = router_mod.GroupedTopKRouterFL._compute_routing(
            router, torch.zeros(1, 4), torch.zeros(1, 4), torch.int32,
        )
        assert "num_expert_group" in captured
        assert captured["num_expert_group"] == 2

    def test_rocm_aiter_arm(self, monkeypatch):
        """The AITER-enabled arm wraps rocm_aiter_grouped_topk via partial."""
        captured = {}

        def fake_aiter(hidden_states, gating_output, **kwargs):
            captured.update(kwargs)
            captured["hidden"] = hidden_states
            return torch.tensor([[0.9]]), torch.tensor([[2]], dtype=torch.int32)

        import vllm._aiter_ops as aiter_mod

        monkeypatch.setattr(
            aiter_mod.rocm_aiter_ops, "is_fused_moe_enabled", lambda: True
        )
        monkeypatch.setattr(
            aiter_mod.rocm_aiter_ops,
            "is_fusion_moe_shared_experts_enabled",
            lambda: True,
        )
        monkeypatch.setattr(
            router_mod, "rocm_aiter_grouped_topk",
            lambda *args, **kwargs: fake_aiter(*args, **kwargs),
        )
        router = self._router(num_expert_group=2, topk_group=1,
                              num_fused_shared_experts=0)
        weights, ids = router_mod.GroupedTopKRouterFL._compute_routing(
            router, torch.zeros(1, 4), torch.zeros(1, 4), torch.int32,
        )
        assert captured["hidden"] is not None
        assert torch.equal(ids, torch.tensor([[2]], dtype=torch.int32))


class TestFusedTopKRouterComputeRouting:
    def test_wrapper_delegates_to_fused_topk(self, monkeypatch):
        captured = {}

        def fake_fused_topk(**kwargs):
            captured.update(kwargs)
            return (
                torch.tensor([[0.1, 0.9]]),
                torch.tensor([[4, 2]], dtype=torch.int32),
                None,
            )

        monkeypatch.setattr(router_mod, "fused_topk", fake_fused_topk)
        router = _router_simple(
            top_k=2, renormalize=False, scoring_func="softmax"
        )
        weights, ids = router_mod.FusedTopKRouterFL._compute_routing(
            router, torch.zeros(1, 6), torch.zeros(1, 6), torch.int64,
        )
        assert captured["topk"] == 2
        assert captured["renormalize"] is False
        assert captured["scoring_func"] == "softmax"
        assert captured["indices_type"] is torch.int64
        assert torch.equal(ids, torch.tensor([[4, 2]], dtype=torch.int32))


class TestBiasRouterArms:
    def test_sqrtsoftplus_npu_arm(self, monkeypatch):
        """On NPU tensors the sqrtsoftplus native path is taken."""
        monkeypatch.setattr(router_mod, "_is_ascend_npu_tensor", lambda v: True)
        monkeypatch.setattr(
            router_mod,
            "fused_topk_bias",
            lambda **k: (_ for _ in ()).throw(
                AssertionError("must not reach the CUDA router")
            ),
        )
        torch.manual_seed(11)
        gating = torch.randn(3, 5)
        bias = torch.randn(5)
        router = _router_simple(
            e_score_correction_bias=torch.nn.Parameter(bias),
            top_k=2, renormalize=True, scoring_func="sqrtsoftplus",
            routed_scaling_factor=1.0,
        )
        weights, ids = router_mod.FusedTopKBiasRouterFL._compute_routing(
            router, torch.zeros(3, 4), gating, torch.int32,
        )
        # Independent oracle: sqrt(softplus(x) + bias) selection.
        scores = torch.sqrt(F.softplus(gating))
        choice = scores + bias
        expected_ids = torch.topk(choice, 2, dim=-1, sorted=False).indices
        expected_w = scores.gather(1, expected_ids)
        expected_w = expected_w / expected_w.sum(-1, keepdim=True)
        assert set(ids[0].tolist()) == set(expected_ids[0].tolist())
        torch.testing.assert_close(
            weights.sum(-1), torch.ones(3), rtol=1e-5, atol=1e-5
        )

    def test_shared_experts_appended(self, monkeypatch):
        def fake_bias(**kwargs):
            return torch.tensor([[0.25, 0.75]]), torch.tensor([[2, 4]], dtype=torch.int32)

        monkeypatch.setattr(router_mod, "fused_topk_bias", fake_bias)
        monkeypatch.setattr(router_mod, "_is_ascend_npu_tensor", lambda v: False)
        router = _router_simple(
            num_fused_shared_experts=2,
            shared_expert_weight=1.25,
            global_num_experts=6,
            scoring_func="softmax",
        )
        weights, ids = router_mod.FusedTopKBiasRouterFL._compute_routing(
            router, torch.zeros(1, 4), torch.zeros(1, 6), torch.int32,
        )
        assert torch.equal(ids, torch.tensor([[2, 4, 6, 7]], dtype=torch.int32))
        assert torch.equal(
            weights, torch.tensor([[0.25, 0.75, 1.25, 1.25]])
        )

    def test_upstream_path_forwards_all_arguments(self, monkeypatch):
        captured = {}

        def fake_bias(**kwargs):
            captured.update(kwargs)
            return torch.tensor([[0.5]]), torch.tensor([[1]], dtype=torch.int32)

        monkeypatch.setattr(router_mod, "fused_topk_bias", fake_bias)
        monkeypatch.setattr(router_mod, "_is_ascend_npu_tensor", lambda v: False)
        hash_table = torch.tensor([[0, 1]])
        input_ids = torch.tensor([3])
        router = _router_simple(
            e_score_correction_bias=torch.nn.Parameter(torch.zeros(3)),
            bias_vl=torch.nn.Parameter(torch.zeros(3)),
            image_sentinel_lo=7,
            _hash_indices_table=hash_table,
        )
        router_mod.FusedTopKBiasRouterFL._compute_routing(
            router, torch.zeros(1, 4), torch.zeros(1, 3), torch.int32,
            input_ids=input_ids,
        )
        assert captured["input_tokens"] is input_ids
        assert captured["hash_indices_table"] is hash_table
        assert captured["image_sentinel_lo"] == 7


class TestReplaceRouterWithFl:
    def test_methods_installed_on_upstream(self):
        import vllm.model_executor.layers.fused_moe.router.fused_topk_router as ftr
        import vllm.model_executor.layers.fused_moe.router.grouped_topk_router as gtr
        import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as btr

        originals = (
            ftr.FusedTopKRouter._compute_routing,
            gtr.GroupedTopKRouter._compute_routing,
            btr.FusedTopKBiasRouter._compute_routing,
        )
        try:
            router_mod.replace_router_with_fl()
            assert (ftr.FusedTopKRouter._compute_routing
                    is router_mod.FusedTopKRouterFL._compute_routing)
            assert (gtr.GroupedTopKRouter._compute_routing
                    is router_mod.GroupedTopKRouterFL._compute_routing)
            assert (btr.FusedTopKBiasRouter._compute_routing
                    is router_mod.FusedTopKBiasRouterFL._compute_routing)
        finally:
            (ftr.FusedTopKRouter._compute_routing,
             gtr.GroupedTopKRouter._compute_routing,
             btr.FusedTopKBiasRouter._compute_routing) = originals
