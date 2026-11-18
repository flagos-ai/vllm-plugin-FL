# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the FL fused-MoE routing layer.

``vllm_fl.ops.fused_moe.router`` allocates the top-k buffers and routes the
selection kernels through ``CachedOp`` dispatch.  The dispatch kernel (the
FlagGems/torch boundary) is replaced with recorders that write explicit
values into the output buffers; the grouped-top-k fallback is compared
against a plain-torch reference derived in the test.  Router instances are
built with ``object.__new__`` because only their configuration attributes
are read by the methods under test.
"""

from __future__ import annotations

import pytest
import torch

from . import _imports  # noqa: F401  (installs CPU-only import placeholders)
from vllm_fl.ops.fused_moe import router as router_module

Mod = router_module


class _KernelRecorder:
    """Recording stand-in for a ``CachedOp`` selection kernel."""

    def __init__(self, impl=None):
        self.calls = []
        self.impl = impl

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.impl is not None:
            return self.impl(*args, **kwargs)
        return None


def _router(cls, **attrs):
    instance = object.__new__(cls)
    for name, value in attrs.items():
        object.__setattr__(instance, name, value)
    return instance


def test_fused_topk_allocates_buffers_and_returns_kernel_writes():
    """Buffer shapes/dtypes and the in-place kernel contract are preserved."""
    seen = {}

    def kernel(topk_weights, topk_ids, token_expert_indices, gating, renormalize):
        seen.update(
            weights=topk_weights,
            ids=topk_ids,
            token_indices=token_expert_indices,
            renormalize=renormalize,
        )
        topk_weights.copy_(torch.tensor([[0.25, 0.75], [1.0, 0.0]]))
        topk_ids.copy_(torch.tensor([[3, 1], [0, 2]], dtype=topk_ids.dtype))
        return topk_weights, topk_ids

    recorder = _KernelRecorder(kernel)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Mod, "_topk_softmax", recorder)
        hidden = torch.zeros(2, 4)
        gating = torch.arange(8, dtype=torch.float32).view(2, 4)
        weights, ids, token_indices = Mod.fused_topk(
            hidden, gating, topk=2, renormalize=True
        )

    assert seen["weights"] is weights
    assert seen["ids"] is ids
    assert seen["token_indices"] is token_indices
    assert seen["renormalize"] is True
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert weights.shape == ids.shape == token_indices.shape == (2, 2)
    assert weights.tolist() == [[0.25, 0.75], [1.0, 0.0]]
    assert ids.tolist() == [[3, 1], [0, 2]]


def test_fused_topk_honours_custom_indices_dtype_and_rejects_token_mismatch():
    """indices_type overrides the id dtype; a token-count mismatch asserts."""

    def kernel(weights, ids, token_indices, gating, renormalize):
        weights.fill_(0.5)
        ids.fill_(1)
        return weights, ids

    recorder = _KernelRecorder(kernel)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Mod, "_topk_softmax", recorder)
        _, ids, _ = Mod.fused_topk(
            torch.zeros(2, 3),
            torch.zeros(2, 5),
            topk=2,
            renormalize=False,
            indices_type=torch.int64,
        )
        with pytest.raises(AssertionError, match="Number of tokens mismatch"):
            Mod.fused_topk(
                torch.zeros(2, 3), torch.zeros(3, 5), topk=2, renormalize=False
            )
    assert ids.dtype == torch.int64


def test_fused_topk_router_routes_config_into_the_kernel():
    """The router forwards top_k/renormalize and returns the kernel pair."""

    def kernel(weights, ids, token_indices, gating, renormalize):
        weights.fill_(0.5)
        ids.fill_(2)
        return weights, ids

    recorder = _KernelRecorder(kernel)
    instance = _router(Mod.FusedTopKRouterFL, top_k=3, renormalize=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Mod, "_topk_softmax", recorder)
        weights, ids = Mod.FusedTopKRouterFL._compute_routing(
            instance,
            torch.zeros(2, 4),
            torch.zeros(2, 6),
            torch.int64,
        )
    call_args = recorder.calls[0][0]
    assert call_args[3].shape == (2, 6)  # gating_output
    assert call_args[4] is False  # renormalize
    assert weights.shape == (2, 3)
    assert ids.dtype == torch.int64
    assert ids.tolist() == [[2, 2, 2], [2, 2, 2]]


def _grouped_reference(
    gating,
    topk,
    renormalize,
    num_expert_group,
    topk_group,
    scoring_func,
    scaling,
):
    """Plain-torch grouped top-k with the documented semantics."""
    scores = (
        torch.softmax(gating, dim=-1)
        if scoring_func == "softmax"
        else torch.sigmoid(gating)
    )
    experts = scores.size(-1)
    group_size = experts // num_expert_group
    grouped = scores.view(-1, num_expert_group, group_size)
    group_scores = grouped.amax(dim=-1)
    _, selected = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)
    mask = torch.zeros_like(scores)
    for i in range(topk_group):
        start = selected[:, i] * group_size
        for j in range(group_size):
            mask.scatter_(1, (start + j).unsqueeze(1), 1.0)
    values, indices = torch.topk(scores * mask, k=topk, dim=-1, sorted=False)
    if renormalize:
        values = values / values.sum(dim=-1, keepdim=True)
    values = values * scaling
    return values.to(torch.float32), indices.to(torch.int32)


@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize(("renormalize", "scaling"), [(True, 1.0), (False, 2.5)])
def test_grouped_topk_fallback_matches_plain_torch_reference(
    scoring_func, renormalize, scaling
):
    """Without a correction bias the pure-torch selection is exact."""
    gating = torch.tensor(
        [
            [0.2, 1.4, -0.3, 0.7, 2.1, -1.2, 0.5, 0.9],
            [1.0, -0.4, 0.3, 2.2, 0.1, 0.8, -0.9, 1.6],
        ]
    )
    expected_w, expected_ids = _grouped_reference(
        gating,
        topk=3,
        renormalize=renormalize,
        num_expert_group=2,
        topk_group=1,
        scoring_func=scoring_func,
        scaling=scaling,
    )
    weights, ids = Mod._fl_grouped_topk(
        gating,
        gating,
        topk=3,
        renormalize=renormalize,
        num_expert_group=2,
        topk_group=1,
        scoring_func=scoring_func,
        routed_scaling_factor=scaling,
        e_score_correction_bias=None,
    )
    torch.testing.assert_close(weights, expected_w)
    assert ids.tolist() == expected_ids.tolist()


def test_grouped_topk_bias_sigmoid_passes_raw_scores_and_scoring_code():
    """Bias + sigmoid forwards the logits unchanged with scoring code 1."""
    recorder = _KernelRecorder(
        lambda *a: (torch.full((2, 2), 0.75), torch.zeros(2, 2, dtype=torch.long))
    )
    bias = torch.tensor([0.1, 0.2, 0.3, 0.4])
    gating = torch.arange(8, dtype=torch.float32).view(2, 4)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Mod, "_grouped_topk", recorder)
        values, indices = Mod._fl_grouped_topk(
            gating,
            gating,
            topk=2,
            renormalize=True,
            num_expert_group=2,
            topk_group=1,
            scoring_func="sigmoid",
            routed_scaling_factor=1.0,
            e_score_correction_bias=bias,
        )
    args = recorder.calls[0][0]
    assert args[0] is gating  # raw logits, not sigmoid scores
    assert args[1:6] == (2, 1, 2, True, 1.0)
    assert args[6] is bias
    assert args[7] == 1
    assert values.tolist() == [[0.75, 0.75], [0.75, 0.75]]
    assert indices.tolist() == [[0, 0], [0, 0]]


def test_grouped_topk_bias_softmax_quantizes_scores_before_dispatch():
    """Bias + softmax passes softmax(gating) with scoring code 0."""
    recorder = _KernelRecorder(
        lambda *a: (torch.zeros(2, 1), torch.zeros(2, 1, dtype=torch.long))
    )
    bias = torch.tensor([0.0, 0.0, 0.0, 1.0])
    gating = torch.tensor([[1.0, 2.0, 3.0, 0.0], [0.0, 1.0, 0.0, 2.0]])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Mod, "_grouped_topk", recorder)
        Mod._fl_grouped_topk(
            gating,
            gating,
            topk=1,
            renormalize=False,
            num_expert_group=2,
            topk_group=1,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            e_score_correction_bias=bias,
        )
    args = recorder.calls[0][0]
    torch.testing.assert_close(args[0], torch.softmax(gating, dim=-1))
    assert args[7] == 0


def test_grouped_topk_rejects_unknown_scoring_function_in_both_paths():
    """An unsupported scoring function is rejected with or without a bias."""
    gating = torch.zeros(1, 4)
    bias = torch.zeros(4)
    with pytest.raises(ValueError, match="Unsupported scoring function: bad"):
        Mod._fl_grouped_topk(
            gating,
            gating,
            topk=1,
            renormalize=False,
            num_expert_group=2,
            topk_group=1,
            scoring_func="bad",
            e_score_correction_bias=bias,
        )
    with pytest.raises(ValueError, match="Unsupported scoring function: bad"):
        Mod._fl_grouped_topk(
            gating,
            gating,
            topk=1,
            renormalize=False,
            num_expert_group=2,
            topk_group=1,
            scoring_func="bad",
            e_score_correction_bias=None,
        )


def _grouped_router(**overrides):
    attrs = {
        "num_expert_group": 2,
        "topk_group": 1,
        "top_k": 2,
        "renormalize": False,
        "scoring_func": "softmax",
        "routed_scaling_factor": 1.0,
        "e_score_correction_bias": None,
        "num_fused_shared_experts": 0,
    }
    attrs.update(overrides)
    return _router(Mod.GroupedTopKRouterFL, **attrs)


def test_grouped_router_invalid_grouping_falls_back_without_bias():
    """Invalid grouping + no bias routes through fused_topk."""

    def kernel(weights, ids, token_indices, gating, renormalize):
        weights.fill_(0.5)
        ids.fill_(1)
        return weights, ids

    recorder = _KernelRecorder(kernel)
    instance = _grouped_router()
    logits = torch.zeros(2, 5)  # 5 % 2 != 0
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Mod, "_topk_softmax", recorder)
        weights, ids = Mod.GroupedTopKRouterFL._compute_routing(
            instance, torch.zeros(2, 3), logits, torch.int32
        )
    assert len(recorder.calls) == 1
    assert weights.tolist() == [[0.5, 0.5], [0.5, 0.5]]
    assert ids.tolist() == [[1, 1], [1, 1]]


def test_grouped_router_invalid_grouping_uses_bias_fusion_and_scales():
    """Invalid grouping + bias uses fused_topk_bias then applies the factor."""
    recorder = _KernelRecorder(
        lambda **kw: (torch.full((2, 2), 0.25), torch.zeros(2, 2, dtype=torch.long))
    )
    bias = torch.tensor([0.0, 0.1, 0.2, 0.3])
    instance = _grouped_router(
        scoring_func="sigmoid",
        routed_scaling_factor=3.0,
        e_score_correction_bias=bias,
    )
    logits = torch.zeros(2, 5)  # 5 % 2 != 0: invalid grouping
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Mod, "fused_topk_bias", recorder)
        weights, ids = Mod.GroupedTopKRouterFL._compute_routing(
            instance, torch.zeros(2, 3), logits, torch.int64
        )
    call = recorder.calls[0][1]
    torch.testing.assert_close(call["e_score_correction_bias"], bias)
    assert call["topk"] == 2
    assert call["renormalize"] is False
    assert call["scoring_func"] == "sigmoid"
    assert call["indices_type"] is torch.int64
    assert weights.tolist() == [[0.75, 0.75], [0.75, 0.75]]


def test_grouped_router_valid_grouping_uses_flaggems_grouped_topk():
    """Valid grouping (FlagGems path) forwards every routing parameter."""
    recorder = _KernelRecorder(
        lambda *a: (torch.full((2, 2), 0.5), torch.ones(2, 2, dtype=torch.long))
    )
    bias = torch.tensor([0.0, 0.0, 0.0, 1.0])
    instance = _grouped_router(
        scoring_func="sigmoid",
        routed_scaling_factor=2.0,
        e_score_correction_bias=bias,
    )
    gating = torch.arange(8, dtype=torch.float32).view(2, 4)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Mod, "_grouped_topk", recorder)
        mp.setattr(
            Mod.rocm_aiter_ops,
            "is_fused_moe_enabled",
            lambda: False,
            raising=False,
        )
        weights, ids = Mod.GroupedTopKRouterFL._compute_routing(
            instance, torch.zeros(2, 3), gating, None
        )
    args = recorder.calls[0][0]
    assert args[0] is gating
    assert args[1:6] == (2, 1, 2, False, 2.0)
    assert args[6] is bias
    assert args[7] == 1
    assert weights.tolist() == [[0.5, 0.5], [0.5, 0.5]]
    assert ids.tolist() == [[1, 1], [1, 1]]


def test_grouped_router_valid_grouping_uses_aiter_when_fused():
    """The ROCm AITER path binds the fused-shared-experts count."""
    recorder = _KernelRecorder(
        lambda **kw: (torch.zeros(2, 1), torch.zeros(2, 1, dtype=torch.long))
    )
    instance = _grouped_router(num_fused_shared_experts=0)
    gating = torch.zeros(2, 4)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Mod.rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
        mp.setattr(
            Mod.rocm_aiter_ops,
            "is_fusion_moe_shared_experts_enabled",
            lambda: False,
        )
        mp.setattr(Mod, "rocm_aiter_grouped_topk", recorder)
        Mod.GroupedTopKRouterFL._compute_routing(
            instance, torch.zeros(2, 3), gating, torch.int32
        )
    kwargs = recorder.calls[0][1]
    assert kwargs["num_fused_shared_experts"] == 0
    assert kwargs["topk"] == 2
    assert kwargs["num_expert_group"] == 2


def test_grouped_router_aiter_requires_zero_shared_experts_without_fusion():
    """AITER without shared-expert fusion rejects nonzero shared experts."""
    instance = _grouped_router(num_fused_shared_experts=2)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Mod.rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)
        mp.setattr(
            Mod.rocm_aiter_ops,
            "is_fusion_moe_shared_experts_enabled",
            lambda: False,
        )
        with pytest.raises(AssertionError):
            Mod.GroupedTopKRouterFL._compute_routing(
                instance, torch.zeros(2, 3), torch.zeros(2, 4), None
            )


class _BiasKernel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return torch.full((2, 2), 0.5), torch.zeros(2, 2, dtype=torch.long)


def test_bias_router_forwards_bias_data_and_scales_weights():
    """The bias router passes the raw bias tensor and applies the factor."""
    kernel = _BiasKernel()
    bias = torch.tensor([0.0, 0.0, 0.0, 0.0])
    instance = _router(
        Mod.FusedTopKBiasRouterFL,
        top_k=2,
        renormalize=True,
        scoring_func="softmax",
        routed_scaling_factor=4.0,
        e_score_correction_bias=bias,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Mod, "fused_topk_bias", kernel)
        weights, _ = Mod.FusedTopKBiasRouterFL._compute_routing(
            instance, torch.zeros(2, 3), torch.zeros(2, 4), torch.int64
        )
    call = kernel.calls[0]
    torch.testing.assert_close(call["e_score_correction_bias"], bias)
    assert call["indices_type"] is torch.int64
    assert weights.tolist() == [[2.0, 2.0], [2.0, 2.0]]


def test_bias_router_without_bias_or_scaling_leaves_weights_unchanged():
    """No bias sends None and a unit factor leaves the kernel values alone."""
    kernel = _BiasKernel()
    instance = _router(
        Mod.FusedTopKBiasRouterFL,
        top_k=2,
        renormalize=False,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        e_score_correction_bias=None,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Mod, "fused_topk_bias", kernel)
        weights, _ = Mod.FusedTopKBiasRouterFL._compute_routing(
            instance, torch.zeros(2, 3), torch.zeros(2, 4), None
        )
    assert kernel.calls[0]["e_score_correction_bias"] is None
    assert weights.tolist() == [[0.5, 0.5], [0.5, 0.5]]


def test_replace_router_with_fl_installs_every_fl_method_once():
    """The installer rebinds all three routers and is idempotent."""
    saved = {
        Mod.FusedTopKRouter: Mod.FusedTopKRouter._compute_routing,
        Mod.GroupedTopKRouter: Mod.GroupedTopKRouter._compute_routing,
        Mod.FusedTopKBiasRouter: Mod.FusedTopKBiasRouter._compute_routing,
    }
    try:
        Mod.replace_router_with_fl()
        assert (
            Mod.FusedTopKRouter._compute_routing
            is Mod.FusedTopKRouterFL._compute_routing
        )
        assert (
            Mod.GroupedTopKRouter._compute_routing
            is Mod.GroupedTopKRouterFL._compute_routing
        )
        assert (
            Mod.FusedTopKBiasRouter._compute_routing
            is Mod.FusedTopKBiasRouterFL._compute_routing
        )
        Mod.replace_router_with_fl()
        assert (
            Mod.FusedTopKRouter._compute_routing
            is Mod.FusedTopKRouterFL._compute_routing
        )
    finally:
        for cls, method in saved.items():
            cls._compute_routing = method


def test_grouped_router_reports_invalid_grouping_with_equal_expert_count():
    """num_experts == num_expert_group is invalid (strictly greater required)."""
    captured = {}

    def fake_fused_topk(hidden_states, gating_output, topk, renormalize, indices_type):
        captured.update(topk=topk, renormalize=renormalize)
        return torch.zeros(1, topk), torch.zeros(1, topk, dtype=torch.int32), None

    instance = _grouped_router(num_expert_group=4, topk_group=1, top_k=1)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Mod, "fused_topk", fake_fused_topk)
        Mod.GroupedTopKRouterFL._compute_routing(
            instance,
            torch.zeros(1, 3),
            torch.zeros(1, 4),
            None,
        )
    assert captured == {"topk": 1, "renormalize": False}
