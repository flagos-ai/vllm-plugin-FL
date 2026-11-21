# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Verify the actual FL router's cached native alias reaches the selected API."""

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.metax import patches
from vllm_fl.dispatch.backends.vendor.metax.patches import sqrtsoftplus_router as patch
from vllm_fl.ops.fused_moe import router as fl


def test_actual_fl_router_alias_and_idempotency(monkeypatch):
    patches._install_v41_patches()
    selected = patch.biased.fused_topk_bias
    patches._install_v41_patches()
    assert fl.fused_topk_bias is selected
    assert patch.grouped.fused_topk_bias is selected
    calls = []

    def operator(logits, bias, topk, **kwargs):
        calls.append((logits.shape, kwargs["input_ids"].clone()))
        return torch.full((2, 6), 0.125), torch.zeros((2, 6), dtype=torch.int32)

    def resolve(name):
        assert name == "sqrtsoftplus_topk"
        return operator

    monkeypatch.setattr(patch, "resolve_op", resolve)
    instance = fl.FusedTopKBiasRouterFL(
        top_k=6,
        global_num_experts=384,
        e_score_correction_bias=torch.zeros(384),
        scoring_func="sqrtsoftplus",
        bias_vl=torch.ones(384),
        image_sentinel_lo=129264,
    )
    tokens = torch.tensor([42, 129264])
    w, i = instance._compute_routing(
        torch.zeros((2, 5120)), torch.zeros((2, 384)), torch.int32, input_ids=tokens
    )
    assert len(calls) == 1 and calls[0][0] == (2, 384)
    torch.testing.assert_close(calls[0][1], tokens)
    torch.testing.assert_close(w, torch.full((2, 6), 0.125))


def test_alias_preflight_failure_keeps_native_binding(monkeypatch):
    selected = patch.biased.fused_topk_bias
    monkeypatch.setattr(fl, "fused_topk_bias", lambda: None)
    with pytest.raises(RuntimeError, match="alias contract"):
        patches._install_v41_patches()
    assert patch.biased.fused_topk_bias is selected
