# SPDX-License-Identifier: Apache-2.0
"""Focused tests for HY4 model integration contracts."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from vllm_fl.models import hy_v4


def test_hy4_prefill_topk_uses_shared_capability(monkeypatch):
    captured = {}

    def fake_topk(logits, row_starts, row_ends, indices):
        captured["logits"] = logits.clone()
        captured["row_starts"] = row_starts.clone()
        captured["row_ends"] = row_ends.clone()
        indices.fill_(-1)
        indices[0, :2].copy_(torch.tensor([2, 0], dtype=torch.int32))

    monkeypatch.setattr(hy_v4, "_top_k_per_row_prefill", fake_topk)
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16)
    weights = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    keys = torch.tensor([[1.0, 1.0], [2.0, -1.0], [3.0, 4.0]], dtype=torch.bfloat16)
    output = torch.empty(4, dtype=torch.int32)

    hy_v4._select_topk(q, weights, keys, topk=4, output=output)

    assert captured["logits"].shape == (1, 3)
    assert captured["logits"].dtype == torch.float32
    assert captured["row_starts"].tolist() == [0]
    assert captured["row_ends"].tolist() == [3]
    assert output.tolist() == [2, 0, -1, -1]


def test_hy4_blocked_prefill_topk_matches_row_reference():
    torch.manual_seed(20260914)
    q = torch.randn((5, 2, 4), dtype=torch.float32)
    weights = torch.randn((5, 2), dtype=torch.float32)
    keys = torch.randn((12, 4), dtype=torch.float32)
    starts = torch.tensor([2, 2, 4, 6, 6], dtype=torch.int32)
    ends = torch.tensor([2, 5, 8, 8, 12], dtype=torch.int32)
    topk = 4
    request_start = 2
    output = torch.empty((5, topk), dtype=torch.int32)

    hy_v4._select_topk_block(
        q,
        weights,
        keys,
        starts,
        ends,
        key_start=2,
        key_end=12,
        request_start=request_start,
        topk=topk,
        output=output,
    )

    expected = torch.full_like(output, -1)
    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        count = min(topk, end - start)
        if count == 0:
            continue
        scores = torch.matmul(keys[start:end], q[row].transpose(0, 1))
        logits = (
            torch.relu(scores) * weights[row].to(scores.dtype).unsqueeze(0)
        ).sum(dim=-1).float()
        selected = torch.topk(logits, count, sorted=True).indices.to(torch.int32)
        expected[row, :count].copy_(selected + start - request_start)

    assert torch.equal(output.sort(dim=1).values, expected.sort(dim=1).values)
    assert output[0].tolist() == [-1, -1, -1, -1]


def test_hy4_blocked_prefill_topk_empty_block():
    output = torch.empty((0, 4), dtype=torch.int32)
    hy_v4._select_topk_block(
        torch.empty((0, 2, 4)),
        torch.empty((0, 2)),
        torch.empty((0, 4)),
        torch.empty((0,), dtype=torch.int32),
        torch.empty((0,), dtype=torch.int32),
        key_start=0,
        key_end=0,
        request_start=0,
        topk=4,
        output=output,
    )
    assert output.shape == (0, 4)


class _FakeGate(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _FakeSharedExperts(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _FakeRoutedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_num_experts = 8


class _FakeMoERunner(nn.Module):
    def __init__(self):
        super().__init__()
        self.routed_experts = _FakeRoutedExperts()


def test_hy4_moe_resolves_patched_factory_at_construction(monkeypatch):
    calls = []

    def patched_factory(*args, **kwargs):
        runner = _FakeMoERunner()
        calls.append((runner, args, kwargs))
        return runner

    monkeypatch.setattr(hy_v4, "GateLinear", _FakeGate)
    monkeypatch.setattr(hy_v4, "HYV4DenseMLP", _FakeSharedExperts)
    monkeypatch.setattr(hy_v4.fused_moe, "FusedMoE", patched_factory)

    config = SimpleNamespace(
        hidden_act="silu",
        hidden_size=16,
        n_routed_experts=8,
        moe_intermediate_size=32,
        n_shared_experts=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
    )
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
        enable_eplb=False,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        quant_config=None,
    )

    layer = hy_v4.HYV4MoE(config, vllm_config, prefix="model.layers.0.mlp")

    assert len(calls) == 1
    assert layer.experts is calls[0][0]
    assert calls[0][2]["num_experts"] == config.n_routed_experts
    assert layer.n_local_physical_experts == 8
