# SPDX-License-Identifier: Apache-2.0
"""Focused tests for HY4 model integration contracts."""

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from vllm_fl.models import hy_v4


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
