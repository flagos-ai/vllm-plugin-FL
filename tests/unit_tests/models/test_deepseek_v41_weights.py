# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared-expert fusion must not rewrite the distinct DSpark Markov factors."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm_fl.models.deepseek_v41 import weights


def test_shared_fusion_preserves_markov_factor_names(monkeypatch):
    monkeypatch.setattr(weights, "expert_mapping", lambda model: [])
    monkeypatch.setattr(weights, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(weights, "get_tensor_model_parallel_rank", lambda: 0)
    model = nn.Module()
    model.config = SimpleNamespace(n_routed_experts=1)
    model.model = nn.Module()
    model.model.markov_head = nn.Module()
    for name in ["markov_w1", "markov_w2"]:
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.empty(2, 3))
        setattr(model.model.markov_head, name, layer)
    block = nn.Module()
    block.ffn = nn.Module()
    block.ffn.shared_experts = nn.Module()
    fused = nn.Module()
    fused.weight = nn.Parameter(torch.empty(4, 3))

    def load_fused(parameter, value, shard):
        parameter.data[shard * 2 : (shard + 1) * 2].copy_(value)

    fused.weight.weight_loader = load_fused
    block.ffn.shared_experts.gate_up_proj = fused
    model.model.layers = nn.ModuleList([block])
    loaded = weights.load_text_model_weights(
        model,
        [
            ("model.markov_head.markov_w1.weight", torch.full((2, 3), 11.0)),
            ("model.layers.0.ffn.shared_experts.w1.weight", torch.full((2, 3), 3.0)),
            ("model.markov_head.markov_w2.weight", torch.full((2, 3), 13.0)),
            ("model.layers.0.ffn.shared_experts.w3.weight", torch.full((2, 3), 7.0)),
        ],
    )
    assert len(loaded) == 3
    torch.testing.assert_close(
        model.model.markov_head.markov_w1.weight, torch.full((2, 3), 11.0)
    )
    torch.testing.assert_close(
        model.model.markov_head.markov_w2.weight, torch.full((2, 3), 13.0)
    )
    torch.testing.assert_close(
        fused.weight,
        torch.tensor(
            [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [7.0, 7.0, 7.0], [7.0, 7.0, 7.0]]
        ),
    )
