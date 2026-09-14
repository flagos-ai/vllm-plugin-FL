"""Focused tests for the vLLM 0.24 fused-expert loader compatibility."""

from __future__ import annotations

import pytest

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from vllm_fl.models.qwen3_8_flash_next.gpu import model as qwen_model  # noqa: E402


def test_legacy_stacked_mapping_covers_attention_gdn_and_shared_expert():
    weights = [
        ("layers.0.self_attn.q_proj.weight", torch.ones(1)),
        ("layers.0.self_attn.k_proj.weight", torch.ones(1)),
        ("layers.0.self_attn.v_proj.weight", torch.ones(1)),
        ("layers.0.linear_attn.in_proj_qkv.weight", torch.ones(1)),
        ("layers.0.linear_attn.in_proj_z.weight", torch.ones(1)),
        ("layers.0.linear_attn.in_proj_b.weight", torch.ones(1)),
        ("layers.0.linear_attn.in_proj_a.weight", torch.ones(1)),
        ("layers.0.mlp.shared_expert.gate_proj.weight", torch.ones(1)),
        ("layers.0.mlp.shared_expert.up_proj.weight", torch.ones(1)),
        # Router and PLE names are intentionally passed through unchanged.
        ("layers.0.mlp.gate.weight", torch.ones(1)),
        ("layers.0.ple.key_proj.weight", torch.ones(1)),
    ]

    mapped = list(qwen_model._map_vllm024_stacked_weights(weights))
    names = [name for name, _ in mapped]
    assert names[:3] == [
        "layers.0.self_attn.qkv_proj.weight",
        "layers.0.self_attn.qkv_proj.weight",
        "layers.0.self_attn.qkv_proj.weight",
    ]
    assert names[3:7] == [
        "layers.0.linear_attn.in_proj_qkvz.weight",
        "layers.0.linear_attn.in_proj_qkvz.weight",
        "layers.0.linear_attn.in_proj_ba.weight",
        "layers.0.linear_attn.in_proj_ba.weight",
    ]
    assert names[7:9] == [
        "layers.0.mlp.shared_expert.gate_up_proj.weight",
        "layers.0.mlp.shared_expert.gate_up_proj.weight",
    ]
    assert names[9:] == [
        "layers.0.mlp.gate.weight",
        "layers.0.ple.key_proj.weight",
    ]
    assert [getattr(weight, "shard_id", None) for _, weight in mapped[:9]] == [
        "q",
        "k",
        "v",
        (0, 1, 2),
        3,
        0,
        1,
        0,
        1,
    ]
    assert all(not hasattr(weight, "shard_id") for _, weight in mapped[9:])


def test_legacy_fused_experts_split_gate_up_and_load_every_expert():
    loaded: list[tuple[str, int, torch.Tensor]] = []

    def weight_loader(
        _param,
        value,
        _name,
        *,
        shard_id,
        expert_id,
        return_success=True,
    ):
        assert return_success is True
        loaded.append((shard_id, expert_id, value.clone()))
        return True

    w13 = nn.Parameter(torch.empty(2, 4, 3))
    w2 = nn.Parameter(torch.empty(2, 3, 2))
    w13.weight_loader = weight_loader
    w2.weight_loader = weight_loader
    params = {
        "layers.0.mlp.experts.routed_experts.w13_weight": w13,
        "layers.0.mlp.experts.routed_experts.w2_weight": w2,
    }
    mapping = [
        ("experts.routed_experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
        ("experts.routed_experts.w2_weight", "experts.down_proj", 0, "w2"),
    ]
    gate_up = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    handled, names = qwen_model._load_vllm024_fused_expert_weight(
        "layers.0.mlp.experts.gate_up_proj",
        gate_up,
        params,
        mapping,
        num_experts=2,
    )

    assert handled
    assert names == {"layers.0.mlp.experts.routed_experts.w13_weight"}
    assert [(shard, expert) for shard, expert, _ in loaded] == [
        ("w1", 0),
        ("w1", 1),
        ("w3", 0),
        ("w3", 1),
    ]
    torch.testing.assert_close(loaded[0][2], gate_up[0, :2])
    torch.testing.assert_close(loaded[3][2], gate_up[1, 2:])

    down = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    handled, names = qwen_model._load_vllm024_fused_expert_weight(
        "layers.0.mlp.experts.down_proj",
        down,
        params,
        mapping,
        num_experts=2,
    )
    assert handled
    assert names == {"layers.0.mlp.experts.routed_experts.w2_weight"}
    assert [(shard, expert) for shard, expert, _ in loaded[-2:]] == [
        ("w2", 0),
        ("w2", 1),
    ]
    torch.testing.assert_close(loaded[-1][2], down[1])
