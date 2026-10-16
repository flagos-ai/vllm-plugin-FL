"""Core Qwen3.8 framework contracts; numerical tests live in FlagGems-vllm."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn  # noqa: E402

from vllm_fl.models.qwen3_8_flash_next.gpu import model as qwen_model  # noqa: E402
from vllm_fl.models.qwen3_8_flash_next.gpu.ple_layer import (
    Qwen3_8FlashNextNGramEmbedding,
    validate_ple_embedding_weights,
)


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

    mapped = list(qwen_model._map_stacked_weights(weights))
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
    handled, names = qwen_model._load_fused_expert_weight(
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
    handled, names = qwen_model._load_fused_expert_weight(
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


def make_loader(start=0, end=8, vocab_size=8, parts=2):
    module = Qwen3_8FlashNextNGramEmbedding.__new__(Qwen3_8FlashNextNGramEmbedding)
    torch.nn.Module.__init__(module)
    module.split_ngram_parts = parts
    for name in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
        module.register_buffer(name, torch.zeros(2, dtype=torch.long))
    embedding = torch.nn.Module()
    embedding.org_vocab_size = vocab_size
    embedding.embedding_dim = 2
    embedding.shard_indices = SimpleNamespace(
        org_vocab_start_index=start, org_vocab_end_index=end
    )
    embedding.weight = torch.nn.Parameter(
        torch.full((end - start, 2), -999.0), requires_grad=False
    )
    module.ngram_embedding = embedding
    return module


def shards(vocab_size=8, parts=2):
    weight = torch.arange(vocab_size * 2, dtype=torch.float32).reshape(-1, 2)
    size = (vocab_size + parts - 1) // parts
    return weight, [
        (f"ngram_embedding.shard_{i}.weight", weight[i * size : (i + 1) * size])
        for i in range(parts)
    ]


class FragmentedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = make_loader()
        self.second = make_loader()

    @validate_ple_embedding_weights
    def load_weights(self, weights):
        from vllm.model_executor.models.utils import AutoWeightsLoader

        return AutoWeightsLoader(self).load_weights(weights)


def fragmented_weights():
    _, weights = shards()
    return [
        (f"{owner}.{name}", value)
        for name, value in weights
        for owner in ("first", "second")
    ]


def test_real_autoloader_accumulates_noncontiguous_module_fragments():
    model = FragmentedModel()
    assert model.load_weights(fragmented_weights()) == {
        "first.ngram_embedding.weight",
        "second.ngram_embedding.weight",
    }
    expected, _ = shards()
    torch.testing.assert_close(model.first.ngram_embedding.weight, expected)
    torch.testing.assert_close(model.second.ngram_embedding.weight, expected)
    # A subsequent complete model reload starts a new transaction.
    assert len(model.load_weights(fragmented_weights())) == 2


@pytest.mark.parametrize("failure", ["missing", "duplicate", "mixed"])
def test_fragmented_load_still_rejects_incomplete_or_ambiguous_checkpoints(failure):
    weights = fragmented_weights()
    if failure == "missing":
        weights.pop()
    elif failure == "duplicate":
        weights.append(weights[0])
    else:
        full, _ = shards()
        weights.append(("first.ngram_embedding.weight", full))
    with pytest.raises(ValueError, match="Missing|Duplicate|Mixed"):
        FragmentedModel().load_weights(weights)
