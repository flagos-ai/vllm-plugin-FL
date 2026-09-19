# SPDX-License-Identifier: Apache-2.0
"""Use the real GLM/DeepSeek loader with small CPU destination parameters."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_fl.model_loader.glm5_next import audit_packed_weights
from vllm_fl.models.glm5_next import Glm5NextForCausalLM, Glm5NextModel
from vllm_fl.models.glm5_next_multimodal import Glm5NextForConditionalGeneration


class _Fixture(nn.Module):
    load_weights = Glm5NextModel.load_weights

    def __init__(self, ep=False):
        super().__init__()
        self.config = SimpleNamespace(
            n_routed_experts=2,
            n_shared_experts=0,
            num_hidden_layers=1,
            num_nextn_predict_layers=0,
        )
        self.use_mha = False
        self.num_redundant_experts = 0
        layer = nn.Module()
        self.layers = nn.ModuleList([layer])
        layer.mlp = nn.Module()
        layer.mlp.gate_up_proj = nn.Module()
        param = nn.Parameter(torch.full((4, 2), -99.0), requires_grad=False)

        def packed_loader(param, value, shard_id=None):
            if shard_id is None:
                param.copy_(value)
            else:
                param[shard_id * 2 : (shard_id + 1) * 2].copy_(value)

        param.weight_loader = packed_loader
        layer.mlp.gate_up_proj.weight = param
        layer.mlp.experts = nn.Module()
        experts = nn.Module()
        layer.mlp.experts.routed_experts = experts
        experts.expert_map = torch.tensor([-1, 0]) if ep else None
        nlocal = 1 if ep else 2
        for name, width in (("w13_weight", 4), ("w2_weight", 2)):
            param = nn.Parameter(
                torch.full((nlocal, width, 2), -99.0), requires_grad=False
            )

            def expert_loader(
                param, value, name, *, shard_id, expert_id, return_success
            ):
                if ep and expert_id == 0:
                    return False
                local_id = 0 if ep else expert_id
                if shard_id == "w2":
                    param[local_id].copy_(value)
                else:
                    start = 0 if shard_id == "w1" else 2
                    param[local_id, start : start + 2].copy_(value)
                return True

            param.weight_loader = expert_loader
            setattr(experts, name, param)


def _weights():
    result = [
        (f"layers.0.mlp.{part}_proj.weight", torch.full((2, 2), float(i)))
        for i, part in enumerate(("gate", "up"), 1)
    ]
    result += [
        (
            f"layers.0.mlp.experts.{expert}.{part}_proj.weight",
            torch.full((2, 2), float(i + expert)),
        )
        for expert in range(2)
        for i, part in enumerate(("gate", "up", "down"), 3)
    ]
    return result


@pytest.mark.parametrize("ep", [False, True])
def test_complete_packed_and_expert_weights(ep):
    model = _Fixture(ep)
    originals = [p.weight_loader for p in model.parameters()]
    with audit_packed_weights(model):
        loaded = model.load_weights(iter(_weights()))
    assert set(dict(model.named_parameters())) <= loaded
    assert all(not (p == -99).any() for p in model.parameters())
    assert all(p.weight_loader is old for p, old in zip(model.parameters(), originals))


@pytest.mark.parametrize(
    "missing",
    [
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.experts.1.up_proj.weight",
        "layers.0.mlp.experts.1.down_proj.weight",
    ],
)
def test_missing_shard_fails_even_when_destination_name_is_loaded(missing):
    model = _Fixture()
    originals = [p.weight_loader for p in model.parameters()]
    with pytest.raises(RuntimeError, match="missing packed weight shards"):
        with audit_packed_weights(model):
            model.load_weights((n, w) for n, w in _weights() if n != missing)
    assert all(p.weight_loader is old for p, old in zip(model.parameters(), originals))


def test_undeclared_fp8_rejected_before_copy():
    model = _Fixture()
    name, value = _weights()[0]
    with pytest.raises(ValueError, match="unquantized checkpoint required"):
        model.load_weights([(name, value.to(torch.float8_e4m3fn))])
    assert (model.layers[0].mlp.gate_up_proj.weight == -99).all()


def _causal_model():
    model = Glm5NextForCausalLM.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(tie_word_embeddings=False)
    model.model = _Fixture()
    model.lm_head = nn.Linear(2, 2, bias=False)
    model.finalizations = 0

    def finalize():
        model.finalizations += 1

    model.model.finalize_mhc_broadcast_weights = finalize
    return model


def _interleaved_weights(multimodal):
    prefix = "model.language_model." if multimodal else "model."
    weights = [(prefix + n, w) for n, w in _weights()]
    # File 45 starts with lm_head, interrupting the text subtree.
    weights.insert(4, ("lm_head.weight", torch.ones(2, 2)))
    if multimodal:
        # A separate vision file can also interrupt the language-model prefix.
        weights.insert(2, ("model.visual.weight", torch.ones(2, 2)))
    return weights


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("missing_up", [False, True])
def test_checkpoint_audit_spans_interleaved_subtree_calls(multimodal, missing_up):
    causal = _causal_model()
    model = causal
    if multimodal:
        model = Glm5NextForConditionalGeneration.__new__(
            Glm5NextForConditionalGeneration
        )
        nn.Module.__init__(model)
        model.language_model = causal
        model.visual = nn.Linear(2, 2, bias=False)
    weights = _interleaved_weights(multimodal)
    if missing_up:
        weights = [
            (n, w) for n, w in weights if not n.endswith("experts.1.up_proj.weight")
        ]
    originals = [p.weight_loader for p in causal.model.parameters()]
    if missing_up:
        with pytest.raises(RuntimeError, match="missing packed weight shards"):
            model.load_weights(iter(weights))
        assert causal.finalizations == 0
    else:
        loaded = model.load_weights(iter(weights))
        assert set(dict(model.named_parameters())) <= loaded
        assert all(not (p == -99).any() for p in causal.model.parameters())
        assert causal.finalizations == 1
    assert not hasattr(causal, "_glm5_loading_names")
    assert all(
        p.weight_loader is old for p, old in zip(causal.model.parameters(), originals)
    )


def test_interleaved_duplicate_shard_rejected():
    model = _causal_model()
    weights = _interleaved_weights(False)
    weights.append(weights[0])
    with pytest.raises(ValueError, match="Duplicate GLM5-Next packed weight"):
        model.load_weights(iter(weights))
    assert model.finalizations == 0
    assert not hasattr(model, "_glm5_loading_names")
