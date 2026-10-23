# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
    should_ignore_layer,
)

from vllm_fl.models.minimax_m3_registration import (
    add_fused_quantization_mapping,
    register_minimax_m3_support,
)


@pytest.mark.parametrize(
    "prefix,fused,parts",
    [
        (
            "language_model.model.layers.0.self_attn",
            "qkv_proj",
            ["q_proj", "k_proj", "v_proj"],
        ),
        (
            "language_model.model.layers.0.mlp",
            "gate_up_proj",
            ["gate_proj", "up_proj"],
        ),
        (
            "language_model.model.layers.3.block_sparse_moe.shared_experts",
            "gate_up_proj",
            ["gate_proj", "up_proj"],
        ),
    ],
)
def test_named_quantization_targets_survive_fusion(prefix, fused, parts):
    class Model:
        pass

    layer = torch.nn.Linear(4, 8, bias=False)
    targets = [f"{prefix}.{part}" for part in parts]
    name = f"{prefix}.{fused}"
    assert find_matched_target(name, layer, targets) is None
    add_fused_quantization_mapping(Model)
    assert (
        find_matched_target(name, layer, targets, Model.packed_modules_mapping)
        == targets[0]
    )


def test_mapping_preserves_vision_exclusions_and_upstream_metadata():
    class Base:
        packed_modules_mapping = {"other_proj": ["left_proj", "right_proj"]}

    class Model(Base):
        pass

    add_fused_quantization_mapping(Model)
    add_fused_quantization_mapping(Model)
    assert Base.packed_modules_mapping == {"other_proj": ["left_proj", "right_proj"]}
    assert Model.packed_modules_mapping["other_proj"] == ["left_proj", "right_proj"]
    prefix = "vision_tower.vision_model.encoder.layers.0.self_attn"
    ignored = [f"{prefix}.{part}" for part in ["q_proj", "k_proj", "v_proj"]]
    assert should_ignore_layer(
        f"{prefix}.qkv_proj", ignored, Model.packed_modules_mapping
    )


def test_existing_fusion_mapping_is_preserved():
    class Model:
        packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"],
        }

    original = Model.packed_modules_mapping
    add_fused_quantization_mapping(Model)
    assert Model.packed_modules_mapping["qkv_proj"] is original["qkv_proj"]
    assert "gate_up_proj" not in original
    completed = Model.packed_modules_mapping
    add_fused_quantization_mapping(Model)
    assert Model.packed_modules_mapping is completed


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize(
    "available",
    [
        [],
        ["MiniMaxM3SparseForCausalLM"],
        ["MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"],
    ],
)
def test_registration_is_lazy_and_platform_scoped(monkeypatch, cuda, available):
    import vllm.platforms as platforms
    from vllm.model_executor.models import ModelRegistry

    calls = []
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_cuda=lambda: cuda)
    )
    monkeypatch.setattr(ModelRegistry, "get_supported_archs", lambda: available)
    monkeypatch.setattr(
        ModelRegistry, "register_model", lambda *args: calls.append(args)
    )
    register_minimax_m3_support()
    assert calls == [
        (name, f"vllm_fl.models.minimax_m3:{name}") for name in available if cuda
    ]
