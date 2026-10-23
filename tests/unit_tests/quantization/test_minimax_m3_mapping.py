# Copyright (c) 2026 BAAI. All rights reserved.

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
    should_ignore_layer,
)

from vllm_fl.models.minimax_m3_registration import add_fused_quantization_mapping


@pytest.mark.parametrize(
    "prefix,fused,parts",
    [
        (
            "language_model.model.layers.0.self_attn", "qkv_proj",
            ["q_proj", "k_proj", "v_proj"],
        ),
        (
            "language_model.model.layers.0.mlp", "gate_up_proj",
            ["gate_proj", "up_proj"],
        ),
        (
            "language_model.model.layers.3.block_sparse_moe.shared_experts",
            "gate_up_proj", ["gate_proj", "up_proj"],
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
    assert find_matched_target(
        name, layer, targets, Model.packed_modules_mapping
    ) == targets[0]


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
