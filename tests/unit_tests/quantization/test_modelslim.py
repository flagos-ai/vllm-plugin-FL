# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from safetensors.torch import save_file

import vllm.config  # noqa: F401 - initialize the public config entry first.

from vllm_fl.quantization.modelslim import ModelSlimConfig


@pytest.fixture
def checkpoint(tmp_path):
    config = {
        "model_type": "deepseek_v41",
        "text_config": {
            "hidden_size": 64,
            "num_hidden_layers": 40,
            "num_nextn_predict_layers": 3,
        },
        "engram_rotation_config": {
            "value_projection_rotated": True,
            "value_basis": "quarot_global",
            "key_and_gate_basis": "original",
            "runtime_delta_rotation": False,
        },
    }
    desc = {
        "version": "1.0.0",
        "group_size": 0,
        "optional": {
            "embedding_storage": {"version": 2, "modules": {}},
            "quarot": {
                "rotation_map": {"global_rotation": "optional/quarot.safetensors"}
            },
        },
    }
    for scope, kind in [("layers.0", "W4A8_DYNAMIC"), ("mtp.0", "FLOAT")]:
        for part in ("w1", "w2", "w3"):
            desc[f"{scope}.ffn.experts.0.{part}.weight"] = kind
    desc.update(
        {
            "layers.0.attn.wq_a.weight": "FLOAT",
            "layers.0.attn.wkv.weight": "FLOAT",
            "layers.0.attn.wq_b.weight": "W8A8_DYNAMIC",
            "mtp.0.main_proj.weight": "FLOAT",
            "layers.0.ffn.experts.0.w1.weight_offset": "W4A8_DYNAMIC",
        }
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "quant_model_description.json").write_text(json.dumps(desc))
    (tmp_path / "optional").mkdir()
    save_file(
        {"global_rotation": torch.eye(64)}, tmp_path / "optional/quarot.safetensors"
    )
    key = "layers.0.attn.wq_a.weight"
    save_file({key: torch.zeros(4, 64)}, tmp_path / "before.safetensors")
    save_file({key: torch.ones(4, 64)}, tmp_path / "final.safetensors")
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "final.safetensors"}})
    )
    return tmp_path


def test_original_format_rules_and_draft_mapping(checkpoint):
    cfg = ModelSlimConfig.from_config({"quant_method": "ascend"})
    cfg.maybe_update_config(str(checkpoint))
    assert cfg.layer_kind("language_model.model.layers.0.ffn.experts") == "W4A8_DYNAMIC"
    assert cfg.layer_kind("model.layers.40.ffn.experts") == "FLOAT"
    assert cfg.layer_kind("model.layers.0.attn.fused_wqa_wkv") == "FLOAT"
    assert cfg.layer_kind("model.layers.0.attn.wq_b") == "W8A8_DYNAMIC"
    assert cfg.layer_kind("model.main_proj") == "FLOAT"
    assert cfg.canonical_name(
        "model.layers.0.ffn.experts.0.w1.weight", draft=True
    ).startswith("mtp.0.")
    with pytest.raises(ValueError, match="missing"):
        cfg.layer_kind("model.layers.0.attn.unknown")
    with pytest.raises(NotImplementedError, match="zero offsets"):
        cfg.auxiliary("model.layers.0.ffn.experts.0.w1.weight_offset", torch.ones(2, 1))


def test_conflicting_fusion_rejected(checkpoint):
    cfg = ModelSlimConfig.from_config({"quant_method": "ascend"})
    cfg.maybe_update_config(str(checkpoint))
    cfg.description["layers.0.attn.wkv.weight"] = "W8A8_DYNAMIC"
    with pytest.raises(NotImplementedError, match="conflicting"):
        cfg.layer_kind("model.layers.0.attn.fused_wqa_wkv")


def test_unknown_format_is_not_float(checkpoint):
    path = checkpoint / "quant_model_description.json"
    description = json.loads(path.read_text())
    description["layers.0.attn.wq_a.weight"] = "UNKNOWN"
    path.write_text(json.dumps(description))
    with pytest.raises(NotImplementedError, match="unsupported"):
        ModelSlimConfig.from_config({"quant_method": "ascend"}).maybe_update_config(
            str(checkpoint)
        )


def test_duplicate_tensor_uses_index_not_iterator_order(checkpoint):
    cfg = ModelSlimConfig.from_config({"quant_method": "ascend"})
    cfg.maybe_update_config(str(checkpoint))
    key = "layers.0.attn.wq_a.weight"
    for first, last in [
        (torch.zeros(4, 64), torch.ones(4, 64)),
        (torch.ones(4, 64), torch.zeros(4, 64)),
    ]:
        rows = list(cfg.authoritative_weights([(key, first), (key, last)]))
        assert len(rows) == 1 and rows[0][0] == key
        assert torch.equal(rows[0][1], torch.ones(4, 64))
