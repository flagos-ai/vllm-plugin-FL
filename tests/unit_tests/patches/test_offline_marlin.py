from vllm_fl.patches import offline_marlin

import torch


def test_offline_marlin_env(monkeypatch):
    monkeypatch.delenv("VLLM_FL_OFFLINE_MARLIN", raising=False)
    assert offline_marlin.offline_marlin_enabled() is False
    monkeypatch.setenv("VLLM_FL_OFFLINE_MARLIN", "yes")
    assert offline_marlin.offline_marlin_enabled() is True


def test_flaggems_fused_marlin_env(monkeypatch):
    monkeypatch.delenv("VLLM_FL_FLAGGEMS_FUSED_MARLIN", raising=False)
    assert offline_marlin.flaggems_fused_marlin_enabled() is False
    monkeypatch.setenv("VLLM_FL_FLAGGEMS_FUSED_MARLIN", "1")
    assert offline_marlin.flaggems_fused_marlin_enabled() is True


def test_prepare_flaggems_packed_tensors_transposes_before_byte_view():
    class Layer:
        pass

    layer = Layer()
    layer.w13_weight_packed = torch.arange(2 * 3 * 4, dtype=torch.int32).view(2, 3, 4)
    layer.w2_weight_packed = torch.arange(2 * 2 * 3, dtype=torch.int32).view(2, 2, 3)
    layer.w13_weight_scale = torch.arange(2 * 3 * 4).view(2, 3, 4)
    layer.w2_weight_scale = torch.arange(2 * 2 * 3).view(2, 2, 3)

    result = offline_marlin._prepare_flaggems_packed_tensors(layer)

    assert result["w13_weight_packed"].shape == (2, 4, 12)
    assert result["w13_weight_packed"].dtype == torch.uint8
    assert result["w2_weight_packed"].shape == (2, 3, 8)
    assert result["w13_weight_scale"].shape == (2, 4, 3)
    assert result["w2_weight_scale"].shape == (2, 3, 2)


def test_layer_checkpoint_prefix_accepts_model_prefix():
    key = "model.layers.0.ffn.experts.routed_experts.w13_weight"
    index = {key: "rank-0.safetensors"}

    assert offline_marlin._layer_checkpoint_prefix(
        "layers.0.ffn.experts.routed_experts", index
    ) == "model.layers.0.ffn.experts.routed_experts"
