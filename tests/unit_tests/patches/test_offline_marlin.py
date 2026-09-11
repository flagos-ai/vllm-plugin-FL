from vllm_fl.patches import offline_marlin


def test_offline_marlin_env(monkeypatch):
    monkeypatch.delenv("VLLM_FL_OFFLINE_MARLIN", raising=False)
    assert offline_marlin.offline_marlin_enabled() is False
    monkeypatch.setenv("VLLM_FL_OFFLINE_MARLIN", "yes")
    assert offline_marlin.offline_marlin_enabled() is True


def test_layer_checkpoint_prefix_accepts_model_prefix():
    key = "model.layers.0.ffn.experts.routed_experts.w13_weight"
    index = {key: "rank-0.safetensors"}

    assert offline_marlin._layer_checkpoint_prefix(
        "layers.0.ffn.experts.routed_experts", index
    ) == "model.layers.0.ffn.experts.routed_experts"
