from vllm_fl.patches import offline_marlin


def test_offline_marlin_env(monkeypatch):
    monkeypatch.delenv("VLLM_FL_OFFLINE_MARLIN", raising=False)
    assert offline_marlin.offline_marlin_enabled() is False
    monkeypatch.setenv("VLLM_FL_OFFLINE_MARLIN", "yes")
    assert offline_marlin.offline_marlin_enabled() is True


def test_checkpoint_key_mapping_only_changes_routed_marlin_aliases():
    prefix = "model.layers.0.ffn.experts.routed_experts"
    assert offline_marlin._checkpoint_key_to_parameter(
        f"{prefix}.w13_weight"
    ) == f"{prefix}.w13_weight_packed"
    assert offline_marlin._checkpoint_key_to_parameter(
        f"{prefix}.w2_weight"
    ) == f"{prefix}.w2_weight_packed"
    assert (
        offline_marlin._checkpoint_key_to_parameter(
            "model.layers.0.ffn.shared_experts.w2_weight"
        )
        == "model.layers.0.ffn.shared_experts.w2_weight"
    )
