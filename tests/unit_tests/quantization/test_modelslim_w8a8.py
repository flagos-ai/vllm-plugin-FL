from vllm_fl.quantization.modelslim_w8a8 import ModelSlimW8A8Config


def _config():
    return ModelSlimW8A8Config(
        {
            "layers.0.attn.wq_a.weight": "W8A8_DYNAMIC",
            "layers.0.attn.wkv.weight": "W8A8_DYNAMIC",
            "layers.0.attn.wq_b.weight": "W8A8_DYNAMIC",
            "layers.0.attn.wo_a.weight": "FLOAT",
            "layers.0.ffn.experts.0.w1.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.w2.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.w3.weight": "W8A8_DYNAMIC",
        }
    )


def test_maps_deepseek_v4_packed_attention_names():
    config = _config()
    assert config._is_dynamic_w8a8("model.layers.0.attn.fused_wqa_wkv")
    assert config._is_dynamic_w8a8("model.layers.0.attn.wq_b")
    assert not config._is_dynamic_w8a8("model.layers.0.attn.wo_a")


def test_maps_deepseek_v4_routed_experts():
    assert _config()._is_dynamic_w8a8("model.layers.0.ffn.experts")


def test_rejects_mixed_packed_quantization():
    config = _config()
    config.description["layers.0.attn.wkv.weight"] = "FLOAT"
    try:
        config._is_dynamic_w8a8("model.layers.0.attn.fused_wqa_wkv")
    except ValueError as error:
        assert "mixes" in str(error)
    else:
        raise AssertionError("mixed packed quantization must fail")
