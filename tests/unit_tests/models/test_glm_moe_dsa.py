from types import SimpleNamespace

import pytest

from vllm.model_executor.models.deepseek_v2 import (
    GlmMoeDsaForCausalLM as VllmGlmMoeDsaForCausalLM,
)

from vllm_fl.models import glm_moe_dsa
from vllm_fl.patches import ascend_glm_dsa


def test_dense_fallback_updates_only_glm_arch_for_short_context():
    hf_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        index_topk=128,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        num_attention_heads=96,
    )
    model_arch_config = SimpleNamespace(
        is_deepseek_mla=True,
        head_size=192,
        total_num_kv_heads=1,
    )

    class ModelConfig:
        def __init__(self):
            self.hf_text_config = hf_config
            self.max_model_len = 128
            self.model_arch_config = model_arch_config

        @property
        def use_mla(self):
            return self.model_arch_config.is_deepseek_mla

    model_config = ModelConfig()

    assert ascend_glm_dsa.prepare_glm_dsa_dense_fallback(model_config) is True
    assert model_arch_config.is_deepseek_mla is False
    assert model_arch_config.head_size == 256
    assert model_arch_config.total_num_kv_heads == 96

    model_arch_config.is_deepseek_mla = True
    model_arch_config.head_size = 192
    model_arch_config.total_num_kv_heads = 1
    model_config.max_model_len = 129
    with pytest.raises(ValueError, match="max-model-len <= index_topk"):
        ascend_glm_dsa.prepare_glm_dsa_dense_fallback(model_config)
    assert model_arch_config.is_deepseek_mla is True
    assert model_arch_config.head_size == 192
    assert model_arch_config.total_num_kv_heads == 1


def test_dense_fallback_leaves_non_glm_models_unchanged():
    model_arch_config = SimpleNamespace(
        is_deepseek_mla=True,
        head_size=256,
        total_num_kv_heads=128,
    )
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="deepseek_v3", index_topk=64),
        max_model_len=4096,
        model_arch_config=model_arch_config,
    )

    assert ascend_glm_dsa.prepare_glm_dsa_dense_fallback(model_config) is False
    assert model_arch_config.is_deepseek_mla is True
    assert model_arch_config.head_size == 256
    assert model_arch_config.total_num_kv_heads == 128


def test_dense_fallback_restores_arch_when_validation_raises():
    hf_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        index_topk=128,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        num_attention_heads=96,
    )
    model_arch_config = SimpleNamespace(
        is_deepseek_mla=True,
        head_size=192,
        total_num_kv_heads=1,
    )

    class ModelConfig:
        def __init__(self):
            self.hf_text_config = hf_config
            self.max_model_len = 128
            self.model_arch_config = model_arch_config

        @property
        def use_mla(self):
            raise RuntimeError("validation failed")

    with pytest.raises(RuntimeError, match="validation failed"):
        ascend_glm_dsa.prepare_glm_dsa_dense_fallback(ModelConfig())
    assert model_arch_config.is_deepseek_mla is True
    assert model_arch_config.head_size == 192
    assert model_arch_config.total_num_kv_heads == 1


def test_dense_hf_config_copy_hides_index_topk_without_mutating_original():
    class BaseConfig:
        index_topk = 1024

    class GlmConfig(BaseConfig):
        pass

    hf_config = GlmConfig()
    hf_config.index_topk = 256
    hf_config.hidden_size = 6144
    model_config = SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config)

    with ascend_glm_dsa.use_glm_dsa_dense_hf_config(model_config) as dense_config:
        assert dense_config is model_config.hf_config
        assert dense_config is model_config.hf_text_config
        assert dense_config is not hf_config
        assert not hasattr(dense_config, "index_topk")
        assert dense_config.hidden_size == 6144
        assert hf_config.index_topk == 256
        assert BaseConfig.index_topk == 1024

    assert model_config.hf_config is hf_config
    assert model_config.hf_text_config is hf_config
    assert hf_config.index_topk == 256
    assert BaseConfig.index_topk == 1024


def test_dense_hf_config_restores_original_when_model_construction_fails():
    class GlmConfig:
        index_topk = 1024

    hf_config = GlmConfig()
    hf_config.index_topk = 256
    model_config = SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config)

    construction_failed = False
    try:
        with ascend_glm_dsa.use_glm_dsa_dense_hf_config(model_config):
            assert not hasattr(model_config.hf_config, "index_topk")
            assert hf_config.index_topk == 256
            raise RuntimeError("construction failed")
    except RuntimeError as error:
        assert str(error) == "construction failed"
        construction_failed = True

    assert construction_failed
    assert model_config.hf_config is hf_config
    assert model_config.hf_text_config is hf_config
    assert hf_config.index_topk == 256
    assert GlmConfig.index_topk == 1024


def test_glm_wrapper_hides_index_topk_only_while_upstream_builds(monkeypatch):
    class GlmConfig:
        index_topk = 1024

    hf_config = GlmConfig()
    hf_config.index_topk = 256
    model_config = SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config)
    vllm_config = SimpleNamespace(model_config=model_config)
    events = []

    monkeypatch.setattr(
        glm_moe_dsa,
        "prepare_glm_dsa_dense_fallback",
        lambda config: events.append(("prepare", config)),
    )

    def fake_upstream_init(self, *, vllm_config, prefix):
        del self
        events.append(("upstream", vllm_config, prefix))
        assert not hasattr(vllm_config.model_config.hf_config, "index_topk")
        assert vllm_config.model_config.hf_config is not hf_config
        assert hf_config.index_topk == 256

    monkeypatch.setattr(VllmGlmMoeDsaForCausalLM, "__init__", fake_upstream_init)
    model = object.__new__(glm_moe_dsa.GlmMoeDsaForCausalLM)

    glm_moe_dsa.GlmMoeDsaForCausalLM.__init__(
        model,
        vllm_config=vllm_config,
        prefix="model",
    )

    assert events == [
        ("prepare", model_config),
        ("upstream", vllm_config, "model"),
    ]
    assert model_config.hf_config is hf_config
    assert model_config.hf_text_config is hf_config
    assert hf_config.index_topk == 256
    assert GlmConfig.index_topk == 1024


def test_glm_wrapper_skips_only_top_level_modelslim_rotation_weight(monkeypatch):
    rotation = object()
    embedding = object()
    nested_rotation = object()
    similarly_named = object()
    unknown_rotation_parameter = object()
    captured_weights = []

    def fake_upstream_load_weights(self, weights):
        del self
        captured_weights.extend(weights)
        return {"model.embed_tokens.weight"}

    monkeypatch.setattr(
        VllmGlmMoeDsaForCausalLM,
        "load_weights",
        fake_upstream_load_weights,
    )
    model = object.__new__(glm_moe_dsa.GlmMoeDsaForCausalLM)

    loaded = model.load_weights(
        iter(
            [
                ("rot.weight", rotation),
                ("model.embed_tokens.weight", embedding),
                ("model.layers.78.rot.weight", nested_rotation),
                ("my_rot.weight", similarly_named),
                ("rot.bias", unknown_rotation_parameter),
            ]
        )
    )

    assert loaded == {"model.embed_tokens.weight"}
    assert captured_weights == [
        ("model.embed_tokens.weight", embedding),
        ("model.layers.78.rot.weight", nested_rotation),
        ("my_rot.weight", similarly_named),
        ("rot.bias", unknown_rotation_parameter),
    ]
