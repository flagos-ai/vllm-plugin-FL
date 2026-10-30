# SPDX-License-Identifier: Apache-2.0

from transformers import DeepseekV2Config

from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig


def test_checkpoint_rope_parameters_are_preserved():
    parameters = {"rope_type": "default", "rope_theta": 8_000_000.0}
    config = GlmMoeDsaConfig(rope_parameters=parameters)

    assert config.rope_parameters["rope_theta"] == 8_000_000.0
    assert config.rope_parameters["rope_type"] == "default"
    assert config.rope_theta == 8_000_000.0

    restored = GlmMoeDsaConfig.from_dict(config.to_dict())
    assert restored.rope_parameters == config.rope_parameters


def test_omitted_rope_parameters_keep_parent_defaults():
    parent = DeepseekV2Config()
    config = GlmMoeDsaConfig()

    assert getattr(config, "rope_parameters", None) == getattr(
        parent, "rope_parameters", None
    )
