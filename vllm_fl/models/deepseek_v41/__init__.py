# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX-only model registration; the root config and native encoders stay standard."""


def register_deepseek_v41_model():
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekV41ForCausalLM",
        "vllm_fl.models.deepseek_v41.vl_model:DeepseekV41ForCausalLM",
    )
    ModelRegistry.register_model(
        "DSparkV41DraftModel",
        "vllm_fl.models.deepseek_v41.dspark:DSparkDeepseekV41ForCausalLM",
    )
