"""Ascend compatibility wrapper for GLM-5/5.2 DSA models."""

from __future__ import annotations

from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_v2 import (
    GlmMoeDsaForCausalLM as VllmGlmMoeDsaForCausalLM,
)

from vllm_fl.patches.ascend_glm_dsa import (
    prepare_glm_dsa_dense_fallback,
    use_glm_dsa_dense_hf_config,
)


class GlmMoeDsaForCausalLM(VllmGlmMoeDsaForCausalLM):
    """Use dense attention for short GLM DSA requests on Ascend."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        model_config = vllm_config.model_config
        prepare_glm_dsa_dense_fallback(model_config)
        with use_glm_dsa_dense_hf_config(model_config):
            super().__init__(vllm_config=vllm_config, prefix=prefix)


__all__ = ["GlmMoeDsaForCausalLM"]
