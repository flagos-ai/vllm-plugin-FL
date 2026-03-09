# SPDX-License-Identifier: Apache-2.0
"""Inference-only GLM-5 (GlmMoeDsa) model.

GLM-5 uses a DeepSeek V2/V3-style architecture with MLA (Multi-head Latent
Attention) and Mixture of Experts.  The HF model type is ``glm_moe_dsa`` and
the architecture class is ``GlmMoeDsaForCausalLM``.

This thin wrapper inherits from vLLM's ``DeepseekV2ForCausalLM`` which already
handles MLA, MoE, the DSA Indexer, and MTP speculative decoding layers.
"""

from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM


class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    """GLM-5 model for causal language modelling."""
    pass
