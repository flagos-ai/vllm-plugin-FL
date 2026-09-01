# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 model entry point with MetaX-specific adaptation."""

from vllm.model_executor.models.deepseek_v2 import (
    GlmMoeDsaForCausalLM as VllmGlmMoeDsaForCausalLM,
)
from vllm.platforms import current_platform

if current_platform.vendor_name == "metax":
    from vllm_fl.patches.glm_moe_dsa_metax import apply_model_patches

    apply_model_patches()


class GlmMoeDsaForCausalLM(VllmGlmMoeDsaForCausalLM):
    pass
