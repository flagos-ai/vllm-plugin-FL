# SPDX-License-Identifier: Apache-2.0
"""Register Hy4-preview configuration and models with the native vLLM host."""


def register_hy_v4():
    from transformers import AutoConfig
    from vllm.model_executor.models import ModelRegistry

    from .config import HYV4Config, HYV4VLConfig

    AutoConfig.register("hy_v4", HYV4Config, exist_ok=True)
    AutoConfig.register("hy_v4_vl", HYV4VLConfig, exist_ok=True)
    ModelRegistry.register_model(
        "HYV4ForCausalLM", "vllm_fl.models.hy_v4.nvidia.model:HYV4ForCausalLM"
    )
    ModelRegistry.register_model(
        "HYV4VLForConditionalGeneration",
        "vllm_fl.models.hy_v4.nvidia.text_compat:HYV4VLForConditionalGeneration",
    )
    ModelRegistry.register_model("HYV4MTP", "vllm_fl.models.hy_v4.nvidia.mtp:HYV4MTP")

    from .mtp_config import (
        install_hyv4_mtp_config_override,
        install_hyv4_mtp_parallel_config,
    )

    install_hyv4_mtp_config_override()
    install_hyv4_mtp_parallel_config()
