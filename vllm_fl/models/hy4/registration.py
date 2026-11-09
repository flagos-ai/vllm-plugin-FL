# Copyright (c) 2026 BAAI. All rights reserved.
"""Register HY4 components with the paired vLLM runtime.

All changes stay inside vllm-plugin-FL. The hook registers the checkpoint
config, compressed-MLA architecture conversion, lazy model implementation,
and expert-sliced safetensors loader without modifying the vLLM installation.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ARCHITECTURE = "HYV4ForCausalLM"
_LOAD_FORMAT = "hy4_safetensors"


def register_hy_v4_support() -> bool:
    """Register the HY4 config, architecture, model and checkpoint loader."""
    from vllm.model_executor import model_loader
    from vllm.model_executor.models import registry as model_registry
    from vllm.transformers_utils import (
        config as transformers_config,
        model_arch_config_convertor,
    )

    from vllm.platforms import current_platform

    is_ppu = getattr(current_platform, "vendor_name", None) == "thead"
    model_path = "vllm_fl.models.hy4.model:HYV4ForCausalLM"
    if is_ppu:
        from vllm_fl.configs.hy_v4 import HYV4Config
        from vllm_fl.model_loader.hy_v4_loader import HYV4SafetensorsLoader

        class HYV4ModelArchConfigConvertor(
            model_arch_config_convertor.ModelArchConfigConvertorBase
        ):
            def get_head_size(self) -> int:
                return int(self.hf_text_config.kv_lora_rank) + int(
                    self.hf_text_config.qk_rope_head_dim
                )

            def is_deepseek_mla(self) -> bool:
                return True

        model_path = "vllm_fl.models.hy_v4:HYV4ForCausalLM"
        transformers_config._CONFIG_REGISTRY["hy_v4"] = HYV4Config
    else:
        from .config import HYV4Config
        from .config_convertor import HYV4ModelArchConfigConvertor
        from .loader import HYV4SafetensorsLoader

        transformers_config._CONFIG_REGISTRY.setdefault("hy_v4", HYV4Config)
    model_arch_config_convertor.MODEL_ARCH_CONFIG_CONVERTORS["hy_v4"] = (
        HYV4ModelArchConfigConvertor
    )
    model_registry.ModelRegistry.register_model(
        _ARCHITECTURE,
        model_path,
    )

    registered_loaders = model_loader._LOAD_FORMAT_TO_MODEL_LOADER
    if registered_loaders.get(_LOAD_FORMAT) is not HYV4SafetensorsLoader:
        model_loader.register_model_loader(_LOAD_FORMAT)(HYV4SafetensorsLoader)

    # Register the public class path before an explicit CUSTOM prefill selection.
    from vllm_fl.attention.mla_prefill import register_flaggems_mla_prefill

    if not is_ppu:
        register_flaggems_mla_prefill()

    logger.info("Registered HY4 runtime components")
    return True


__all__ = ["register_hy_v4_support"]
