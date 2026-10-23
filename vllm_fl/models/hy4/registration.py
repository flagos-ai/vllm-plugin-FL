# Copyright (c) 2026 BAAI. All rights reserved.
"""Register HY4 components with the paired vLLM runtime.

All changes stay inside vllm-plugin-FL. The hook registers the checkpoint
config, compressed-MLA architecture conversion, lazy model implementation,
and expert-sliced safetensors loader without modifying the vLLM installation.

Both the text model (``hy_v4`` / ``HYV4ForCausalLM``) and the vision-language
composition (``hy_v4_vl`` / ``HYV4VLForConditionalGeneration``) are registered;
the VL pieces mirror the upstream reference implementation so that a FlagOS
host resolves the same graph as an NVIDIA host.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_ARCHITECTURE = "HYV4ForCausalLM"
_VL_ARCHITECTURE = "HYV4VLForConditionalGeneration"
_LOAD_FORMAT = "hy4_safetensors"

# ``HFV4VLProcessor`` is named by the checkpoint's ``preprocessor_config.json``;
# the paired vLLM resolves processor classes through this lazy map, so pointing
# the entry at our module is enough to make ``AutoProcessor`` find it.
_VL_PROCESSOR_CLASSES = (
    "HYV4VLProcessor",
    "HYV4VLImageProcessor",
    "HYV4VLVideoProcessor",
)


def _register_vl_processors() -> None:
    """Teach vLLM's lazy processor registry about the HY4 VL processors."""
    from vllm.transformers_utils import processors as vllm_processors

    class_to_module = getattr(vllm_processors, "_CLASS_TO_MODULE", None)
    if class_to_module is None:  # pragma: no cover - unexpected runtime layout
        logger.warning("vLLM processor registry not found; HY4 VL processors unregistered")
        return
    for name in _VL_PROCESSOR_CLASSES:
        class_to_module[name] = "vllm_fl.models.hy4.mm_processor"
    exported = getattr(vllm_processors, "__all__", None)
    if isinstance(exported, list):
        for name in _VL_PROCESSOR_CLASSES:
            if name not in exported:
                exported.append(name)


def register_hy_v4_support() -> bool:
    """Register the HY4 config, architecture, model and checkpoint loader."""
    from vllm.model_executor import model_loader
    from vllm.model_executor.models import registry as model_registry
    from vllm.transformers_utils import (
        config as transformers_config,
        model_arch_config_convertor,
    )

    from .config import HYV4Config
    from .config_convertor import HYV4ModelArchConfigConvertor
    from .loader import HYV4SafetensorsLoader

    transformers_config._CONFIG_REGISTRY.setdefault("hy_v4", HYV4Config)
    model_arch_config_convertor.MODEL_ARCH_CONFIG_CONVERTORS["hy_v4"] = (
        HYV4ModelArchConfigConvertor
    )
    model_registry.ModelRegistry.register_model(
        _ARCHITECTURE,
        "vllm_fl.models.hy4.model:HYV4ForCausalLM",
    )

    # ---- vision-language composition (hy_v4_vl) ----
    from .vl_config import HYV4VLConfig

    transformers_config._CONFIG_REGISTRY["hy_v4_vl"] = HYV4VLConfig
    # ``hf_text_config`` resolves to the nested HYV4Config; reuse the same
    # convertor so MLA head size / quantization metadata behave as in text mode.
    model_arch_config_convertor.MODEL_ARCH_CONFIG_CONVERTORS["hy_v4_vl"] = (
        HYV4ModelArchConfigConvertor
    )
    model_registry.ModelRegistry.register_model(
        _VL_ARCHITECTURE,
        "vllm_fl.models.hy4.vl_model:HYV4VLForConditionalGeneration",
    )
    _register_vl_processors()

    registered_loaders = model_loader._LOAD_FORMAT_TO_MODEL_LOADER
    if registered_loaders.get(_LOAD_FORMAT) is not HYV4SafetensorsLoader:
        model_loader.register_model_loader(_LOAD_FORMAT)(HYV4SafetensorsLoader)

    # Register the public class path before an explicit CUSTOM prefill selection.
    from vllm_fl.attention.mla_prefill import register_flaggems_mla_prefill

    register_flaggems_mla_prefill()

    logger.info("Registered HY4 runtime components (text + VL)")
    return True


__all__ = ["register_hy_v4_support"]
