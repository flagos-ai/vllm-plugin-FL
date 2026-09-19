# Copyright (c) 2026 BAAI. All rights reserved.
"""Install HY4 support into a pristine vLLM 0.24 runtime.

All changes stay inside vllm-plugin-FL. The hook registers the checkpoint
config, compressed-MLA architecture conversion, lazy model implementation,
and expert-sliced safetensors loader without modifying the vLLM installation.
"""

from __future__ import annotations

import logging
from functools import wraps
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any

logger = logging.getLogger(__name__)

_ARCHITECTURE = "HYV4ForCausalLM"
_LOAD_FORMAT = "hy4_safetensors"


def is_vllm_024() -> bool:
    """Return whether the active vLLM belongs to the 0.24 ABI line.

    HY4 is intentionally implemented against vLLM 0.24.  Keep this probe
    local to the model adapter so the model commit does not import the
    unrelated compatibility module from another branch.
    """
    try:
        release = package_version("vllm")
    except PackageNotFoundError:
        try:
            import vllm

            release = getattr(vllm, "__version__", "")
        except Exception:
            return False
    parts = release.split("+", 1)[0].split(".")
    return len(parts) >= 2 and parts[:2] == ["0", "24"]


def _patch_mxfp8_override_order(me_quant: Any) -> None:
    """Make vLLM 0.24 probe the canonical ModelOpt MXFP8 entry first.

    vLLM 0.24 maps both ``modelopt_mxfp8`` and the online shorthand
    ``mxfp8`` to ``ModelOptMxFp8Config``, but only the former is present in
    ``ModelConfig._verify_quantization``'s ordered override list.  Therefore
    the shorthand reports an override before the canonical entry is reached
    and ModelConfig rejects a serialized MXFP8 checkpoint.  Returning a
    no-override view for the shorthand is equivalent to placing ``mxfp8``
    after ``modelopt_mxfp8`` in that list, without replacing ModelConfig or
    modifying the vLLM installation.
    """
    current_getter = me_quant.get_quantization_config
    if getattr(current_getter, "_hy4_v024_mxfp8_order", False):
        return

    alias = None

    def make_alias():
        class MXFP8AliasAfterCanonical(current_getter("mxfp8")):
            @classmethod
            def override_quantization_method(
                cls,
                hf_quant_cfg: dict[str, Any],
                user_quant: str | None,
                hf_config: Any = None,
            ) -> None:
                if getattr(hf_config, "model_type", None) == "hy_v4":
                    return None
                return current_getter("mxfp8").override_quantization_method(
                    hf_quant_cfg, user_quant, hf_config=hf_config
                )

        return MXFP8AliasAfterCanonical

    @wraps(current_getter)
    def get_quantization_config(name: str):
        nonlocal alias
        if name == "mxfp8":
            if alias is None:
                alias = make_alias()
            return alias
        return current_getter(name)

    get_quantization_config._hy4_v024_mxfp8_order = True
    me_quant.get_quantization_config = get_quantization_config


def apply_hy_v4_v024_patches() -> bool:
    """Register the HY4 runtime components required by vLLM 0.24.x."""
    if not is_vllm_024():
        return False

    from vllm_fl.configs.hy_v4 import HYV4Config
    from vllm_fl.configs.hy_v4_convertor import HYV4ModelArchConfigConvertor
    from vllm_fl.model_loader.hy_v4_loader import HYV4SafetensorsLoader

    from vllm.model_executor import model_loader
    from vllm.model_executor.models import registry as model_registry
    from vllm.transformers_utils import (
        config as transformers_config,
        model_arch_config_convertor,
    )

    transformers_config._CONFIG_REGISTRY.setdefault("hy_v4", HYV4Config)
    model_arch_config_convertor.MODEL_ARCH_CONFIG_CONVERTORS["hy_v4"] = (
        HYV4ModelArchConfigConvertor
    )
    model_registry.ModelRegistry.register_model(
        _ARCHITECTURE,
        "vllm_fl.models.hy_v4:HYV4ForCausalLM",
    )

    registered_loaders = model_loader._LOAD_FORMAT_TO_MODEL_LOADER
    if registered_loaders.get(_LOAD_FORMAT) is not HYV4SafetensorsLoader:
        model_loader.register_model_loader(_LOAD_FORMAT)(HYV4SafetensorsLoader)

    logger.info("Installed HY4 runtime compatibility for vLLM 0.24")
    return True


__all__ = [
    "apply_hy_v4_v024_patches",
]


def __getattr__(name):
    if name == "HYV4ModelArchConfigConvertor":
        from vllm_fl.configs.hy_v4_convertor import HYV4ModelArchConfigConvertor

        return HYV4ModelArchConfigConvertor
    if name == "HYV4Config":
        from vllm_fl.configs.hy_v4 import HYV4Config

        return HYV4Config
    if name == "HYV4SafetensorsLoader":
        from vllm_fl.model_loader.hy_v4_loader import HYV4SafetensorsLoader

        return HYV4SafetensorsLoader
    raise AttributeError(name)
