# SPDX-License-Identifier: Apache-2.0
from functools import wraps
from typing import Any

from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)


class HYV4ModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Expose HY4's compressed MLA and ModelOpt MXFP8 metadata to vLLM."""

    def get_head_size(self) -> int:
        return int(self.hf_text_config.kv_lora_rank) + int(
            self.hf_text_config.qk_rope_head_dim
        )

    def is_deepseek_mla(self) -> bool:
        return True

    def get_quantization_config(self) -> dict[str, Any] | None:
        from vllm.model_executor.layers import quantization as me_quant

        quant_config = super().get_quantization_config()
        if quant_config is not None:
            _patch_mxfp8_override_order(me_quant)
        if quant_config is None or quant_config.get("quant_method") != "mxfp8":
            return quant_config

        # The paired vLLM runtime's ModelOpt MXFP8 parser already understands the raw
        # MiniMax-style schema during weight construction, but its earlier
        # override-selection pass only recognizes a ModelOpt-shaped config.
        # Return a normalized copy for architecture detection; keep the HF
        # config untouched so ModelOptMxFp8Config.from_config handles it later.
        return {
            "quant_method": "modelopt",
            "quantization": {
                "quant_algo": "MXFP8",
                "kv_cache_quant_algo": quant_config.get("kv_cache_quant_algo"),
                "exclude_modules": quant_config.get("ignored_layers", []) or [],
            },
        }


def _patch_mxfp8_override_order(me_quant: Any) -> None:
    """Make vLLM probe the canonical ModelOpt MXFP8 entry first.

    The paired vLLM runtime maps both ``modelopt_mxfp8`` and the online shorthand
    ``mxfp8`` to ``ModelOptMxFp8Config``, but only the former is present in
    ``ModelConfig._verify_quantization``'s ordered override list.  Therefore
    the shorthand reports an override before the canonical entry is reached
    and ModelConfig rejects a serialized MXFP8 checkpoint.  Returning a
    no-override view for the shorthand is equivalent to placing ``mxfp8``
    after ``modelopt_mxfp8`` in that list, without replacing ModelConfig or
    modifying the vLLM installation.
    """
    current_getter = me_quant.get_quantization_config
    if getattr(current_getter, "_hy4_mxfp8_override_order", False):
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

    get_quantization_config._hy4_mxfp8_override_order = True
    me_quant.get_quantization_config = get_quantization_config
