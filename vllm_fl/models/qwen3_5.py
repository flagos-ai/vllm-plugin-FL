# Copyright (c) 2026 BAAI. All rights reserved.
"""Runtime compatibility shim for Qwen3.5 text-only causal models.

The model implementation remains the one shipped by vLLM. Importing this
module adds the hybrid-model metadata, cache helpers, and VL checkpoint prefix
mapping that the upstream text-only classes are missing, then re-exports those
classes for lazy registration by the FL plugin.
"""

from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models import qwen3_5 as _upstream
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper


def _get_mamba_state_dtype_from_config(cls, vllm_config):
    return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        vllm_config.model_config.dtype,
        vllm_config.cache_config.mamba_cache_dtype,
        vllm_config.cache_config.mamba_ssm_cache_dtype,
    )


def _get_mamba_state_shape_from_config(cls, vllm_config):
    parallel_config = vllm_config.parallel_config
    hf_config = vllm_config.model_config.hf_text_config
    num_speculative_tokens = (
        vllm_config.speculative_config.num_speculative_tokens
        if vllm_config.speculative_config
        else 0
    )
    return MambaStateShapeCalculator.gated_delta_net_state_shape(
        parallel_config.tensor_parallel_size,
        hf_config.linear_num_key_heads,
        hf_config.linear_num_value_heads,
        hf_config.linear_key_head_dim,
        hf_config.linear_value_head_dim,
        hf_config.linear_conv_kernel_dim,
        num_speculative_tokens,
    )


def _get_mamba_state_copy_func(cls):
    return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


# Some Qwen3.5 text-only checkpoints retain a VL architecture and store language
# weights under ``model.language_model.*``. Unlike the VL classes there is no
# ``language_model`` submodule wrapper on the text-only classes, so the target
# prefix is ``model.``.
_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_prefix={"model.language_model.": "model."},
)


def _load_weights(self, weights):
    """Load text-only or VL-prefixed Qwen3.5 checkpoints."""
    loader = AutoWeightsLoader(
        self,
        skip_prefixes=["mtp."],
        ignore_unexpected_prefixes=["model.visual."],
    )
    return loader.load_weights(weights, mapper=_WEIGHTS_MAPPER)


def _patch_upstream_base() -> None:
    base = _upstream.Qwen3_5ForCausalLMBase

    # vLLM's is_hybrid() is intentionally attribute based, so adding this
    # marker at runtime is equivalent to inheriting the IsHybrid protocol.
    if not getattr(base, "is_hybrid", False):
        base.is_hybrid = True

    helpers = {
        "get_mamba_state_dtype_from_config": _get_mamba_state_dtype_from_config,
        "get_mamba_state_shape_from_config": _get_mamba_state_shape_from_config,
        "get_mamba_state_copy_func": _get_mamba_state_copy_func,
    }
    for name, helper in helpers.items():
        if not hasattr(base, name):
            setattr(base, name, classmethod(helper))

    # Remap quantization metadata together with checkpoint tensor names.
    base.load_weights = _load_weights

    # Rebinding load_weights alone only covers weight names.  configure_quant_config
    # (model_executor/model_loader/utils.py) reads this class attribute to call
    # quant_config.apply_vllm_mapper(), which rewrites Fp8Config.ignored_layers.
    # Without it, modules that must remain in high precision no longer match
    # their configured names. The failure is silent because weight loading can
    # still complete successfully. Guard the assignment so this composes with
    # an equivalent vLLM-side source patch.
    if getattr(base, "hf_to_vllm_mapper", None) is None:
        base.hf_to_vllm_mapper = _WEIGHTS_MAPPER


_patch_upstream_base()

# Keep the exact upstream class identities. The plugin's model registry points
# here so the compatibility behavior is applied lazily before inspection.
Qwen3_5ForCausalLM = _upstream.Qwen3_5ForCausalLM
Qwen3_5MoeForCausalLM = _upstream.Qwen3_5MoeForCausalLM

__all__ = ["Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM"]
