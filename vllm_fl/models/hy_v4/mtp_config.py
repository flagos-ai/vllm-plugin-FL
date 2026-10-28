# SPDX-License-Identifier: Apache-2.0
"""Narrow HYV4 draft metadata integration; no runtime imports at module load."""

from __future__ import annotations

import copy

_ORIGINAL_HF_CONFIG_OVERRIDE = None
_ORIGINAL_SPECULATIVE_POST_INIT = None


def hyv4_mtp_hf_config_override(hf_config):
    """Draft-only callback; every non-HYV4 model delegates to native unchanged.

    This module-level callable is spawn-picklable. Do not replace it with a
    local closure: native ModelConfig stores this callback in its draft config.
    """
    if getattr(hf_config, "model_type", None) not in ("hy_v4", "hy_v4_vl"):
        original = _ORIGINAL_HF_CONFIG_OVERRIDE
        if original is None:
            from vllm.config.speculative import SpeculativeConfig

            original = SpeculativeConfig.hf_config_override
            if getattr(original, "_hyv4_mtp_native_override", None) is not None:
                original = original._hyv4_mtp_native_override
        return original(hf_config)

    draft = copy.deepcopy(hf_config)
    outer_quant = getattr(draft, "quantization_config", None)
    text = getattr(draft, "text_config", draft)
    if outer_quant is not None and getattr(text, "quantization_config", None) is None:
        text.quantization_config = outer_quant
    count = getattr(text, "num_nextn_predict_layers", 0)
    if type(count) is not int or count < 1:
        raise ValueError("HYV4 checkpoint must declare at least one MTP layer")
    text.update({"model_type": "mtp", "architectures": ["HYV4MTP"], "n_predict": count})
    return text


def install_hyv4_mtp_config_override(speculative_config_cls=None):
    """Install only a selective callback; keep native Literal/config library intact."""
    global _ORIGINAL_HF_CONFIG_OVERRIDE
    if speculative_config_cls is None:
        from vllm.config.speculative import SpeculativeConfig

        speculative_config_cls = SpeculativeConfig
    current = speculative_config_cls.hf_config_override
    original = getattr(current, "_hyv4_mtp_native_override", current)
    _ORIGINAL_HF_CONFIG_OVERRIDE = original
    hyv4_mtp_hf_config_override._hyv4_mtp_native_override = original
    speculative_config_cls.hf_config_override = staticmethod(
        hyv4_mtp_hf_config_override
    )


def hyv4_mtp_speculative_post_init(self):
    """Align HYV4 EP draft validation with the target parallel config used to load.

    Native construction runs first. Pydantic's after-validators must then run
    normally on the inherited config; no native validation is bypassed here.
    This hook creates no distributed group and changes no global TP/EP group.
    """
    result = _ORIGINAL_SPECULATIVE_POST_INIT(self)
    draft = getattr(self, "draft_model_config", None)
    target = getattr(self, "target_model_config", None)
    draft_hf = getattr(draft, "hf_config", None)
    target_hf = getattr(target, "hf_config", None)
    parallel = getattr(self, "target_parallel_config", None)
    if not (
        self.method == "mtp"
        and getattr(draft_hf, "architectures", None) == ["HYV4MTP"]
        and getattr(target_hf, "model_type", None) in ("hy_v4", "hy_v4_vl")
        and getattr(parallel, "enable_expert_parallel", False)
    ):
        return result
    if self.draft_tensor_parallel_size != parallel.tensor_parallel_size:
        raise ValueError(
            "HYV4 MTP with shared target expert parallel groups requires "
            "draft_tensor_parallel_size == target tensor_parallel_size"
        )
    # Preserve every validated field, including nnodes/node_rank/world size and
    # nested config references. Do not reconstruct ParallelConfig or TP/EP groups.
    self.draft_parallel_config = copy.copy(parallel)
    return result


def install_hyv4_mtp_parallel_config(speculative_config_cls=None):
    """Wrap instance post-init only; leave the all-model static factory intact."""
    global _ORIGINAL_SPECULATIVE_POST_INIT
    if speculative_config_cls is None:
        from vllm.config.speculative import SpeculativeConfig

        speculative_config_cls = SpeculativeConfig
    current = speculative_config_cls.__post_init__
    original = getattr(current, "_hyv4_mtp_native_post_init", current)
    _ORIGINAL_SPECULATIVE_POST_INIT = original
    hyv4_mtp_speculative_post_init._hyv4_mtp_native_post_init = original
    speculative_config_cls.__post_init__ = hyv4_mtp_speculative_post_init


def remap_compressed_mtp_quant_config(
    quant_config, mtp_start_layer_idx, num_mtp_layers
):
    """Copy CT scheme targets/ignore; append only MTP draft-prefix aliases.

    Native CT matching uses ``target_scheme_map`` and ``ignore``. Mapping only
    FP8's ignored_layers leaves explicit W8A8 eh_proj/attention targets unmatched.
    Keep originals, non-MTP metadata and the target's shared quant object intact.
    """

    def alias(pattern):
        for offset in range(num_mtp_layers):
            layer = mtp_start_layer_idx + offset
            if pattern.startswith("re:"):
                old = rf"mtp_layers\.{offset}\."
                new = rf"layers\.{layer}\."
                if old in pattern:
                    return pattern.replace(old, new)
            else:
                marker = f"model.mtp_layers.{offset}."
                at = pattern.find(marker)
                if at >= 0:
                    return f"model.layers.{layer}." + pattern[at + len(marker) :]
        return None

    result = copy.copy(quant_config)
    result.target_scheme_map = copy.deepcopy(quant_config.target_scheme_map)
    result.ignore = list(quant_config.ignore)
    for target, scheme in list(result.target_scheme_map.items()):
        mapped = alias(target)
        if mapped is not None and mapped not in result.target_scheme_map:
            result.target_scheme_map[mapped] = scheme
    for pattern in list(result.ignore):
        mapped = alias(pattern)
        if mapped is not None and mapped not in result.ignore:
            result.ignore.append(mapped)
    return result
