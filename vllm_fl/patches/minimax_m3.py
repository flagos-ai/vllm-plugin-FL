# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 metadata and a direct bridge for its missing native operator."""

from __future__ import annotations

import logging
from copy import copy
from functools import wraps
from types import MethodType

import torch

logger = logging.getLogger(__name__)

_OP_NAME = "fused_minimax_m3_qknorm_rope_kv_insert"


def patch_minimax_m3_fused_preprocess() -> bool:
    """Patch the Python call site only when vLLM's native op is absent.

    MiniMax-M3's NVIDIA and AMD modules both retain a reference to the
    :mod:`vllm._custom_ops` module. Replacing this one model-specific wrapper
    therefore reaches both callers without editing vLLM source or registering
    a synthetic operator in vLLM's private ``torch.ops._C`` namespace.
    """
    from vllm import _custom_ops as ops

    current = getattr(ops, _OP_NAME, None)
    if current is None:
        logger.debug("vLLM has no Python wrapper for %s", _OP_NAME)
        return False
    if getattr(current, "_vllm_fl_minimax_m3_patch", False):
        return False
    if hasattr(torch.ops._C, _OP_NAME):
        logger.debug("vLLM native %s is available", _OP_NAME)
        return False

    @wraps(current)
    def fused_preprocess(*args, **kwargs):
        from flaggems_vllm import fused_minimax_m3_qknorm_rope_kv_insert

        return fused_minimax_m3_qknorm_rope_kv_insert(*args, **kwargs)

    fused_preprocess._vllm_fl_minimax_m3_patch = True
    setattr(ops, _OP_NAME, fused_preprocess)
    logger.info("Patched missing vLLM %s with the FlagGems-vllm API", _OP_NAME)
    return True


PACKED_MODULES = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def install_minimax_m3_vision_mapping(config) -> bool:
    """Scope the three-way vision QKV map to one model's quant config.

    The text model uses five packed projections. Bind the compatibility
    method to this config instance only; the upstream config class and
    configurations belonging to other models retain their original method.
    """
    original = config.get_scheme_dict
    if getattr(original, "_flagos_m31_vision_qkv_mapping", False):
        return False

    @wraps(original)
    def get_scheme_dict(self, layer, layer_name=None):
        if (
            layer_name is not None
            and layer_name.startswith("vision_tower.")
            and layer_name.endswith(".qkv_proj")
            and self.packed_modules_mapping.get("qkv_proj")
            == PACKED_MODULES["qkv_proj"]
        ):
            scoped = copy(self)
            scoped.packed_modules_mapping = {
                **self.packed_modules_mapping,
                "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            }
            return original.__func__(scoped, layer, layer_name)
        return original(layer, layer_name)

    get_scheme_dict._flagos_m31_vision_qkv_mapping = True
    config.get_scheme_dict = MethodType(get_scheme_dict, config)
    return True


def prepare_minimax_m3_model(vllm_config) -> None:
    """Prepare only the selected MiniMax model, immediately before loading."""
    architectures = getattr(vllm_config.model_config.hf_config, "architectures", ())
    if not set(architectures or ()).intersection(
        {"MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"}
    ):
        return

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    if isinstance(vllm_config.quant_config, CompressedTensorsConfig):
        install_minimax_m3_vision_mapping(vllm_config.quant_config)
    patch_minimax_m3_fused_preprocess()


def install_minimax_m3_metadata(model_classes) -> bool:
    """Expose text packed projections without wrapping model constructors."""
    changed = False
    for cls in model_classes:
        if getattr(cls, "_flagos_m31_w8a8_metadata", False):
            continue
        cls.packed_modules_mapping = {
            **getattr(cls, "packed_modules_mapping", {}),
            **{name: list(parts) for name, parts in PACKED_MODULES.items()},
        }
        cls._flagos_m31_w8a8_metadata = True
        changed = True
    return changed


def apply_minimax_m3_patches() -> bool:
    """Expose model metadata without installing process-wide execution hooks."""
    try:
        from vllm.models.minimax_m3 import (
            MiniMaxM3SparseForCausalLM,
            MiniMaxM3SparseForConditionalGeneration,
        )
    except ModuleNotFoundError as error:
        if error.name not in {"vllm.models", "vllm.models.minimax_m3"}:
            raise
        return False
    return install_minimax_m3_metadata(
        (MiniMaxM3SparseForCausalLM, MiniMaxM3SparseForConditionalGeneration)
    )
