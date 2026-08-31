# Copyright (c) 2026 BAAI. All rights reserved.

"""Orchestrate categorized Hygon vLLM compatibility patches."""

import logging

from .patches.custom_ops import patch_custom_op_dispatch
from .patches.models.deepseek_v4 import (
    _hygon_has_cutedsl,
    _install_compressed_tensors_scale_fallback,
    _scale_inv_alias,
    patch_compressed_tensors_scale_fallback,
    patch_deepseek_v4_bf16_indexer_cache,
    patch_deepseek_v4_cutedsl_selection,
    patch_deepseek_v4_qnorm_rope_kv_insert,
    patch_mhc_ops,
    patch_sparse_attn_indexer,
)
from .patches.moe import patch_int8_moe_quant_scheme
from .patches.native_ops import (
    patch_moe_sum,
    patch_topk_softplus_sqrt,
)


logger = logging.getLogger(__name__)
_patches_applied = False


def apply_hygon_patches() -> None:
    """Install all Hygon operator, backend, and model patches once."""
    global _patches_applied
    if _patches_applied:
        return

    # Keep kernel-heavy modules lazy, matching the previous entry-point
    # behavior and allowing package metadata to be inspected without Triton.
    from .impl.attention.mla import patch_rocm_wo_a_int8_group_gemm
    from vllm_fl.ops.native_ops_registry import register_native_ops

    register_native_ops()
    patch_rocm_wo_a_int8_group_gemm()

    patch_custom_op_dispatch()
    # patch_topk_softplus_sqrt()  # Managed by FL dispatch.
    # patch_moe_sum()  # Replaced by the YAML-controlled native-op bridge.
    patch_mhc_ops()
    patch_deepseek_v4_cutedsl_selection()
    patch_deepseek_v4_bf16_indexer_cache()
    patch_deepseek_v4_qnorm_rope_kv_insert()
    patch_sparse_attn_indexer()
    patch_int8_moe_quant_scheme()
    patch_compressed_tensors_scale_fallback()

    _patches_applied = True
    logger.info("Applied Hygon vLLM compatibility patches")


# Compatibility exports for existing tests and downstream imports. New code
# should import the public ``patch_*`` names from the categorized modules.
_patch_custom_op_dispatch = patch_custom_op_dispatch
_patch_topk_softplus_sqrt = patch_topk_softplus_sqrt
_patch_moe_sum = patch_moe_sum
_patch_mhc_ops = patch_mhc_ops
_patch_deepseek_v4_cutedsl_selection = patch_deepseek_v4_cutedsl_selection
_patch_deepseek_v4_bf16_indexer_cache = patch_deepseek_v4_bf16_indexer_cache
_patch_deepseek_v4_qnorm_rope_kv_insert = patch_deepseek_v4_qnorm_rope_kv_insert
_patch_sparse_attn_indexer = patch_sparse_attn_indexer
_patch_int8_moe_quant_scheme = patch_int8_moe_quant_scheme
_patch_compressed_tensors_scale_fallback = patch_compressed_tensors_scale_fallback


__all__ = ["apply_hygon_patches"]
