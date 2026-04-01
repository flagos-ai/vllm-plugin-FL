# Copyright (c) 2026 BAAI. All rights reserved.

import logging

import vllm

logger = logging.getLogger(__name__)
_patches_applied = False

def apply_ascend_patches():
    """Apply all Ascend-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True
    # Patch modules for Ascend platform
    patch_causal_conv1d()
    patch_fla_ops()
    patch_op_cls()

def patch_mamba_config():
    """Patch HybridAttentionMambaModelConfig for Ascend."""
    from .patches.patch_mamba_config import verify_and_update_config

    vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = verify_and_update_config
    logger.info("Patched HybridAttentionMambaModelConfig for Ascend")

def patch_causal_conv1d():
    """Patch causal_conv1d ops with Ascend implementations."""
    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as _conv1d_lib
        import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

        from .impl.causal_conv1d import causal_conv1d_fn as ascend_causal_conv1d_fn
        from .impl.causal_conv1d import (
            causal_conv1d_update_npu as ascend_causal_conv1d_update,
        )

        _conv1d_lib.causal_conv1d_fn = ascend_causal_conv1d_fn
        _conv1d_lib.causal_conv1d_update = ascend_causal_conv1d_update
        _qwen3_next_lib.causal_conv1d_fn = ascend_causal_conv1d_fn
        _qwen3_next_lib.causal_conv1d_update = ascend_causal_conv1d_update
        logger.info("Patched causal_conv1d ops for Ascend")
    except Exception as e:
        logger.warning("Failed to patch causal_conv1d ops: %s", e)

def patch_fla_ops():
    """Patch FLA ops and fused_gdn_gating with Ascend implementations."""
    try:
        import vllm.model_executor.layers.fla.ops as _fla_ops_lib
        import vllm.model_executor.layers.fla.ops.chunk as _fla_chunk_lib
        import vllm.model_executor.layers.fla.ops.fused_recurrent as _fla_recurrent_lib
        import vllm.model_executor.layers.fla.ops.layernorm_guard as _fla_layernorm_lib
        import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

        from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.chunk import (
            chunk_gated_delta_rule as ascend_chunk_gated_delta_rule,
        )
        from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule as ascend_fused_recurrent,
        )
        from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_fwd_kernel as ascend_fused_recurrent_kernel,
        )
        from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.layernorm_guard import (
            LayerNormFn as ascend_LayerNormFn,
        )

        _fla_ops_lib.chunk_gated_delta_rule = ascend_chunk_gated_delta_rule
        _fla_chunk_lib.chunk_gated_delta_rule = ascend_chunk_gated_delta_rule
        _fla_ops_lib.fused_recurrent_gated_delta_rule = ascend_fused_recurrent
        _fla_recurrent_lib.fused_recurrent_gated_delta_rule = ascend_fused_recurrent
        _fla_recurrent_lib.fused_recurrent_gated_delta_rule_fwd_kernel = (
            ascend_fused_recurrent_kernel
        )
        _fla_layernorm_lib.LayerNormFn = ascend_LayerNormFn
        _qwen3_next_lib.chunk_gated_delta_rule = ascend_chunk_gated_delta_rule
        _qwen3_next_lib.fused_recurrent_gated_delta_rule = ascend_fused_recurrent
        logger.info("Patched FLA ops + fused_gdn_gating for Ascend")
    except Exception as e:
        logger.warning("Failed to patch FLA ops: %s", e)

def patch_op_cls():
    """Patch MMEncoderAttention to use manual matmul attention on NPU.

    The NPU npu_fused_infer_attention_score kernel only supports head_dim
    in {64, 128, 192}. The vision encoder may have non-standard head_dim
    (e.g. 72 for Qwen3.5). F.scaled_dot_product_attention on NPU may also
    dispatch to the same problematic kernel. Use pure-PyTorch matmul
    attention instead.
    """
    try:
        from vllm.model_executor.custom_op import CustomOp

        from .impl.fused_moe import AscendSharedFusedMoE
        from .impl.mm_encoder_attention import AscendMMEncoderAttention
        from .impl.vocab_parallel_embedding import AscendVocabParallelEmbedding
        REGISTERED_ASCEND_OPS = {
            "SharedFusedMoE": AscendSharedFusedMoE,
            "VocabParallelEmbedding": AscendVocabParallelEmbedding,
            "MMEncoderAttention": AscendMMEncoderAttention,
            }
        for name, op_cls in REGISTERED_ASCEND_OPS.items():
            CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)
        logger.info("Patched MMEncoderAttention for NPU (matmul attention)")
    except Exception as e:
        logger.warning("Failed to patch MMEncoderAttention: %s", e)
