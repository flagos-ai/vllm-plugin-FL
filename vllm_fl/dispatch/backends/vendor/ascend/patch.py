# Copyright (c) 2026 BAAI. All rights reserved.

import logging

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
    patch_fused_moe()
    patch_gdn()
    patch_vision_position_interpolation()
    patch_sampler_filter()
    from .patches.patch_mamba_utils import apply_mamba_copy_patch

    apply_mamba_copy_patch()


def patch_sampler_filter():
    """Use the packaged A2/A3 filter while retaining vLLM's sampler contract."""
    import torch_npu
    from vllm.v1.sample.ops import topk_topp_sampler

    def apply_top_k_top_p(logits, k, p):
        if k is None and p is None:
            return logits
        # vLLM's metadata uses int64; CANN's filter accepts int32 only.
        if k is not None:
            import torch

            k = k.to(torch.int32)
        return torch_npu.npu_top_k_top_p(logits, k=k, p=p)

    # vLLM chooses its CUDA-oriented Triton filter for batches >=8 even on
    # OOT platforms. Match rc1's built-in A2/A3 filtering path instead.
    topk_topp_sampler.apply_top_k_top_p = apply_top_k_top_p


def patch_vision_position_interpolation():
    """Use vLLM's tensor reference instead of the CUDA-oriented launch grid."""
    from vllm.model_executor.models import qwen3_vl

    # The Triton implementation launches one block per spatial patch. Vision
    # profiling can exceed Ascend's 65535-block limit; the upstream tensor
    # implementation has the same interpolation/spatial-merge contract.
    qwen3_vl.triton_pos_embed_interpolate = qwen3_vl.pos_embed_interpolate_native


def patch_gdn():
    """Keep the current upstream GDN model, replacing its numerical core."""
    from vllm.model_executor.custom_op import PluggableLayer

    from .impl.gdn import AscendQwenGatedDeltaNetAttention

    PluggableLayer.register_oot(
        _decorated_layer_cls=AscendQwenGatedDeltaNetAttention,
        name="QwenGatedDeltaNetAttention",
    )

def patch_causal_conv1d():
    """Patch causal_conv1d ops with Ascend implementations."""
    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as _conv1d_lib
        import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

        from .impl.causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_npu
        from .impl.causal_conv1d import causal_conv1d_update_npu

        _conv1d_lib.causal_conv1d_fn = causal_conv1d_fn_npu
        _conv1d_lib.causal_conv1d_update = causal_conv1d_update_npu
        _qwen3_next_lib.causal_conv1d_fn = causal_conv1d_fn_npu
        _qwen3_next_lib.causal_conv1d_update = causal_conv1d_update_npu
        logger.info("Patched causal_conv1d ops for Ascend")
    except Exception as e:
        logger.warning("Failed to patch causal_conv1d ops: %s", e)

def patch_fused_moe():
    """Patch fused MoE ops with Ascend implementations."""
    # TODO ops' triton implementation is not ready yet
    from .impl.fused_moe import fused_experts_impl
    try:
        import vllm_fl.ops.fused_moe.fused_moe as fused_moe_lib

        fused_moe_lib.fused_experts_impl = fused_experts_impl

        logger.info("Patched fused_moe for Ascend")
    except Exception as e:
        logger.warning("Failed to patch fused_moe ops: %s", e)

def patch_fla_ops():
    """Patch FLA ops and fused_gdn_gating with Ascend implementations."""
    try:
        import vllm.model_executor.layers.fla.ops as _fla_ops_lib
        import vllm.model_executor.layers.fla.ops.chunk as _fla_chunk_lib
        import vllm.model_executor.layers.fla.ops.fused_recurrent as _fla_recurrent_lib
        import vllm.model_executor.layers.fla.ops.layernorm_guard as _fla_layernorm_lib
        import vllm.model_executor.models.qwen3_next as _qwen3_next_lib
        from flag_gems.runtime.backend._ascend.fla import (
            chunk_gated_delta_rule_fwd,
            fused_recurrent_gated_delta_rule_fwd,
        )
        from flag_gems.runtime.backend._ascend.fla.layernorm_guard import (
            LayerNormFn as ascend_LayerNormFn,
        )

        from .impl.fla import chunk_gated_delta_rule_npu

        _fla_ops_lib.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd
        _fla_chunk_lib.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd
        _fla_chunk_lib.chunk_gated_delta_rule = chunk_gated_delta_rule_npu
        _fla_recurrent_lib.fused_recurrent_gated_delta_rule_fwd = fused_recurrent_gated_delta_rule_fwd
        _fla_layernorm_lib.LayerNormFn = ascend_LayerNormFn
        _qwen3_next_lib.chunk_gated_delta_rule = chunk_gated_delta_rule_npu
        logger.info("Patched FLA ops for Ascend")
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

        from .impl.mm_encoder_attention import AscendMMEncoderAttention
        from .impl.vocab_parallel_embedding import AscendVocabParallelEmbedding
        REGISTERED_ASCEND_OPS = {
            "VocabParallelEmbedding": AscendVocabParallelEmbedding,
            "MMEncoderAttention": AscendMMEncoderAttention,
        }
        for name, op_cls in REGISTERED_ASCEND_OPS.items():
            CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)
        logger.info("Patched MMEncoderAttention for NPU (matmul attention)")
    except Exception as e:
        logger.warning("Failed to patch MMEncoderAttention: %s", e)

def refresh_block_size(vllm_config, block_size = 128):
    """
    Refresh the block size in cache config.
    """
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config

    if not cache_config:
        return

    if cache_config.block_size is None:
        cache_config.block_size = block_size

    if not scheduler_config or not model_config:
        return

    # Hybrid state pages are aligned by the Mamba configuration verifier.
    if not model_config.is_hybrid and cache_config.block_size != block_size:
        if cache_config.enable_prefix_caching or scheduler_config.enable_chunked_prefill:
            logger.info(f"Block size is set to {block_size} if prefix cache or chunked prefill is enabled.")
            cache_config.block_size = block_size
