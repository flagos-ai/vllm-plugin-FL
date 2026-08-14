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
    patch_fused_moe()
    patch_qwen3_5_attention()
    patch_qwen3_6_gdn()
    patch_qwen3_mtp()
    patch_graph()
    patch_npugraph_ex()
    patch_dynamo_safe_ops()

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
    """Patch fused MoE ops with Ascend implementations.

    Always replaces ``fused_experts_impl`` (it dispatches between the AscendC
    custom-op path and the legacy torch_npu path at call time based on the
    weight layout).  When the AscendC MoE custom ops are available
    (``ascendc_moe_available``), additionally:

    * replace ``fused_topk`` with the fused AscendC ``moe_gating_top_k``
      kernel (softmax + top-k + renorm in one launch);
    * wrap ``UnquantizedFusedMoEMethod.process_weights_after_loading`` so
      expert weights are stored pre-transposed, removing the per-forward
      ``transpose(1, 2).contiguous()`` copies of the legacy path.
    """
    # TODO ops' triton implementation is not ready yet
    from .impl.fused_moe import (
        ascendc_moe_available,
        convert_moe_weights_pretransposed,
        fused_experts_impl,
        fused_topk_ascend,
    )
    try:
        import vllm_fl.ops.fused_moe.fused_moe as fused_moe_lib

        fused_moe_lib.fused_experts_impl = fused_experts_impl

        logger.info("Patched fused_moe for Ascend")
    except Exception as e:
        logger.warning("Failed to patch fused_moe ops: %s", e)
        fused_moe_lib = None

    if fused_moe_lib is None or not ascendc_moe_available():
        return

    try:
        fused_moe_lib.fused_topk = fused_topk_ascend
        logger.info("Patched fused_topk with AscendC moe_gating_top_k")
    except Exception as e:
        logger.warning("Failed to patch fused_topk: %s", e)

    try:
        from vllm.model_executor.layers.fused_moe.layer import (
            UnquantizedFusedMoEMethod,
        )

        orig_process_weights = UnquantizedFusedMoEMethod.process_weights_after_loading

        def process_weights_after_loading_pretransposed(self, layer):
            orig_process_weights(self, layer)
            convert_moe_weights_pretransposed(layer)

        UnquantizedFusedMoEMethod.process_weights_after_loading = (
            process_weights_after_loading_pretransposed
        )
        logger.info("Patched MoE weight loading with pre-transposed layout for AscendC ops")
    except Exception as e:
        logger.warning("Failed to patch MoE process_weights_after_loading: %s", e)

def patch_qwen3_5_attention():
    """Patch Qwen3.5/Qwen3.6 attention to use the fused Ascend kernel."""
    try:
        from .patches.patch_qwen3_5 import patch_qwen3_5_attention as _do_patch

        _do_patch()
    except Exception as e:
        logger.warning("Failed to patch Qwen3NextAttention for Ascend: %s", e)


def patch_qwen3_6_gdn():
    """Patch Qwen3.5/Qwen3.6 GatedDeltaNet and GemmaRMSNorm with AscendC ops.

    Falls back to the existing Triton path when the CANN custom-op package
    is not available at runtime.
    """
    try:
        from .patches.patch_qwen3_6_gdn import patch_qwen3_6_gdn as _do_patch

        _do_patch()
    except Exception as e:
        logger.warning("Failed to patch Qwen3.6 GDN AscendC ops: %s", e)


def patch_qwen3_mtp():
    """Patch Qwen3.5/Qwen3.6 Multi-Token Prediction for Ascend."""
    try:
        from .patches.patch_qwen3_mtp import patch_qwen3_mtp as _do_patch

        _do_patch()
    except Exception as e:
        logger.warning("Failed to patch Qwen3 MTP for Ascend: %s", e)


def patch_graph():
    """Patch GraphWrapper with Ascend ACL graph behavior."""
    try:
        from .patches.patch_graph import patch_graph as _do_patch

        _do_patch()
    except Exception as e:
        logger.warning("Failed to patch GraphWrapper for Ascend: %s", e)


def patch_npugraph_ex():
    """Patch npugraph_ex/torchair ValuePack handling."""
    try:
        from .patches.patch_npugraph_ex import patch_npugraph_ex as _do_patch

        _do_patch()
    except Exception as e:
        logger.warning("Failed to patch npugraph_ex for Ascend: %s", e)


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

def patch_dynamo_safe_ops():
    """Bypass the FL dispatch manager for OOT ops when running on NPU.

    vLLM v1 graph mode compiles model forward with torch.compile(fullgraph=True).
    The FL dispatch manager's call_op uses Python RLock/context managers that
    Dynamo cannot trace, so on NPU we replace the OOT forward methods with
    direct calls to the Ascend (or reference PyTorch) implementations.  This
    keeps the dispatch manager available for eager-mode / non-graph use.
    """
    try:
        from vllm.platforms import current_platform

        if getattr(current_platform, "device_type", None) != "npu":
            return

        from vllm_fl.ops.activation import GeluAndMulFL, SiluAndMulFL
        from vllm_fl.ops.layernorm import RMSNormFL
        from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

        from .impl.activation import silu_and_mul_ascend
        from .impl.normalization import rms_norm_ascend
        from .impl.rotary import rotary_embedding_ascend
        from vllm_fl.dispatch.backends.reference.impl.activation import (
            gelu_and_mul_torch,
            silu_and_mul_torch,
        )

        SiluAndMulFL.forward_oot = silu_and_mul_ascend
        GeluAndMulFL.forward_oot = gelu_and_mul_torch
        RMSNormFL.forward_oot = rms_norm_ascend
        RotaryEmbeddingFL.forward_oot = _make_rotary_forward_oot(
            rotary_embedding_ascend
        )

        logger.info("Patched FL OOT ops for Dynamo-safe NPU execution")
    except Exception as e:
        logger.warning("Failed to patch FL OOT ops for NPU: %s", e)


def _make_rotary_forward_oot(rotary_impl):
    """Build a RotaryEmbeddingFL.forward_oot that calls ``rotary_impl`` directly.

    The original forward_oot reshapes query/key and extracts cos/sin before
    calling the operator; we keep that logic and only bypass call_op.
    """

    def forward_oot(
        self,
        positions: "torch.Tensor",
        query: "torch.Tensor",
        key: "torch.Tensor | None" = None,
    ) -> tuple["torch.Tensor", "torch.Tensor | None"]:
        # Use a local tensor instead of assigning back to the buffer; assigning
        # to self.cos_sin_cache inside forward is forbidden when cudagraph is
        # used inside torch.compile.
        cos_sin_cache = self.cos_sin_cache.to(positions.device)
        positions = positions.flatten()
        num_tokens = positions.shape[0]

        query_shape = query.shape
        key_shape = key.shape
        query = query.view(num_tokens, -1, self.head_size)
        key = key.view(num_tokens, -1, self.head_size)

        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        cos, sin = cos_sin_cache.chunk(2, dim=-1)

        q_embed, k_embed = rotary_impl(
            self,
            query_rot,
            key_rot,
            cos,
            sin,
            positions,
            not self.is_neox_style,
            True,
        )

        if self.rotary_dim < self.head_size:
            query = torch.cat((q_embed, query_pass), dim=-1).reshape(query_shape)
            key = torch.cat((k_embed, key_pass), dim=-1).reshape(key_shape)
        else:
            query = q_embed.reshape(query_shape)
            key = k_embed.reshape(key_shape)

        return query, key

    return forward_oot


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

    # TODO(MengqingCao): Remove the model_type check, after resolving the hidden error in get_kv_cache_groups.
    if model_config.hf_text_config.model_type != "qwen3_next" and cache_config.block_size != block_size:
        if cache_config.enable_prefix_caching or scheduler_config.enable_chunked_prefill:
            logger.info(f"Block size is set to {block_size} if prefix cache or chunked prefill is enabled.")
            cache_config.block_size = block_size
