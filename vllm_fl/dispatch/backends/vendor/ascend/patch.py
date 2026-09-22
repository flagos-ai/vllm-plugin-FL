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
    from .patches.triton_compat import patch_triton_compile_hooks

    patch_triton_compile_hooks()
    patch_topk_topp_sampler()
    # Patch modules for Ascend platform
    patch_mamba_batch_memcpy()
    patch_causal_conv1d()
    patch_fla_ops()
    patch_op_cls()
    patch_fused_moe()


def patch_topk_topp_sampler():
    """Use vLLM's PyTorch sampler when FlagTree cannot lower large batches."""
    try:
        import vllm.v1.sample.ops.topk_topp_sampler as sampler

        # vLLM selects its Triton kernel at batch sizes of eight or larger.
        # FlagTree 0.6.2a1 cannot lower that kernel on 910C, including the
        # max-num-seqs profiling batch used by the OpenAI server.
        sampler.HAS_TRITON = False
        logger.info("Disabled the vLLM Triton top-k/top-p sampler for Ascend")
    except Exception as exc:
        logger.warning("Failed to patch the top-k/top-p sampler: %s", exc)


def patch_mamba_batch_memcpy():
    """Replace vLLM's Mamba state copy with the Ascend-safe Triton kernel."""
    try:
        import torch

        from vllm.v1.worker import mamba_utils

        from .impl.batch_memcpy import batch_memcpy, batch_memcpy_kernel

        mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel
        mamba_utils.batch_memcpy = batch_memcpy

        create = mamba_utils.MambaCopyBuffers.create
        create_func = getattr(create, "__func__", create)
        if not getattr(create_func, "_vllm_fl_ascend_patched", False):
            original_create = create

            @classmethod
            def _patched_create(
                cls,
                max_num_reqs,
                kv_cache_config,
                copy_funcs,
                make_buffer,
            ):
                def _make_buffer(*args, **kwargs):
                    if kwargs.get("dtype") == torch.uint64:
                        kwargs["dtype"] = torch.int64
                    return make_buffer(*args, **kwargs)

                return original_create(
                    max_num_reqs,
                    kv_cache_config,
                    copy_funcs,
                    _make_buffer,
                )

            _patched_create.__func__._vllm_fl_ascend_patched = True
            mamba_utils.MambaCopyBuffers.create = _patched_create
        logger.info("Patched Mamba batch_memcpy for Ascend")
    except Exception as exc:
        logger.warning("Failed to patch Mamba batch_memcpy: %s", exc)


def patch_causal_conv1d():
    """Patch causal_conv1d ops with Ascend implementations."""
    try:
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as _qwen_gdn_lib
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as _conv1d_lib

        from .impl.causal_conv1d import (
            causal_conv1d_fn as causal_conv1d_fn_npu,
            causal_conv1d_update_npu,
        )

        _conv1d_lib.causal_conv1d_fn = causal_conv1d_fn_npu
        _conv1d_lib.causal_conv1d_update = causal_conv1d_update_npu
        _qwen_gdn_lib.causal_conv1d_fn = causal_conv1d_fn_npu
        _qwen_gdn_lib.causal_conv1d_update = causal_conv1d_update_npu
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
    """Bridge vLLM 0.28 Qwen GDN prefill to Ascend FlagGems."""
    try:
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as _qwen_gdn_lib
        import vllm.third_party.flash_linear_attention.ops as _fla_ops_lib
        import vllm.third_party.flash_linear_attention.ops.chunk as _fla_chunk_lib

        from .impl.fla.compat import (
            chunk_gated_delta_rule as chunk_gated_delta_rule_npu,
        )

        _fla_ops_lib.chunk_gated_delta_rule = chunk_gated_delta_rule_npu
        _fla_chunk_lib.chunk_gated_delta_rule = chunk_gated_delta_rule_npu
        _qwen_gdn_lib.fla_chunk_gated_delta_rule = chunk_gated_delta_rule_npu
        logger.info("Patched FLA ops for Ascend")
    except Exception as e:
        logger.warning("Failed to patch FLA ops: %s", e)


def patch_op_cls():
    """Register NPU embedding and padded native vision attention.

    The vision implementation pads head dimensions such as Qwen's 72 to
    128 for fused infer attention and retains the original scale.
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
        logger.info("Patched MMEncoderAttention for NPU (padded FIA)")
    except Exception as e:
        logger.warning("Failed to patch MMEncoderAttention: %s", e)


def refresh_block_size(vllm_config, block_size=128):
    """
    Refresh the block size in cache config.
    """
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config

    if not cache_config:
        return

    # vLLM 0.28 aligns hybrid attention/Mamba page sizes in
    # Platform.update_block_size_for_backend, after the attention backend is
    # known. Do not replace that upstream-aligned value with the generic NPU
    # block size.
    if model_config is not None and getattr(model_config, "is_hybrid", False):
        return

    if cache_config.block_size is None:
        cache_config.block_size = block_size

    if not scheduler_config or not model_config:
        return

    # TODO(MengqingCao): Remove the model_type check, after resolving the hidden error in get_kv_cache_groups.
    if (
        model_config.hf_text_config.model_type != "qwen3_next"
        and cache_config.block_size != block_size
        and (
            cache_config.enable_prefix_caching
            or scheduler_config.enable_chunked_prefill
        )
    ):
        logger.info(
            "Block size is set to %s if prefix cache or chunked prefill is enabled.",
            block_size,
        )
        cache_config.block_size = block_size
