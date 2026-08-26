# Copyright (c) 2026 BAAI. All rights reserved.

"""Ascend patches owned by FL.

Only the current vLLM-Ascend-derived Qwen3.5/Qwen3.6 path is installed here.
Legacy FlagGems and pure-PyTorch GDN fallbacks intentionally do not belong to
this registration chain.
"""

from __future__ import annotations

import logging

import vllm

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_ascend_patches() -> None:
    """Install the FL-local Ascend implementations used by Qwen3.6."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_mamba_config()
    patch_gdn_ascend_ops()
    patch_op_cls()

    # Importing linearnorm registers the current Ascend custom-op schemas and
    # implementations used by the copied Qwen attention forward.
    from .impl.triton.linearnorm import (  # noqa: F401
        split_qkv_rmsnorm_mrope,
        split_qkv_rmsnorm_rope,
        split_qkv_tp_rmsnorm_rope,
    )
    from .patches.patch_qwen3_5 import apply_qwen3_5_patch

    apply_qwen3_5_patch()
    patch_vit_pos_embed()


def patch_gdn_ascend_ops() -> None:
    """Load the FL-local Ascend ABI and patch GDN scheduling metadata."""
    try:
        from vllm_fl.ascend_custom_ops import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("FL Ascend custom operators are unavailable")

        # Import applies the metadata-builder patch as a module side effect.
        from .patches import patch_gdn_attn  # noqa: F401

        logger.info("Enabled FL-local Ascend C GDN operators and metadata builder")
    except Exception as exc:
        logger.warning("Failed to enable FL-local Ascend C GDN operators: %s", exc)


def patch_mamba_config() -> None:
    """Install the Ascend hybrid-attention cache sizing rules."""
    from .patches.patch_mamba_config import verify_and_update_config

    vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = (  # noqa: E501
        verify_and_update_config
    )
    logger.info("Patched HybridAttentionMambaModelConfig for Ascend")


def patch_op_cls() -> None:
    """Register current FL-local Ascend out-of-tree implementations."""
    try:
        from vllm.model_executor.custom_op import op_registry_oot

        from .impl.activation_layer import AscendQuickGELU, AscendSiluAndMul
        from .impl.gdn import AscendGatedDeltaNetAttention
        from .impl.layernorm import (
            AscendGemmaRMSNorm,
            AscendRMSNorm,
            AscendRMSNormGated,
        )
        from .impl.linear import (
            AscendColumnParallelLinear,
            AscendMergedColumnParallelLinear,
            AscendQKVParallelLinear,
            AscendReplicatedLinear,
            AscendRowParallelLinear,
        )
        from .impl.mm_encoder_attention import AscendMMEncoderAttention
        from .impl.vocab_parallel_embedding import (
            AscendLogitsProcessor,
            AscendParallelLMHead,
            AscendVocabParallelEmbedding,
        )

        registered_ascend_ops = {
            "QuickGELU": AscendQuickGELU,
            "SiluAndMul": AscendSiluAndMul,
            "ColumnParallelLinear": AscendColumnParallelLinear,
            "RowParallelLinear": AscendRowParallelLinear,
            "MergedColumnParallelLinear": AscendMergedColumnParallelLinear,
            "QKVParallelLinear": AscendQKVParallelLinear,
            "ReplicatedLinear": AscendReplicatedLinear,
            "VocabParallelEmbedding": AscendVocabParallelEmbedding,
            "ParallelLMHead": AscendParallelLMHead,
            "LogitsProcessor": AscendLogitsProcessor,
            "MMEncoderAttention": AscendMMEncoderAttention,
            "GatedDeltaNetAttention": AscendGatedDeltaNetAttention,
            "RMSNorm": AscendRMSNorm,
            "GemmaRMSNorm": AscendGemmaRMSNorm,
            "RMSNormGated": AscendRMSNormGated,
        }
        for name, op_cls in registered_ascend_ops.items():
            # FL's generic OOT layer set is registered before the platform
            # callback runs.  Ascend must replace those entries here instead
            # of calling register_oot(), which rejects duplicate names and
            # previously left RMSNormFL active for this model.
            op_cls.name = name
            op_registry_oot[name] = op_cls
        logger.info("Installed FL-local Ascend custom-op implementations")
    except Exception as exc:
        logger.warning("Failed to register FL-local Ascend custom ops: %s", exc)


def refresh_block_size(vllm_config, block_size: int = 128) -> None:
    """Apply the generic Ascend cache block size without breaking hybrids."""
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config

    if not cache_config:
        return
    if cache_config.block_size is None:
        cache_config.block_size = block_size
    if not scheduler_config or not model_config:
        return

    # Hybrid attention/Mamba models use the page size produced by the patched
    # HybridAttentionMambaModelConfig. Replacing it with 128 here corrupts the
    # combined attention/SSM page layout.
    if model_config.is_hybrid:
        return

    if (
        cache_config.block_size != block_size
        and (cache_config.enable_prefix_caching or scheduler_config.enable_chunked_prefill)
    ):
        logger.info(
            "Setting Ascend block size to %s because prefix caching or "
            "chunked prefill is enabled.",
            block_size,
        )
        cache_config.block_size = block_size


def patch_vit_pos_embed() -> None:
    """Use vLLM's native ViT interpolation instead of its CUDA Triton path."""
    try:
        import vllm.model_executor.models.qwen3_vl as qwen3_vl

        if getattr(qwen3_vl, "HAS_TRITON", False):
            qwen3_vl.HAS_TRITON = False
            logger.info("Forced native ViT position interpolation on Ascend")
    except Exception as exc:
        logger.warning("Failed to patch ViT position interpolation: %s", exc)
