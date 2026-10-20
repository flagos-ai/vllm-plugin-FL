# Copyright (c) 2026 BAAI. All rights reserved.

"""Select real FlagGems operations for an EMPTY Hopper DSV4.1 worker.

The upstream model owns its forward, cache layout and runner integration.
Only the external kernel entry points of that model are replaced here.
"""

from vllm_fl.nvidia_compat import has_native_vllm_kernels

_installed = False


def install_dsv41_empty_adapters(vllm_config) -> bool:
    global _installed
    if has_native_vllm_kernels():
        return False
    model = vllm_config.model_config
    if model is None or "DeepseekV41ForCausalLM" not in model.architectures:
        return False

    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is None or capability.major != 9:
        raise NotImplementedError(
            "FlagGems DSV4.1 EMPTY currently requires Hopper SM90"
        )
    if vllm_config.speculative_config is not None:
        raise NotImplementedError(
            "FlagGems DSV4.1 EMPTY does not yet implement DSpark/speculative decode"
        )
    attention = vllm_config.attention_config
    allowed = {None, "FLASHMLA_SPARSE", "FLASHMLA_SPARSE_DSV4", "FLASHMLA_SPARSE_DSV41"}
    if getattr(attention.backend, "name", attention.backend) not in allowed:
        raise NotImplementedError(
            "FlagGems DSV4.1 EMPTY supports the Hopper sparse MLA backend"
        )
    if (
        attention.indexer_kv_dtype not in ("auto", "fp8")
        or attention.indexer_sparse_logits
    ):
        raise NotImplementedError(
            "FlagGems DSV4.1 EMPTY supports FP8 dense indexer logits only"
        )
    if vllm_config.cache_config.cache_dtype not in ("auto", "fp8", "fp8_ds_mla"):
        raise NotImplementedError(
            "FlagGems DSV4.1 EMPTY requires the 584-byte fp8_ds_mla cache"
        )
    from vllm_fl.ops.nvidia_sparse_indexer import register_nvidia_sparse_indexer

    if not register_nvidia_sparse_indexer():
        return False
    if _installed:
        return True

    from vllm.logger import init_logger
    from vllm.model_executor.layers import sparse_attn_indexer
    from vllm.models.deepseek_v4.nvidia.ops import o_proj
    from vllm.models.deepseek_v41.nvidia import flashmla

    from vllm_fl.dispatch.backends.flaggems.impl import dsv41_mla
    from vllm_fl.ops import nvidia_dsv41_indexer
    from vllm_fl.ops.nvidia_dsv41_ops import register_nvidia_dsv41_ops

    register_nvidia_dsv41_ops()
    flashmla.flash_mla_sparse_fwd = dsv41_mla.flash_mla_sparse_fwd
    flashmla.flash_mla_with_kvcache = dsv41_mla.flash_mla_with_kvcache
    flashmla.DeepseekSparseSWAFlashMLAMetadataBuilder.build_tile_scheduler = (
        dsv41_mla.build_tile_scheduler
    )
    sparse_attn_indexer.fp8_fp4_mqa_logits = nvidia_dsv41_indexer.fp8_mqa_logits
    sparse_attn_indexer.fp8_fp4_paged_mqa_logits = (
        nvidia_dsv41_indexer.fp8_paged_mqa_logits
    )
    o_proj.fp8_einsum = nvidia_dsv41_indexer.fp8_einsum
    _installed = True
    init_logger(__name__).info(
        "Using FlagGems DSV4.1 sparse MLA and FP8 indexer on Hopper; "
        "native FlashMLA/DeepGEMM availability is unchanged"
    )
    return True
