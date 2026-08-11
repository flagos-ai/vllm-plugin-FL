from vllm.logger import init_logger

logger = init_logger(__name__)


def patch_mm_encoder_attention():
    """
    Patch vllm.attention.layers.mm_encoder_attention.maybe_get_vit_flash_attn_backend
    to support OOT platforms.

    The original implementation imports flash_attn_varlen_func from fa_utils,
    which may not have it defined for OOT platforms. This patch changes the
    FLASH_ATTN branch to import directly from vllm.vllm_flash_attn with a
    fallback to flash_attn.
    """
    import vllm.model_executor.layers.attention.mm_encoder_attention as mm_mod
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.platforms import current_platform

    # PlatformFL is registered as an OOT platform, so CustomOp would otherwise
    # bind MMEncoderAttention to forward_oot -> forward_native (Torch SDPA).
    # NVIDIA must keep vLLM's CUDA dispatch so the selected FLASH_ATTN backend
    # is actually used during multimodal profiling and inference.
    if current_platform.is_cuda():
        mm_mod.MMEncoderAttention.forward_oot = mm_mod.MMEncoderAttention.forward_cuda

    def _patched_maybe_get_vit_flash_attn_backend(attn_backend):
        if attn_backend == AttentionBackendEnum.FLASH_ATTN:
            try:
                from vllm.vllm_flash_attn import flash_attn_varlen_func

                logger.info_once("Using vllm.vllm_flash_attn for vit attention")
            except (ImportError, ModuleNotFoundError):
                from flash_attn import flash_attn_varlen_func

                logger.info_once("Using flash_attn for vit attention")
            return flash_attn_varlen_func
        elif attn_backend == AttentionBackendEnum.ROCM_AITER_FA:
            from aiter import flash_attn_varlen_func

            return flash_attn_varlen_func
        else:
            return None

    mm_mod.maybe_get_vit_flash_attn_backend = _patched_maybe_get_vit_flash_attn_backend
