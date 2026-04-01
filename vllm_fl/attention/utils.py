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
    import vllm.attention.layers.mm_encoder_attention as mm_mod
    from vllm.attention.backends.registry import AttentionBackendEnum

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
