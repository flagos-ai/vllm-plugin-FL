from vllm.logger import init_logger

logger = init_logger(__name__)


def _is_uva_available() -> bool:
    return False


def patch_v2_uva() -> None:
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", "") != "iluvatar":
        return

    from vllm.v1.worker.gpu import buffer_utils

    if buffer_utils.is_uva_available is _is_uva_available:
        return
    buffer_utils.is_uva_available = _is_uva_available
    logger.info("Iluvatar V2 buffers use explicit host-to-device copies instead of UVA")
