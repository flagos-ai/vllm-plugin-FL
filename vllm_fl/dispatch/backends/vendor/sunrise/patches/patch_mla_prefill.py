# Copyright (c) 2026 BAAI. All rights reserved.
"""Register Sunrise's MLA prefill backend into vLLM 0.24's prefill registry.

vLLM 0.24 selects the MLA prefill backend via ``get_mla_prefill_backend``. On
PTPU, ``get_device_capability()`` returns ``None``, so the selector
short-circuits to ``MLAPrefillBackendEnum.FLASH_ATTN`` without consulting user
config. The stock ``FlashAttnPrefillBackend`` asserts on ``vllm_flash_attn``
(absent on PTPU), so we override the ``FLASH_ATTN`` registry entry with our
FlagGems-backed backend. This is the only injection point that survives the
``device_capability is None`` short-circuit.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)


def apply_patch() -> None:
    """Override the FLASH_ATTN MLA prefill backend on PTPU. Idempotent."""
    from vllm.platforms import current_platform

    if getattr(current_platform, "device_type", None) != "ptpu":
        return

    from vllm.v1.attention.backends.mla.prefill.registry import (
        MLAPrefillBackendEnum,
        register_mla_prefill_backend,
    )

    register_mla_prefill_backend(
        MLAPrefillBackendEnum.FLASH_ATTN,
        "vllm_fl.dispatch.backends.vendor.sunrise.impl.mla_prefill."
        "SunriseMLAPrefillBackend",
    )
    logger.info_once(
        "Sunrise MLA: overrode FLASH_ATTN prefill backend with "
        "SunriseMLAPrefillBackend (FlagGems flash_attn_varlen_func)."
    )
