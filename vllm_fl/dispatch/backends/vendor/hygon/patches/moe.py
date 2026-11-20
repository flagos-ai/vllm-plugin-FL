# Copyright (c) 2026 BAAI. All rights reserved.

"""Compatibility patches for vLLM's MoE backend selection."""

import logging


logger = logging.getLogger(__name__)


def patch_int8_moe_quant_scheme() -> None:
    """Allow compressed-tensors W8A8 INT8 MoE on Hygon ROCm."""
    import vllm.model_executor.layers.fused_moe.experts.triton_moe as triton_moe
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )
    from vllm.platforms import current_platform

    original = triton_moe.TritonExperts._supports_quant_scheme

    def _supports_quant_scheme_hygon(weight_key, activation_key) -> bool:
        if original(weight_key, activation_key):
            return True
        return bool(
            (weight_key, activation_key)
            == (kInt8StaticChannelSym, kInt8DynamicTokenSym)
            and str(getattr(current_platform, "vendor_name", "")).lower() == "hygon"
            and current_platform.has_device_capability((7, 5))
        )

    triton_moe.TritonExperts._supports_quant_scheme = staticmethod(
        _supports_quant_scheme_hygon
    )
    logger.info("Patched TritonExperts int8 MoE quant scheme for Hygon ROCm")


_patch_int8_moe_quant_scheme = patch_int8_moe_quant_scheme

__all__ = ["patch_int8_moe_quant_scheme"]
