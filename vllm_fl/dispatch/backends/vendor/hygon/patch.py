# Copyright (c) 2026 BAAI. All rights reserved.

"""Thin installers for Hygon-specific empty-vLLM compatibility hooks."""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)


_patches_applied = False


def _ensure_vit_flash_attn_func():
    """Install the ViT FlashAttention callable in the current Hygon worker."""
    import vllm.v1.attention.backends.fa_utils as fa_utils

    flash_attn_varlen_func = getattr(
        fa_utils,
        "flash_attn_varlen_func",
        None,
    )
    if flash_attn_varlen_func is not None:
        return flash_attn_varlen_func

    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        if flash_attn_varlen_func is None:
            raise ImportError(
                "vLLM's bundled FlashAttention is unavailable"
            )

        logger.info("Using vllm.vllm_flash_attn for Hygon ViT attention")
    except (ImportError, ModuleNotFoundError):
        from flash_attn import flash_attn_varlen_func

        logger.info("Using flash_attn for Hygon ViT attention")

    fa_utils.flash_attn_varlen_func = flash_attn_varlen_func
    return flash_attn_varlen_func


def patch_mm_encoder_attention() -> None:
    """Route Hygon OOT FlashAttention calls through the FA implementation."""
    import vllm.model_executor.layers.attention.mm_encoder_attention as mm_mod

    current_forward_native = mm_mod.MMEncoderAttention.forward_native
    if getattr(
        current_forward_native,
        "_vllm_fl_hygon_mm_encoder_patch",
        False,
    ):
        return

    original_forward_native = current_forward_native

    @functools.wraps(original_forward_native)
    def forward_native_hygon(
        self,
        query,
        key,
        value,
        cu_seqlens=None,
        max_seqlen=None,
        sequence_lengths=None,
    ):
        if self.is_flash_attn_backend:
            _ensure_vit_flash_attn_func()
            return self._forward_fa(
                query,
                key,
                value,
                cu_seqlens,
                max_seqlen,
            )

        return original_forward_native(
            self,
            query,
            key,
            value,
            cu_seqlens,
            max_seqlen,
            sequence_lengths,
        )

    forward_native_hygon._vllm_fl_hygon_mm_encoder_patch = True
    mm_mod.MMEncoderAttention.forward_native = forward_native_hygon
    logger.info(
        "Hygon MMEncoderAttention patched: "
        "forward_native routes FLASH_ATTN to _forward_fa"
    )


def patch_empty_moe() -> None:
    """Route empty-vLLM MoE hooks through the shared FL dispatch layer."""
    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_module

    if getattr(
        fused_moe_module,
        "_vllm_fl_hygon_empty_moe_patched",
        False,
    ):
        return

    from vllm_fl.dispatch import CachedOp
    from vllm_fl.ops.fused_moe.activation import (
        apply_moe_activation as apply_moe_activation_fl,
    )

    fused_moe_module.moe_align_block_size = CachedOp(
        "moe_align_block_size"
    )
    fused_moe_module.apply_moe_activation = apply_moe_activation_fl

    # Hygon has CUDA-shaped capabilities but cannot execute NVIDIA Marlin.
    # Force compressed-tensors WNA16 MoE onto the generic Triton path.
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        compressed_tensors_moe as ct_moe,
    )

    current_check = ct_moe.check_moe_marlin_supports_layer
    if not getattr(current_check, "_vllm_fl_hygon_patched", False):

        @functools.wraps(current_check)
        def check_moe_marlin_supports_layer_hygon(layer, group_size):
            from vllm.platforms import current_platform

            vendor = getattr(current_platform, "vendor_name", "").lower()
            if vendor == "hygon":
                return False
            return current_check(layer, group_size)

        check_moe_marlin_supports_layer_hygon._vllm_fl_hygon_patched = True
        ct_moe.check_moe_marlin_supports_layer = (
            check_moe_marlin_supports_layer_hygon
        )

    fused_moe_module._vllm_fl_hygon_empty_moe_patched = True


def apply_hygon_moe_patches() -> None:
    """Install the Hygon MoE compatibility hook before model construction."""
    from .hygon import HygonBackend

    if not HygonBackend().is_available():
        return

    patch_empty_moe()


def apply_hygon_patches() -> None:
    """Install Hygon-specific compatibility patches once."""
    global _patches_applied

    if _patches_applied:
        return

    from .hygon import HygonBackend

    if not HygonBackend().is_available():
        return

    patch_mm_encoder_attention()
    patch_empty_moe()
    _patches_applied = True


__all__ = [
    "patch_empty_moe",
    "apply_hygon_moe_patches",
    "apply_hygon_patches",
    "patch_mm_encoder_attention",
]
