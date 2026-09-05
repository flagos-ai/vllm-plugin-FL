# Copyright (c) 2026 BAAI. All rights reserved.

import logging

from .impl.bilinear_pos_embed import apply_bilinear_pos_embed_gcu_patch
from .impl.chunk_delta_h import apply_chunk_delta_h_gcu_patch
from .impl.fused_recurrent_packed_decode import (
    apply_fused_recurrent_packed_decode_gcu_patch,
)
from .impl.slot_mapping import apply_slot_mapping_gcu_patch
from .impl.flash_attn_backend import apply_flash_attn_backend_gcu_patch

logger = logging.getLogger(__name__)
_patches_applied = False


def _patch_rotary_flash_attn_import():
    """Guard vllm 0.20.2's ungated flash_attn.ops.triton.rotary import.

    vllm 0.20.2 ApplyRotaryEmb.__init__ imports flash_attn.ops.triton.rotary
    whenever flash_attn is installed, with no platform/compiler gate. That
    module hard-imports triton_gcu.triton, which only the vendor triton side
    (/opt/triton) provides — so a pure flagtree stack (/opt/flagtree) that
    has flash_attn installed dies on ModuleNotFoundError at model build.

    Applied here (GCU backend load, i.e. PlatformFL.pre_register_and_update —
    after current_platform is resolved) rather than at plugin register():
    importing vllm.model_executor.layers.rotary_embedding.common during
    platform-plugin registration resolves current_platform before the FL
    platform is installed, caching UnspecifiedPlatform and breaking device
    inference ("Failed to infer device type"). On failure the guard leaves
    apply_rotary_emb_flash_attn None and rotary falls back to forward_native,
    matching vllm 0.24.0 upstream.
    """
    import contextlib
    from importlib import import_module
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    if getattr(ApplyRotaryEmb, "_fl_rotary_import_guarded", False):
        return

    def _guarded_init(self, enforce_enable=False, is_neox_style=True,
                      enable_fp32_compute=False):
        super(ApplyRotaryEmb, self).__init__(enforce_enable=enforce_enable)
        self.is_neox_style = is_neox_style
        self.enable_fp32_compute = enable_fp32_compute
        self.apply_rotary_emb_flash_attn = None
        with contextlib.suppress(ModuleNotFoundError):
            self.apply_rotary_emb_flash_attn = import_module(
                "flash_attn.ops.triton.rotary").apply_rotary

    ApplyRotaryEmb.__init__ = _guarded_init
    ApplyRotaryEmb._fl_rotary_import_guarded = True


def apply_gcu_patches() -> None:
    """Apply all GCU-specific kernel / model monkey-patches."""
    global _patches_applied
    if _patches_applied:
        return

    apply_bilinear_pos_embed_gcu_patch()
    apply_chunk_delta_h_gcu_patch()
    apply_fused_recurrent_packed_decode_gcu_patch()
    apply_slot_mapping_gcu_patch()
    apply_flash_attn_backend_gcu_patch()
    _patch_rotary_flash_attn_import()
    _patches_applied = True


def apply_op_kernel_patches() -> None:
    """Alias kept for callers that use the older name."""
    apply_gcu_patches()
