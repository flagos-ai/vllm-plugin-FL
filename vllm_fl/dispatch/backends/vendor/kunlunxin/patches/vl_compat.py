# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Replace the Triton bilinear pos-embed interpolation with torch on FlagTree.

FlagTree silently miscompiles ``_bilinear_pos_embed_kernel`` for the
Qwen3-VL vision encoder: image tasks receive corrupted positional
embeddings, producing wrong shape/colour names and garbled multilingual
text.  ``pos_embed_interpolate_native`` (already in the upstream module)
produces identical results in pure torch and is used as the replacement.

``Qwen3VLVisionModel.fast_pos_embed_interpolate`` selects the
interpolation function at call-time (not at construction), so patching
``triton_pos_embed_interpolate`` in the module namespace is sufficient to
redirect all existing model instances.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_vl_pos_embed_native"


def apply_vl_compat() -> None:
    """Redirect triton_pos_embed_interpolate → pos_embed_interpolate_native."""
    try:
        import vllm.model_executor.models.qwen3_vl as qwen3_vl_mod
    except ImportError:
        return  # VL model not installed; nothing to patch

    if getattr(qwen3_vl_mod, _PATCH_FLAG, False):
        return  # idempotent

    native_fn = getattr(qwen3_vl_mod, "pos_embed_interpolate_native", None)
    if native_fn is None:
        logger.warning(
            "kunlunxin vl_compat: pos_embed_interpolate_native not found in "
            "qwen3_vl; skipping pos-embed patch"
        )
        return

    triton_fn = getattr(qwen3_vl_mod, "triton_pos_embed_interpolate", None)
    if triton_fn is None:
        # Triton not compiled or already absent; native path is already used
        setattr(qwen3_vl_mod, _PATCH_FLAG, True)
        return

    # Replace the Triton entry point with the native fallback so that
    # fast_pos_embed_interpolate's `triton_pos_embed_interpolate if HAS_TRITON`
    # branch resolves to the torch implementation on every call.
    setattr(qwen3_vl_mod, "triton_pos_embed_interpolate", native_fn)
    setattr(qwen3_vl_mod, _PATCH_FLAG, True)
    logger.info(
        "kunlunxin vl_compat: triton_pos_embed_interpolate → "
        "pos_embed_interpolate_native (FlagTree miscompiles bilinear "
        "interpolation kernel, producing wrong pos-embeds for image inputs)"
    )
