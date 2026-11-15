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
from contextvars import ContextVar
from functools import wraps

import torch

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
    # Also force HAS_TRITON = False so the conditional branch in
    # fast_pos_embed_interpolate never selects the Triton path, even
    # if _bilinear_pos_embed_kernel is called via other routes (e.g.
    # CUDA graph capture warmup that runs before the name-swap above
    # takes effect for a particular call site).
    setattr(qwen3_vl_mod, "HAS_TRITON", False)
    setattr(qwen3_vl_mod, _PATCH_FLAG, True)
    logger.info(
        "kunlunxin vl_compat: triton_pos_embed_interpolate → "
        "pos_embed_interpolate_native, HAS_TRITON → False "
        "(FlagTree miscompiles bilinear interpolation kernel, "
        "producing wrong pos-embeds for image inputs)"
    )


_vision_tp_scope = ContextVar("kunlunxin_vision_tp_scope", default=False)


def apply_vision_tp_scope() -> None:
    """Limit ordered reduction to the Qwen MoE weight-sharded image encoder."""
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration

    model = Qwen3_5MoeForConditionalGeneration
    original = model._process_image_input
    if getattr(original, "_kunlunxin_vision_tp_scope", False):
        return

    @wraps(original)
    def scoped(self, image_input):
        if self.use_data_parallel:
            return original(self, image_input)
        token = _vision_tp_scope.set(True)
        try:
            return original(self, image_input)
        finally:
            _vision_tp_scope.reset(token)

    scoped._kunlunxin_vision_tp_scope = True
    model._process_image_input = scoped


def maybe_reduce_vision_tp(communicator, tensor: torch.Tensor) -> torch.Tensor | None:
    """Reduce eager BF16 vision partials in logical rank order in FP32."""
    if not _vision_tp_scope.get():
        return None
    flagcx = communicator.pyflagcx_comm
    if (
        communicator.world_size != 4
        or not communicator.unique_name.startswith("tp:")
        or flagcx is None
        or flagcx.disabled
        or tensor.dtype != torch.bfloat16
        or not tensor.ndim
        or not tensor.numel()
    ):
        return None

    # The native sum changes the vision embedding when the same cards are
    # enumerated in reverse order. Gather BF16 partials and round only once.
    partials = communicator.all_gather(tensor, dim=0).reshape(
        (communicator.world_size, *tensor.shape)
    )
    total = partials[0].float()
    for partial in partials[1:]:
        total.add_(partial.float())
    return total.to(dtype=tensor.dtype)
