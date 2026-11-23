# Copyright (c) 2026 BAAI. All rights reserved.
"""Make both rms_norm operands acceptable to the evas dual-die row kernel.

The dispatch op is repaired here, not in the caller, so
``vllm_fl/ops/layernorm.py`` stays byte-identical to ``upstream/main``.  Two
defects: a row slice of the persistent ``inputs_embeds`` buffer inherits its
parent's double-die descriptor, so ``unzip`` hands the kernel the parent's
halves (rebuilt behind a host-only probe); and the row kernel takes only
``[M, N]``, so the Qwen3 ``[T, H, D]`` q/k norms flatten, call, and reshape back
(a row-split flatten is repaired first).

"""

from __future__ import annotations

import torch

_PATCHED_ATTR = "_lpu_rms_norm_operands_patched"


def _flatten_for_row_kernel(x, weight):
    """Collapse leading dims when the norm is over the single trailing dim.

    Returns ``(x, None)`` when the shape is already kernel-shaped, so the
    caller can tell "reshaped" from "untouched" without a second shape test.
    """
    if (
        x.dim() > 2
        and weight is not None
        and weight.dim() == 1
        and weight.shape[0] == x.shape[-1]
    ):
        flat = x.reshape(-1, x.shape[-1])
        if flat.split_dim() == 0:
            from vllm_fl.dispatch.backends.vendor.evas.impl.attention.dual_die_view import (
                repair_view,
            )

            flat = repair_view(flat, 0)
        return flat, x.shape
    return x, None


def _needs_repair(tensor) -> bool:
    """True when the storage is wider than the logical extent it addresses.

    A view of a bigger buffer reports the parent's storage size while ``numel``
    counts only its own elements; an ordinary allocation reports exactly its
    own extent.  This is the host-only form of the staleness test: it does not
    ask the device how the descriptor splits, it only asks whether the operand
    can possibly have inherited one.  Do not add ``is_contiguous()`` here -- on
    this platform a row slice of a dual-die buffer reports non-contiguous
    (stride ``(8, 1)`` yet ``False``), i.e. the very case that needs repair is
    the one that guard would skip.
    """
    if not isinstance(tensor, torch.Tensor) or tensor.dim() == 0:
        return False
    if tensor.device.type != "lpu":
        return False
    if tensor.numel() == 0:
        return False
    return tensor.untyped_storage().size() // tensor.element_size() != tensor.numel()


def _repair(tensor):
    if tensor is None or not _needs_repair(tensor):
        return tensor
    from vllm_fl.dispatch.backends.vendor.evas.impl.attention.dual_die_view import (
        repair_view,
    )

    return repair_view(tensor, 0)


def _wrap_rms_norm(upstream):
    """Wrap the dispatch op so its operands are kernel-shaped and faithful."""

    def rms_norm(obj, x, residual=None):
        if not torch.lpu.is_double_die_enabled():
            return upstream(obj, x, residual)
        orig_shape = None
        if residual is None:
            x, orig_shape = _flatten_for_row_kernel(x, getattr(obj, "weight", None))
        out = upstream(obj, _repair(x), _repair(residual))
        if orig_shape is None:
            return out
        return out.reshape(orig_shape)

    return rms_norm


def apply_lpu_rms_norm_operands_patch() -> None:
    """Wrap ``vllm_fl.ops.layernorm._rms_norm`` on LPU."""
    import vllm_fl.ops.layernorm as layernorm

    if getattr(layernorm, _PATCHED_ATTR, False):
        return
    layernorm._rms_norm = _wrap_rms_norm(layernorm._rms_norm)
    setattr(layernorm, _PATCHED_ATTR, True)
