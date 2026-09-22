# Copyright (c) 2026 BAAI. All rights reserved.
"""Fused Triton RMSNorm for GCU (plain ``RMSNorm`` and ``GemmaRMSNorm``).

The previous vendor impl was the decomposed torch path
(``pow -> mean -> rsqrt -> mul -> mul``), i.e. 4-6 aten launches per norm, and
``GemmaRMSNorm`` was not routed through the dispatch layer at all (vLLM looks
OOT implementations up by the *exact* class name, see
``CustomOp.__new__`` -> ``op_registry_oot[cls.__name__]``), so every Gemma-style
norm ran the torch decomposition.  Both now resolve to **one** Triton launch.

Semantics
---------
* ``RMSNorm``: ``vllm.model_executor.layers.layernorm.RMSNorm.forward_static``
  (f32 accumulate, residual stored in the original dtype, optional weight).
* ``GemmaRMSNorm``: ``forward_native`` -- weight is ``(1 + w).float()`` and the
  whole math stays in f32 before the final cast.

Both differ from upstream by at most 1 bf16 ULP because the weight multiply is
done in f32 instead of "cast then multiply".

GCU300 caveats
--------------
* ``tl.arange``-derived load/store masks may only use ``<`` / ``~(a < b)``.
* fixed grid + contiguous row chunks (``NUM_SPC`` is a runtime arg so a new
  batch size never triggers a fresh JIT compilation).
* past ~1M elements the vectorized torch path wins, so we delegate upward
  (measured crossover on GCU300, hidden=2048: fused 10-17x at <=8 rows,
  1.78x at 512 rows, negative from 1024 rows).
"""

from __future__ import annotations

import importlib
import logging
from typing import Optional, Union

import torch

from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

GCU_NUM_GRID = 48
GCU_MAX_WARPS = 4
_MAX_BLOCK = 8192
_FUSED_MAX_ELEMS = 1 << 20  # ~1M elements


@triton.jit(do_not_specialize=["num_rows", "row_stride"])
def _rms_norm_fwd_kernel_gcu(
    x_ptr,
    w_ptr,
    y_ptr,
    res_ptr,
    num_rows,
    row_stride,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_W: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL: tl.constexpr,
    GEMMA: tl.constexpr,
    NUM_SPC: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < N  # single '<' comparison -> safe on GCU300
    per = (num_rows + NUM_SPC - 1) // NUM_SPC
    row_start = pid * per
    row_end = tl.minimum(row_start + per, num_rows)
    for row in range(row_start, row_end):
        base = row * row_stride
        x = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
        if ADD_RESIDUAL:
            r = tl.load(res_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
            x = x + r
            if STORE_RESIDUAL:
                tl.store(res_ptr + base + offs, x.to(res_ptr.dtype.element_ty), mask=m)
        var = tl.sum(x * x, axis=0) / N
        x = x * tl.rsqrt(var + eps)
        if HAS_W:
            w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
            if GEMMA:
                w = w + 1.0  # GemmaRMSNorm: x * (1 + w)
            x = x * w
        tl.store(y_ptr + base + offs, x.to(y_ptr.dtype.element_ty), mask=m)


def _fused_supported(x: torch.Tensor) -> bool:
    """True when the fused kernel is the right choice for this tensor."""
    if x.device.type not in ("gcu", "cuda") or x.numel() == 0:
        return False
    n = x.shape[-1]
    if n > _MAX_BLOCK:
        return False
    return x.numel() <= _FUSED_MAX_ELEMS


def _launch(x, weight, eps, residual=None, gemma=False):
    """One Triton launch; returns (out, residual_or_None)."""
    n = x.shape[-1]
    out = torch.empty_like(x)
    x2 = x if x.is_contiguous() else x.contiguous()
    y2 = out if out.is_contiguous() else out.contiguous()
    rows = x2.numel() // n
    x2 = x2.view(rows, n)
    y2 = y2.view(rows, n)

    add_res = residual is not None
    res2 = None
    if add_res:
        res2 = residual if residual.is_contiguous() else residual.contiguous()
        res2 = res2.view(rows, n)

    block = triton.next_power_of_2(n)
    nspc = min(max(rows, 1), GCU_NUM_GRID)
    _rms_norm_fwd_kernel_gcu[(nspc,)](
        x2,
        weight if weight is not None else x2,
        y2,
        res2 if res2 is not None else x2,
        rows,
        x2.stride(0),
        float(eps),
        N=n,
        BLOCK=block,
        HAS_W=weight is not None,
        ADD_RESIDUAL=add_res,
        STORE_RESIDUAL=add_res,
        GEMMA=gemma,
        NUM_SPC=nspc,
        num_warps=GCU_MAX_WARPS,
    )
    if y2.data_ptr() != out.data_ptr():
        out.copy_(y2.view_as(out))
    if add_res:
        return out, residual
    return out


def _rms_norm_reference(obj, x, residual, weight):
    """``RMSNorm.forward_static`` semantics, used for the non-fused shapes."""
    orig = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual
        residual = x.to(orig)
    override = getattr(obj, "variance_size_override", None)
    x_var = x if override is None else x[..., :override]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + obj.variance_epsilon)
    x = x.to(orig)
    if weight is not None:
        x = x * weight
    if residual is None:
        return x
    return x, residual


def rms_norm_gcu(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """``RMSNorm`` forward: one fused Triton launch when profitable."""
    weight = obj.weight.data if getattr(obj, "has_weight", True) else None
    fused = getattr(obj, "variance_size_override", None) is None and _fused_supported(x)
    if not fused:
        return _rms_norm_reference(obj, x, residual, weight)
    return _launch(x, weight, obj.variance_epsilon, residual=residual)


def gemma_rms_norm_gcu(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """``GemmaRMSNorm`` forward: one fused Triton launch when profitable."""
    if not _fused_supported(x):
        return _gemma_rms_norm_reference(obj, x, residual)
    return _launch(x, obj.weight, obj.variance_epsilon, residual=residual, gemma=True)


def _gemma_rms_norm_reference(obj, x, residual):
    """``GemmaRMSNorm.forward_native`` semantics (all-f32 math, weight + 1)."""
    orig = x.dtype
    weight = obj.weight.data.float() + 1.0
    if residual is not None:
        x = x.float() + residual.float() if orig == torch.float16 else x + residual
        residual = x
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(variance + obj.variance_epsilon)
    out = (out.to(weight.dtype) * weight).to(orig)
    if residual is None:
        return out
    return out, residual


# ---------------------------------------------------------------------------
# monkey-patch entry (repo convention: impl/<kernel>.py + apply_*_gcu_patch())
# ---------------------------------------------------------------------------
_LN_MODULE = "vllm.model_executor.layers.layernorm"

_patched = False


def _rms_norm_forward_gcu(self, x, residual=None):
    return rms_norm_gcu(self, x, residual)


def _gemma_rms_norm_forward_gcu(self, x, residual=None):
    return gemma_rms_norm_gcu(self, x, residual)


def apply_rms_norm_gcu_patch() -> None:
    """Route ``RMSNorm`` / ``GemmaRMSNorm`` to the fused GCU Triton kernel.

    ``forward_native`` is the effective hook: vLLM resolves OOT ops by the
    *exact* class name, so ``GemmaRMSNorm`` has no ``forward_oot`` and lands on
    ``CustomOp.forward_oot -> self.forward_native``.  ``GemmaRMSNorm.forward_cuda``
    just delegates to ``forward_native``, so it is covered as well.
    """
    global _patched
    if _patched:
        return
    gcu = getattr(torch, "gcu", None)
    if gcu is None or not gcu.is_available():
        return
    try:
        mod = importlib.import_module(_LN_MODULE)
        done = True
        for cls_name, fn in (
            ("RMSNorm", _rms_norm_forward_gcu),
            ("GemmaRMSNorm", _gemma_rms_norm_forward_gcu),
        ):
            cls = getattr(mod, cls_name, None)
            if cls is None:
                done = False  # partially imported module -> retry on next call
                continue
            if cls.__dict__.get("forward_native") is not fn:
                cls.forward_native = fn
        _patched = done
        if _patched:
            logger.info(
                "Patched RMSNorm/GemmaRMSNorm.forward_native -> fused GCU Triton "
                "kernel (1 launch, grid<=%d, fallback above %d elems)",
                GCU_NUM_GRID,
                _FUSED_MAX_ELEMS,
            )
    except Exception as exc:
        logger.debug("rms_norm patch deferred (will retry): %s", exc)


__all__ = ["rms_norm_gcu", "gemma_rms_norm_gcu", "apply_rms_norm_gcu_patch"]
