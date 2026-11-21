# Copyright (c) 2026 BAAI. All rights reserved.
"""Route unquantized SwiGLU MLP (gate_up_proj + SiluAndMul) through FlagOS
Fused Dual-Accumulator SwiGLU custom operator.

Registered in FlagOS OpRegistry under ``fused_swiglu_proj``:
  * ``default.flagos`` -> ``flag_gems.fused.fused_swiglu_proj`` (microarchitecture-aware
    dual-accumulator Triton kernel; eliminates [M, 2*I] HBM round-trip).
  * ``reference.torch`` -> PyTorch ``F.linear`` + ``F.silu`` * up fallback.

Wrapped in ``@torch.library.custom_op("vllm_fl::fused_swiglu_proj")`` so ``torch.compile``
and CUDA Graph capture trace it with zero graph breaks.
"""

from __future__ import annotations

import functools
import logging
import os

import torch
import torch.nn.functional as F

from vllm_fl.dispatch import CachedOp
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

logger = logging.getLogger(__name__)

_MODE = os.environ.get("VLLM_FL_FUSED_SWIGLU", "auto").strip().lower()


def _flagos_swiglu_available() -> bool:
    try:
        from flag_gems.fused import fused_swiglu_proj  # noqa: F401

        return True
    except ImportError:
        return False


def _torch_available() -> bool:
    return True


def _bind(fn, avail):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = avail
    return wrapper


def fused_swiglu_proj_flagos(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from flag_gems.fused.fused_swiglu_proj import fused_swiglu_proj

    return fused_swiglu_proj(x, weight)


def fused_swiglu_proj_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gu = F.linear(x, weight)
    d = gu.shape[-1] // 2
    return F.silu(gu[..., :d]) * gu[..., d:]


def register_fused_swiglu_proj(registry) -> None:
    registry.register_many(
        [
            OpImpl(
                op_name="fused_swiglu_proj",
                impl_id="default.flagos",
                kind=BackendImplKind.DEFAULT,
                fn=_bind(fused_swiglu_proj_flagos, _flagos_swiglu_available),
                priority=BackendPriority.DEFAULT,
            ),
            OpImpl(
                op_name="fused_swiglu_proj",
                impl_id="reference.torch",
                kind=BackendImplKind.REFERENCE,
                fn=_bind(fused_swiglu_proj_reference, _torch_available),
                priority=BackendPriority.REFERENCE,
            ),
        ]
    )


_fused_swiglu_op = CachedOp("fused_swiglu_proj")


@torch.library.custom_op("vllm_fl::fused_swiglu_proj", mutates_args=())
def _fused_swiglu_custom(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _fused_swiglu_op(x, weight)


@_fused_swiglu_custom.register_fake
def _fused_swiglu_custom_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0] // 2))


def patch_vllm_fused_swiglu_mlp() -> bool:
    if _MODE in ("0", "false", "no", "off"):
        return False
    if _MODE == "auto":
        try:
            from vllm_fl.platform import PlatformFL

            # Dual-accumulator fusion wins across all M on >=128 KiB SRAM architectures (e.g. Iluvatar BI-V150);
            # on 64 KiB SRAM architectures (MetaX C500), 128-wide cpasync linear + vectorized SiluAndMul avoids tile halving.
            if PlatformFL.vendor_name != "iluvatar":
                return False
        except Exception:  # noqa: BLE001
            return False
    try:
        from vllm.model_executor.layers.activation import SiluAndMul
        from vllm.model_executor.layers.linear import (
            MergedColumnParallelLinear,
            UnquantizedLinearMethod,
        )
        from vllm.model_executor.models.llama import LlamaMLP
    except Exception as exc:  # noqa: BLE001
        logger.debug("LlamaMLP unavailable for fused SwiGLU patch: %s", exc)
        return False

    if getattr(LlamaMLP, "_vllm_fl_fused_swiglu_patched", False):
        return True

    orig_init = LlamaMLP.__init__
    orig_forward = LlamaMLP.forward

    @functools.wraps(orig_init)
    def _fused_llama_mlp_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        gu_proj = getattr(self, "gate_up_proj", None)
        ok = (
            type(gu_proj) is MergedColumnParallelLinear
            and getattr(gu_proj, "tp_size", 1) == 1
            and type(getattr(gu_proj, "quant_method", None)) is UnquantizedLinearMethod
            and getattr(gu_proj, "bias", None) is None
            and isinstance(getattr(gu_proj, "weight", None), torch.Tensor)
            and gu_proj.weight.ndim == 2
            and gu_proj.weight.shape[0] % 2 == 0
            and isinstance(getattr(self, "act_fn", None), SiluAndMul)
        )
        self._vllm_fl_swiglu_ok = bool(ok)
        if ok:
            _fused_swiglu_op.pre_resolve()

    def _fused_llama_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._vllm_fl_swiglu_ok
            and type(self.gate_up_proj) is MergedColumnParallelLinear
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
            and self.gate_up_proj.weight.dtype == x.dtype
        ):
            act = torch.ops.vllm_fl.fused_swiglu_proj(x, self.gate_up_proj.weight)
            out, _ = self.down_proj(act)
            return out
        return orig_forward(self, x)

    LlamaMLP.__init__ = _fused_llama_mlp_init
    LlamaMLP.forward = _fused_llama_mlp_forward
    LlamaMLP._vllm_fl_fused_swiglu_patched = True
    return True
