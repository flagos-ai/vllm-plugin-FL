# Copyright (c) 2026 BAAI. All rights reserved.
"""Keep vLLM's gated RMSNorm out of Inductor fusion on GCU.

``Qwen3.5/3.6`` GDN layers use ``vllm.model_executor.layers.layernorm.
RMSNormGated``. On GCU platforms the ``CustomOp`` dispatches to
``forward_native``, which Dynamo traces into, so Inductor fuses

    sum(x*x) -> mean -> rsqrt -> * weight -> * silu(z)

into one Triton kernel. The GCU300 compiler cannot allocate registers for that
fused kernel and aborts the whole graph compilation with

    error: ran out of registers during register allocation

The tile was already minimal (``XBLOCK=32``, ``num_warps=2``, ``R0_BLOCK=128``),
so this is register-file pressure in the fused epilogue, not tile sizing.

Registering the op as an opaque ``torch.library`` custom op keeps the math
bit-identical to eager (it just calls ``RMSNormGated.forward_static`` at
runtime) while stopping Inductor from fusing it into the surrounding graph.
``torch._dynamo.disable`` is *not* usable here: vLLM compiles the model with
``fullgraph=True``, where a graph break raises
``Unsupported: Skip calling torch.compiler.disable()d function``.

Disable with ``VLLM_FL_GCU_BREAK_NORM_FUSION=0``.
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_OP_NAME = "vllm_fl::rms_norm_gated"
_INSTALL_STATE = {"installed": False}


def _register_op() -> None:
    from vllm.model_executor.layers.layernorm import RMSNormGated

    @torch.library.custom_op(_OP_NAME, mutates_args=())
    def _rms_norm_gated(
        x: torch.Tensor,
        z: Optional[torch.Tensor],
        weight: torch.Tensor,
        epsilon: float,
        orig_dtype: torch.dtype,
        group_size: Optional[int],
        norm_before_gate: bool,
        activation: str,
    ) -> torch.Tensor:
        return RMSNormGated.forward_static(
            x,
            z,
            weight,
            epsilon,
            orig_dtype,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            activation=activation,
        )

    @_rms_norm_gated.register_fake
    def _(x, z, weight, epsilon, orig_dtype, group_size, norm_before_gate, activation):
        return torch.empty_like(x)

    def forward_native(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        return torch.ops.vllm_fl.rms_norm_gated(
            x,
            z,
            self.weight,
            self.eps,
            x.dtype,
            self.group_size,
            self.norm_before_gate,
            self.activation,
        )

    forward_native._fl_gcu_opaque_norm_gated = True
    RMSNormGated.forward_native = forward_native


def install_rms_norm_gated_opaque() -> None:
    """Install the opaque gated-RMSNorm wrapper (idempotent, never fatal)."""
    if _INSTALL_STATE["installed"]:
        return
    # GCU400/L600 does not hit the GCU300 register-allocation failure and keeps
    # Inductor's fused gated-RMSNorm (the L600-verified path).  Only the
    # gcu300 family (S60) needs the opaque op by default.
    #   VLLM_FL_GCU_BREAK_NORM_FUSION=1 -> force on (any GCU)
    #   VLLM_FL_GCU_BREAK_NORM_FUSION=0 -> force off
    mode = os.environ.get("VLLM_FL_GCU_BREAK_NORM_FUSION")
    if mode is not None and mode.lower() in ("0", "false", "no"):
        return
    if not (mode and mode.lower() in ("1", "true", "yes")):
        from .device_arch import is_gcu300

        if not is_gcu300():
            return
    try:
        _register_op()
        _INSTALL_STATE["installed"] = True
        logger.info(
            "[FL] gated RMSNorm is now an opaque custom op "
            "(GCU300 inductor register-allocation limit)"
        )
    except Exception:  # pragma: no cover - never block startup
        logger.warning(
            "[FL] failed to register the opaque gated RMSNorm op", exc_info=True
        )
