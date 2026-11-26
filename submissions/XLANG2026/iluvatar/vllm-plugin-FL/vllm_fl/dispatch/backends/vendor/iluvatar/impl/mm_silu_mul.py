# SPDX-License-Identifier: Apache-2.0
"""Fuse MiniCPM gate/up projection and SiLU multiplication on BI-V150."""

import logging
import os

import torch

logger = logging.getLogger(__name__)


@torch.library.custom_op("fl::mm_silu_mul", mutates_args=())
def _mm_silu_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from flag_gems.runtime.backend._iluvatar.fused.mm_silu_mul import mm_silu_mul

    return mm_silu_mul(a, b)


@torch.library.register_fake("fl::mm_silu_mul")
def _mm_silu_mul_fake(a, b):
    return a.new_empty((a.shape[0], b.shape[1] // 2))


_DISABLED = os.getenv("VLLM_FL_DISABLE_MLP_FUSION", "0") == "1"
_applied = False
_warned = False


def _fused_gate_up(mlp, x):
    global _warned
    if _DISABLED or x.ndim != 2:
        return None
    proj = mlp.gate_up_proj
    if getattr(proj, "quant_config", None) is not None:
        return None
    weight = getattr(proj, "weight", None)
    if weight is None or tuple(weight.shape) != (12288, 2048):
        return None
    if x.shape[0] > 128 and x.shape[0] < 1024:
        return None
    if x.shape[1] != 2048 or x.dtype != weight.dtype:
        return None
    if x.dtype not in (torch.bfloat16, torch.float16):
        return None
    try:
        return torch.ops.fl.mm_silu_mul(x, weight.t())
    except Exception as exc:
        if not _warned:
            logger.warning("MLP fusion failed; using stock gate/up path: %s", exc)
            _warned = True
        return None


def _forward(self, x):
    hidden = _fused_gate_up(self, x)
    if hidden is None:
        hidden, _ = self.gate_up_proj(x)
        hidden = self.act_fn(hidden)
    hidden, _ = self.down_proj(hidden)
    return hidden


def apply():
    global _applied
    if _applied:
        return
    from vllm.model_executor.models.llama import LlamaMLP
    from vllm.model_executor.models.minicpm import MiniCPMMLP

    LlamaMLP.forward = _forward
    MiniCPMMLP.forward = _forward
    _applied = True
    logger.info("Iluvatar dual-accumulator MLP fusion enabled")
