# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import inspect
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Resolve the optional internal entry point before Dynamo tracing. An unknown
# FlagGems ABI uses the reference implementation rather than breaking startup.
try:
    from flaggems_vllm.ops.silu_and_mul import silu_and_mul_kernel as _compiled_silu

    if len(inspect.signature(_compiled_silu).parameters) != 2:
        raise ValueError("expected two input tensors")
except (ImportError, AttributeError, TypeError, ValueError) as error:
    logger.warning("GCU compiled SiLU ABI unavailable; using Torch SiLU: %s", error)
    _compiled_silu = None


def silu_and_mul_gcu(obj, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    if torch.compiler.is_compiling() and not torch.is_grad_enabled():
        if _compiled_silu is not None:
            return _compiled_silu(x1, x2)
        return F.silu(x1) * x2
    from flaggems_vllm import silu_and_mul

    return silu_and_mul(x1, x2)
