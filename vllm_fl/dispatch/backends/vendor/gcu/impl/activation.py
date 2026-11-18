# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import inspect
import logging
from dataclasses import replace

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _gcu_pointwise_kernel(kernel):
    # The provider's NVIDIA fallback permits 65536 CTAs; GCU permits 65535.
    # A private config preserves other kernels, and the generated grid-stride
    # loop still covers all tiles when the launch grid is capped.
    config = replace(
        kernel.config,
        max_grid_size=tuple(min(limit, 65535) for limit in kernel.config.max_grid_size),
        # GCU block-pointer grid-stride loads fault on large strided MLP inputs.
        # Masked pointer loads retain the same tiling and cover partial tiles.
        prefer_block_pointer=False,
    )
    return type(kernel)(kernel.fx, kernel._scalar_fn, config=config)


# Resolve the optional internal entry point before Dynamo tracing. An unknown
# FlagGems ABI uses the reference implementation rather than breaking startup.
try:
    from flaggems_vllm.ops.silu_and_mul import silu_and_mul_kernel as _compiled_silu

    if len(inspect.signature(_compiled_silu).parameters) != 2:
        raise ValueError("expected two input tensors")
    _compiled_silu = _gcu_pointwise_kernel(_compiled_silu)
except (ImportError, AttributeError, TypeError, ValueError) as error:
    logger.warning("GCU compiled SiLU ABI unavailable; using Torch SiLU: %s", error)
    _compiled_silu = None


@torch.library.custom_op("vllm_fl::gcu_silu_and_mul", mutates_args=())
def _silu_and_mul_op(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return _compiled_silu(x1, x2)


@_silu_and_mul_op.register_fake
def _silu_and_mul_fake(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x1)


def silu_and_mul_gcu(obj, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    if torch.compiler.is_compiling() and not torch.is_grad_enabled():
        if _compiled_silu is not None:
            return _silu_and_mul_op(x1, x2)
        return F.silu(x1) * x2
    from flaggems_vllm import silu_and_mul

    return silu_and_mul(x1, x2)
