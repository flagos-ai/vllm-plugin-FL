# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagGems-backed quantization operator implementations."""

from __future__ import annotations

import torch


def dynamic_per_token_quant_int8_flaggems_vllm(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FlagGems-vLLM dynamic symmetric per-token INT8 quantization.

    ``scaled_int8_quant`` selects its dynamic path when ``scale`` is ``None``.
    Import and runtime failures propagate to dispatch, which then tries the
    local FlagGems/Triton implementation before the PyTorch reference.
    """
    from flaggems_vllm.ops.scaled_int8_quant import scaled_int8_quant

    output, scale, _ = scaled_int8_quant(
        x,
        scale=None,
        azp=None,
        symmetric=True,
    )
    return output, scale


def dynamic_per_token_quant_int8_flaggems_triton(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the in-tree backend-neutral FlagGems/Triton fallback kernel."""
    from vllm_fl.quantization.w8a8.flaggems_kernels import (
        dynamic_per_token_quant_int8,
    )

    return dynamic_per_token_quant_int8(x)
