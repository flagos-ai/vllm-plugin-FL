# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagGems-backed quantization operator implementations."""

from __future__ import annotations

import torch


def dynamic_per_token_quant_int8_flaggems(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the FlagGems-vLLM fused dynamic per-token INT8 kernel.

    FlagGems core does not expose this standalone operator, so resolve the
    vLLM integration entry point lazily. Import and runtime failures propagate
    to ``CachedOp``/``OpManager``, which marks this implementation unavailable
    and falls through to the registered ``reference.torch`` implementation.
    """
    from flaggems_vllm.ops.scaled_int8_quant import (
        dynamic_per_token_quant_int8,
    )

    return dynamic_per_token_quant_int8(x)
