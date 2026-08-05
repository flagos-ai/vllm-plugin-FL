# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagGems-backed quantization operator implementations."""

from __future__ import annotations

import torch


def dynamic_per_token_quant_int8_flaggems(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run vLLM-compatible activation quantization under FlagGems dispatch.

    FlagGems does not expose a standalone public INT8 per-token operator.
    Keeping the vLLM dynamic-scaled-INT8 decomposition here preserves the
    Dense linear numerical contract without importing FlagGems' private MoE
    helper. The worker enables FlagGems ATen replacements globally.
    """
    import flag_gems  # noqa: F401

    from vllm_fl.quantization.w8a8.reference import (
        dynamic_per_token_quant_int8,
    )

    return dynamic_per_token_quant_int8(x)
