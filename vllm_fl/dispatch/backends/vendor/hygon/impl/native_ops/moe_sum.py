# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon implementation of vLLM's raw ``moe_sum`` output ABI."""

from __future__ import annotations

import torch


def moe_sum_hygon_out(input: torch.Tensor, output: torch.Tensor) -> None:
    """Reduce expert outputs with the validated Hygon Triton kernel."""
    from ..moe.bf16_moe_fusions import try_hygon_fixed_topk8_reduce

    if try_hygon_fixed_topk8_reduce(input, output):
        return

    raise NotImplementedError(
        "vendor:hygon moe_sum supports only the validated fixed-topk=8 "
        "BF16 shape; select flagos or reference for other inputs"
    )


__all__ = ["moe_sum_hygon_out"]
