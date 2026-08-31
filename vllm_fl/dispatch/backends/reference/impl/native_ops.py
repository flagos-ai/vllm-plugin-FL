# Copyright (c) 2026 BAAI. All rights reserved.

"""PyTorch reference implementations for vLLM native out ABIs."""

from __future__ import annotations

import torch


def silu_and_mul_out_torch(
    output: torch.Tensor,
    input: torch.Tensor,
) -> None:
    """Write SiLU-and-mul directly into the supplied output tensor."""
    hidden_size = input.shape[-1] // 2
    gate = input[..., :hidden_size]
    up = input[..., hidden_size:]
    torch.sigmoid(gate, out=output)
    output.mul_(gate)
    output.mul_(up)


__all__ = ["silu_and_mul_out_torch"]
