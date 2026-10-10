# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM custom-op boundaries for FlagGems-vllm HyperConnection kernels."""

import torch
from flaggems_vllm import (
    qwen4_grouped_gemma_rmsnorm,
    qwen4_hc_gate_reduce,
    qwen4_hc_inject_combine,
    qwen4_hc_combine_norm,
)
from flaggems_vllm.ops.qwen4.hyperconnection import (
    can_use_hc_triton,
    can_use_hc_inject_triton,
    can_use_hc_combine_norm_triton,
)
from vllm.utils.torch_utils import direct_register_custom_op

def qwen4_grouped_gemma_rmsnorm_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    hc_count: int,
    eps: float,
) -> torch.Tensor:
    del weight, hc_count, eps
    return torch.empty_like(hidden_states)



def qwen4_hc_gate_reduce_fake(
    logits: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del normed
    return torch.empty(
        (*logits.shape[:-1], logits.shape[-1] // hc_count),
        dtype=logits.dtype,
        device=logits.device,
    )



def qwen4_hc_inject_combine_fake(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del injection_logits, block_output, hc_count
    return torch.empty_like(residual)



def qwen4_hc_combine_norm_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_output, injection_logits, norm_weight, eps, hc_count
    return torch.empty_like(residual), torch.empty_like(residual)


direct_register_custom_op(
    op_name="qwen4_grouped_gemma_rmsnorm",
    op_func=qwen4_grouped_gemma_rmsnorm,
    mutates_args=[],
    fake_impl=qwen4_grouped_gemma_rmsnorm_fake,
)
direct_register_custom_op(
    op_name="qwen4_hc_gate_reduce",
    op_func=qwen4_hc_gate_reduce,
    mutates_args=[],
    fake_impl=qwen4_hc_gate_reduce_fake,
)
direct_register_custom_op(
    op_name="qwen4_hc_inject_combine",
    op_func=qwen4_hc_inject_combine,
    mutates_args=[],
    fake_impl=qwen4_hc_inject_combine_fake,
)
direct_register_custom_op(
    op_name="qwen4_hc_combine_norm",
    op_func=qwen4_hc_combine_norm,
    mutates_args=[],
    fake_impl=qwen4_hc_combine_norm_fake,
)


__all__ = [
    "can_use_hc_combine_norm_triton",
    "can_use_hc_inject_triton",
    "can_use_hc_triton",
    "qwen4_grouped_gemma_rmsnorm",
    "qwen4_hc_combine_norm",
    "qwen4_hc_gate_reduce",
    "qwen4_hc_inject_combine",
]
