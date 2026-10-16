# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM custom-op boundary and model threshold for FlagGems-vllm PLE."""

import torch
from flaggems_vllm import ple_gate_norm_ as qwen3_8_flash_next_ple_gate_norm_
from flaggems_vllm import ple_prefill_short_conv_ as _ple_prefill_short_conv_
from flaggems_vllm.ops.qwen4.ple_fusion import (
    can_use_ple_gate_norm_triton,
    can_use_ple_prefill_conv_triton,
)
from vllm.utils.torch_utils import direct_register_custom_op

PLE_FUSION_MIN_TOKENS = 1024

def qwen3_8_flash_next_ple_gate_norm_fake(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_weight: torch.Tensor,
    query_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    gated: torch.Tensor,
    normalized: torch.Tensor,
    hc_count: int,
    eps: float,
) -> None:
    return None



def ple_gate_norm(
    key: torch.Tensor,
    query: torch.Tensor,
    value: torch.Tensor,
    key_weight: torch.Tensor,
    query_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    hc_count: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the two required results and launch the fused custom op."""

    gated = torch.empty_like(key)
    normalized = torch.empty_like(key)
    torch.ops.vllm.qwen3_8_flash_next_ple_gate_norm_(
        key,
        query,
        value,
        key_weight,
        query_weight,
        conv_weight,
        gated,
        normalized,
        hc_count,
        eps,
    )
    return gated, normalized


def ple_prefill_short_conv_(
    x: torch.Tensor,
    output: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    num_prefills: int,
    max_len: int,
    state_len: int,
    kernel_width: int,
    dilation: int,
    null_block_id: int,
    min_tokens: int = 0,
    block_t: int | None = None,
    block_c: int | None = None,
) -> bool:
    """Write PLE prefill conv output/state directly; return whether fused."""

    if not can_use_ple_prefill_conv_triton(
        x,
        output,
        conv_state,
        conv_weight,
        query_start_loc,
        state_indices,
        has_initial_state,
        num_prefills=num_prefills,
        max_len=max_len,
        state_len=state_len,
        kernel_width=kernel_width,
        dilation=dilation,
        min_tokens=min_tokens,
    ):
        return False
    return _ple_prefill_short_conv_(
        x, output, conv_state, conv_weight, query_start_loc, state_indices, has_initial_state,
        num_prefills=num_prefills, max_len=max_len, state_len=state_len,
        kernel_width=kernel_width, dilation=dilation, null_block_id=null_block_id,
        min_tokens=min_tokens, block_t=block_t, block_c=block_c,
    )

direct_register_custom_op(
    op_name="qwen3_8_flash_next_ple_gate_norm_",
    op_func=qwen3_8_flash_next_ple_gate_norm_,
    mutates_args=["gated", "normalized"],
    fake_impl=qwen3_8_flash_next_ple_gate_norm_fake,
)


__all__ = [
    "PLE_FUSION_MIN_TOKENS",
    "can_use_ple_gate_norm_triton",
    "can_use_ple_prefill_conv_triton",
    "ple_gate_norm",
    "ple_prefill_short_conv_",
]
