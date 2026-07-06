# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 2026 - Modified by Kunlunxin, Inc. All Rights Reserved.

"""
Kunlunxin fused MoE kernel implementations.

In vllm 0.20+, SharedFusedMoE is removed. Instead, this module provides
`fused_experts_impl` which is patched into `vllm_fl.ops.fused_moe.fused_moe`
via the Kunlunxin patch system.
"""

from __future__ import annotations

from typing import Optional

import torch

import xtorch_ops

# for kunlunxin vendor
_KLX_MOE_BLOCK_NUM = 12


def _klx_fused_experts(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    use_int8_w8a8: bool = False,
    use_int8_w4a8: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
) -> None:
    """
    Fused MoE expert computation using xtorch_ops (sorted path).

    Pipeline: gen_block_statistic -> moe_pre_sorted -> moe_fc(w1) -> swiglu -> moe_fc(w2) -> post (weight+sum)
    """
    if use_int8_w8a8 or use_int8_w4a8:
        raise NotImplementedError("_klx_fused_experts is not supported for int8 w8a8 and w4a8.")

    seq_num, hidden_dim = hidden_states.shape
    moe_topk = topk_ids.shape[1]
    expert_num = w1.shape[0]
    double_ffn_hd = w1.shape[1]
    moe_input_num = seq_num * moe_topk

    device = hidden_states.device
    dtype = hidden_states.dtype

    # Step 1: Generate block statistics
    block_statistic = torch.zeros(
        _KLX_MOE_BLOCK_NUM, expert_num, dtype=torch.int32, device=device,
    )
    xtorch_ops.gen_block_statistic(topk_ids, block_statistic)

    # Step 2: Sort tokens by expert assignment
    moe_expand = torch.empty(moe_input_num, hidden_dim, dtype=dtype, device=device)
    moe_index = torch.full((moe_input_num,), -1, dtype=torch.int32, device=device)
    expert_m = torch.empty(expert_num, dtype=torch.int32, device=device)
    sorted_tokens_num_lod = torch.empty(expert_num + 1, dtype=torch.int32, device=device)

    xtorch_ops.moe_pre_sorted(
        hidden_states, topk_ids, block_statistic,
        moe_expand, moe_index, expert_m, sorted_tokens_num_lod
    )

    # Step 3: Inner FC (gate+up projection)
    inner_fc_out = torch.empty(seq_num, moe_topk, double_ffn_hd, dtype=dtype, device=device)
    xtorch_ops.moe_fc(
        moe_expand, w1, sorted_tokens_num_lod, moe_index, moe_topk, inner_fc_out,
    )
    inner_fc_out = inner_fc_out.view(moe_input_num, double_ffn_hd)

    # Step 4: SwiGLU activation (in-place on first half)
    ffn_hd = double_ffn_hd // 2
    swiglu_out = torch.empty(moe_input_num, ffn_hd, dtype=dtype, device=device)
    xtorch_ops.swiglu(inner_fc_out, swiglu_out)

    # Step 5: Outer FC (down projection)
    outer_fc_out = torch.empty(seq_num, moe_topk, hidden_dim, dtype=dtype, device=device)
    xtorch_ops.moe_fc(
        swiglu_out, w2, sorted_tokens_num_lod, moe_index, moe_topk, outer_fc_out,
    )
    outer_fc_out = outer_fc_out.view(moe_input_num, hidden_dim)

    # Step 6: Post-processing (weight and scatter-sum back)
    # New API: moe_post(x, moe_index, normed_scale, dequant_scale, y)
    # moe_index needs shape [seq_num, moe_topk]
    # dequant_scale: ones since no quantization
    moe_index_2d = moe_index.view(seq_num, moe_topk)
    dequant_scale = torch.ones(seq_num, moe_topk, dtype=torch.float32, device=device)
    xtorch_ops.moe_post(
        outer_fc_out, moe_index_2d, topk_weights, dequant_scale, output,
    )


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Kunlunxin fused experts implementation.
    
    This function matches the signature of vllm_fl.ops.fused_moe.fused_moe.fused_experts_impl
    and is patched in by the Kunlunxin patch system.
    """
    if use_fp8_w8a8 or use_int8_w8a16 or use_int4_w4a16:
        raise NotImplementedError(
            "Kunlunxin fused_experts does not support fp8/int8_w8a16/int4 quantization yet."
        )

    num_tokens = hidden_states.size(0)
    E = w1.size(0)
    if global_num_experts == -1:
        global_num_experts = E

    # Map global expert ids to local expert ids if needed
    if expert_map is not None:
        topk_ids = expert_map[topk_ids.long()].to(topk_ids.dtype)

    if inplace:
        output = hidden_states
    else:
        output = torch.zeros_like(hidden_states)

    _klx_fused_experts(
        hidden_states=hidden_states,
        output=output,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        use_int8_w8a8=use_int8_w8a8,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    return output
