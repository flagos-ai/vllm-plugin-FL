# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 2026 - Modified by Kunlunxin, Inc. All Rights Reserved.

"""
Kunlunxin fused MoE kernel implementations and KunlunxinSharedFusedMoE layer.
"""

from __future__ import annotations

import functools
from typing import Callable, Optional

import torch

from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEQuantConfig,
)
from vllm.utils.torch_utils import direct_register_custom_op

import xtorch_ops

# for kunlunxin vendor
_KLX_MOE_BLOCK_NUM= 12


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

    Pipeline: gen_block_statistic → moe_pre_sorted → moe_fc(w1) → swiglu → moe_fc(w2) → post (weight+sum)
    """
    if use_int8_w8a8 or use_int8_w4a8:
        raise NotImplementedError("_klx_fused_experts is not supported for int8 w8a8 and w4a8.")

    seq_num, hidden_dim = hidden_states.shape
    moe_topk = topk_ids.shape[1]
    expert_num = w1.shape[0]
    double_ffn_hd = w1.shape[1]       # gate+up output dim (2 * intermediate_size for swiglu)
    moe_input_num = seq_num * moe_topk

    device = hidden_states.device
    dtype = hidden_states.dtype

    # Step 1: Generate block statistics for expert-parallel sorting
    block_statistic = torch.zeros(
        _KLX_MOE_BLOCK_NUM, expert_num,
        dtype=torch.int32,
        device=device,
    )
    xtorch_ops.gen_block_statistic(topk_ids, block_statistic)

    # Step 2: Sort tokens by expert assignment (physical reorder)
    moe_expand = torch.empty(moe_input_num, hidden_dim, dtype=dtype, device=device)
    moe_index = torch.full((moe_input_num,), -1, dtype=torch.int32, device=device)
    expert_m = torch.empty(expert_num, dtype=torch.int32, device=device)
    sorted_tokens_num_lod = torch.empty(expert_num + 1, dtype=torch.int32, device=device)

    xtorch_ops.moe_pre_sorted(
        hidden_states, topk_ids, block_statistic,
        moe_expand, moe_index, expert_m, sorted_tokens_num_lod
    )

    # Step 3: Inner FC (gate+up projection) — NO activation, NO topk_weights
    # Output: [moe_input_num, double_ffn_hd] (will be split by swiglu)
    inner_fc_out = torch.empty(moe_input_num, double_ffn_hd, dtype=dtype, device=device)

    # leave it for other functions (e.g. w8a8)
    moe_fc_w1_kwargs = {}

    xtorch_ops.moe_fc_v3(
        moe_expand,               # sorted input
        w1,                       # weight
        sorted_tokens_num_lod,
        moe_index,                # sorted_tokens_idx
        moe_topk,
        inner_fc_out,             # output
        sort_mode=True,
        **moe_fc_w1_kwargs,
    )

    # Step 4: SwiGLU activation
    swiglu_out = torch.empty(moe_input_num, double_ffn_hd // 2, dtype=dtype, device=device)
    xtorch_ops.swiglu(inner_fc_out, swiglu_out)

    # Step 5: Outer FC (down projection)
    # leave it for other functions (e.g. w8a8)
    moe_fc_w2_kwargs = {}

    outer_fc_out = torch.empty(moe_input_num, hidden_dim, dtype=dtype, device=device)

    xtorch_ops.moe_fc_v3(
        swiglu_out,               # sorted intermediate
        w2,                       # weight
        sorted_tokens_num_lod,
        moe_index,                # sorted_tokens_idx
        moe_topk,
        outer_fc_out,             # output
        sort_mode=True,
        **moe_fc_w2_kwargs,
    )

    # Step 6: Apply topk_weights and reduce across topk dimension (post fusion)
    moe_index = moe_index.reshape(-1, moe_topk)
    post_scale = torch.ones(seq_num, moe_topk, device=device, dtype=torch.float32)

    xtorch_ops.moe_post(outer_fc_out,
                        moe_index,
                        topk_weights,
                        post_scale,
                        output,
                        )


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_int8_w4a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> None:
    """In-place fused expert computation."""
    if use_int8_w8a8 and not per_channel_quant:
        raise NotImplementedError(
            f"Per-tensor quantization is not supported in int8-w8a8 mode. "
            f"Current settings: use_int8_w8a8={use_int8_w8a8}, "
            f"per_channel_quant={per_channel_quant}"
        )
    if use_int8_w4a8 and not per_channel_quant:
        raise NotImplementedError(
            f"Per-tensor quantization is not supported in int4-w4a8 mode. "
            f"Current settings: use_int8_w4a8={use_int8_w4a8}, "
            f"per_channel_quant={per_channel_quant}"
        )
    assert topk_weights.dtype == torch.float32
    _klx_fused_experts(
        hidden_states,
        hidden_states,
        w1,
        w2,
        topk_weights, topk_ids,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w4a8=use_int8_w4a8,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )



def inplace_fused_experts_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    use_int8_w4a8: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> None:
    pass


direct_register_custom_op(
    op_name="kunlunxin_inplace_fused_experts",
    op_func=inplace_fused_experts,
    mutates_args=["hidden_states"],
    fake_impl=inplace_fused_experts_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
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
    a2_scale: Optional[torch.Tensor] = None) -> torch.Tensor:

    output = torch.empty_like(hidden_states)
    if use_int8_w8a8 and not per_channel_quant:
        raise NotImplementedError(
            f"Per-tensor quantization is not supported in int8-w8a8 mode. "
            f"Current settings: use_int8_w8a8={use_int8_w8a8}, "
            f"per_channel_quant={per_channel_quant}"
        )
    assert topk_weights.dtype == torch.float32
    _klx_fused_experts(
        hidden_states,
        output,
        w1,
        w2,
        topk_weights,
        topk_ids,
        use_int8_w8a8=use_int8_w8a8,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )
    return output



def outplace_fused_experts_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="kunlunxin_outplace_fused_experts",
    op_func=outplace_fused_experts,
    mutates_args=[],
    fake_impl=outplace_fused_experts_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def torch_vllm_inplace_fused_experts(**kwargs) -> torch.Tensor:
    torch.ops.vllm.kunlunxin_inplace_fused_experts(**kwargs)
    return kwargs["hidden_states"]


def torch_vllm_outplace_fused_experts(**kwargs) -> torch.Tensor:
    return torch.ops.vllm.kunlunxin_outplace_fused_experts(**kwargs)


def dispatch_fused_experts_func(inplace: bool) -> Callable[..., torch.Tensor]:
    if inplace:
        return torch_vllm_inplace_fused_experts
    return torch_vllm_outplace_fused_experts


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output_input: Optional[torch.Tensor] = None,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    quant_config: Optional[FusedMoEQuantConfig] = None,
    allow_deep_gemm: bool = False,
    allow_cutlass_block_scaled_grouped_gemm: bool = False,
) -> torch.Tensor:
    """High-level fused experts entry point."""
    if quant_config is None:
        quant_config = FUSED_MOE_UNQUANTIZED_CONFIG

    assert allow_deep_gemm == False
    assert quant_config.use_fp8_w8a8 == False

    fn = dispatch_fused_experts_func(inplace)
    return fn(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=quant_config.use_fp8_w8a8,
        use_int8_w8a8=quant_config.use_int8_w8a8,
        use_int8_w8a16=quant_config.use_int8_w8a16,
        use_int4_w4a16=quant_config.use_int4_w4a16,
        per_channel_quant=quant_config.per_act_token_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=quant_config.w1_scale,
        w2_scale=quant_config.w2_scale,
        w1_zp=quant_config.w1_zp,
        w2_zp=quant_config.w2_zp,
        a1_scale=quant_config.a1_scale,
        a2_scale=quant_config.a2_scale,
    )


def _kunlunxin_quant_method_forward(
    original_forward,
    layer,
    x: torch.Tensor,
    router_logits: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Replacement forward for the quant_method that uses Kunlunxin kernels."""
    from .experts_selector import select_experts

    topk_weights, topk_ids, zero_expert_result = select_experts(
        layer=layer,
        hidden_states=x,
        router_logits=router_logits,
    )

    result = fused_experts(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=layer.activation,
        apply_router_weight_on_input=layer.apply_router_weight_on_input,
        global_num_experts=layer.global_num_experts,
        expert_map=layer.expert_map,
    )

    if layer.zero_expert_num != 0 and layer.zero_expert_type is not None:
        assert not isinstance(result, tuple), (
            "Shared + zero experts are mutually exclusive not yet supported"
        )
        return result, zero_expert_result
    else:
        return result


class KunlunxinSharedFusedMoE(SharedFusedMoE):
    """SharedFusedMoE with Kunlunxin-optimized expert routing and computation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Patch the quant_method to use Kunlunxin expert kernels
        if self.quant_method is not None:
            original_apply = self.quant_method.apply
            self.quant_method.apply = functools.partial(
                _kunlunxin_quant_method_forward,
                original_apply,
            )

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        from .experts_selector import select_experts

        return select_experts(self, hidden_states, router_logits)
