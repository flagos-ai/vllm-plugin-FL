# Copyright (c) 2026 FlagOS Contributors. All rights reserved.

"""FlagGems provider for dynamic per-token INT8 MoE experts."""


def w8a8_moe_experts(
    *,
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    w1_scale,
    w2_scale,
    w1_bias,
    w2_bias,
    expert_map,
    apply_router_weight_on_input,
    activation,
    global_num_experts,
    clamp_limit,
):
    if activation != "silu":
        raise NotImplementedError("FlagGems W8A8 MoE supports SwiGLU only")
    if clamp_limit is not None and clamp_limit > 0:
        raise NotImplementedError(
            "FlagGems fused experts does not expose vLLM's SwiGLU clamp contract"
        )

    from vllm_fl.quantization.w8a8.moe_experts import _flaggems_fused_experts_impl

    return _flaggems_fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=True,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )
