# SPDX-License-Identifier: Apache-2.0
"""Qwen expert GEMM selection; assignment and activation remain framework-owned."""

import torch


def prepare_moe_stages(x, w1, w2, c1, activated, c2, *, eligible):
    if not eligible or not 31 <= x.shape[0] <= 2048:
        return None, None
    from flaggems_vllm.runtime.backend._metax.ops.mctlass_bf16 import get_plan

    p1 = get_plan(x, w1, c1, topk=10)
    # Keep the tuned small-M G2: the closed kernel regressed on few-hot routes.
    p2 = (
        get_plan(activated, w2, c2, topk=1, weighted=True)
        if x.shape[0] >= 1024
        else None
    )
    return p1, p2


def moe_eligible(
    x,
    w1,
    w2,
    topk,
    device_name,
    activation,
    quantized,
    has_lora,
    has_bias,
    expert_map,
    global_experts,
    apply_router_weight_on_input,
):
    return (
        31 <= x.shape[0] <= 2048
        and "C550" in device_name
        and x.dtype == w1.dtype == w2.dtype == torch.bfloat16
        and tuple(w1.shape) == (512, 160, 2560)
        and tuple(w2.shape) == (512, 2560, 80)
        and topk == 10
        and activation == "silu"
        and not quantized
        and not has_lora
        and not has_bias
        and expert_map is None
        and global_experts == 512
        and not apply_router_weight_on_input
    )
