# SPDX-License-Identifier: Apache-2.0
"""Qwen expert GEMM selection; assignment and activation remain framework-owned."""

from functools import lru_cache

import torch

from .shape_limits import load_tuning_config, max_prefill_rows


@lru_cache(maxsize=1)
def _capacity_settings():
    return load_tuning_config()["moe_capacity"]


def _closed_rows(rows):
    return (
        31 <= rows <= max_prefill_rows()
        or rows in _capacity_settings()["additional_closed_rows"]
    )


def prepare_moe_stages(x, w1, w2, c1, activated, c2, *, eligible):
    if not eligible or not _closed_rows(x.shape[0]):
        return None, None
    from flaggems_vllm.runtime.backend._metax.ops.mctlass_bf16 import get_plan

    p1 = get_plan(x, w1, c1, topk=10)
    # Keep tuned small-M G2 except the separately validated M16 graph bucket.
    p2 = (
        get_plan(activated, w2, c2, topk=1, weighted=True)
        if x.shape[0] >= 1024 or x.shape[0] in _capacity_settings()["closed_down_rows"]
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
        _closed_rows(x.shape[0])
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


def direct_outputs_supported(config, quant_config, device_name):
    """Opt in only the reviewed unquantized Qwen TP8 expert geometry."""
    parallel = config.moe_parallel_config
    return (
        "C550" in device_name
        and config.in_dtype == torch.bfloat16
        and config.hidden_dim == 2560
        and config.intermediate_size_per_partition == 80
        and config.num_experts == config.num_local_experts == 512
        and config.experts_per_token == 10
        and config.activation.value == "silu"
        and parallel.tp_size == 8
        and parallel.pcp_size
        == parallel.dp_size
        == parallel.ep_size
        == parallel.sp_size
        == 1
        and not parallel.use_all2all_kernels
        and not config.has_bias
        and not config.is_lora_enabled
        and quant_config.quant_dtype is None
        and quant_config.gemm1_clamp_limit is None
    )


def silu_activation_out(input, output):
    """Reuse the library's identical pointwise kernel with the supplied buffer."""
    from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out

    width = input.shape[-1] // 2
    return silu_and_mul_out(input[..., :width], input[..., width:], output)
