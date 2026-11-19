"""Model-scoped materialized-sigmoid routing through the ordinary FL registry."""

import torch
from flag_gems import grouped_topk as native_grouped_topk, runtime
from flaggems_vllm.ops.grouped_topk import grouped_topk as materialized_grouped_topk


def grouped_topk(
    scores,
    n_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    bias,
    scoring_func=0,
):
    if (
        runtime.device.vendor_name == "metax"
        and scores.ndim == 2
        and 1 <= scores.shape[0] <= 4096
        and scores.shape[1] == 896
        and n_group == topk_group == 1
        and topk == 16
        and renormalize
        and routed_scaling_factor == 1.0
        and scoring_func == 1
        and scores.dtype in (torch.bfloat16, torch.float32)
        and bias.dtype in (torch.bfloat16, torch.float32)
        and scores.is_cuda
        and bias.device == scores.device
        and scores.stride(1) == 1
        and bias.numel() == 896
        and bias.is_contiguous()
    ):
        return materialized_grouped_topk(
            scores,
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            bias,
            scoring_func,
            sigmoid_mode="fp32_then_input",
        )
    return native_grouped_topk(
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )
