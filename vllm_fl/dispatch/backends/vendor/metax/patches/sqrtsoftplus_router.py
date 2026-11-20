# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Select stable public routing numerics without changing other native routers."""

from functools import wraps
from inspect import signature

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.router import (
    fused_topk_bias_router as biased,
    grouped_topk_router as grouped,
)

from vllm_fl.dispatch import resolve_op

logger = init_logger(__name__)


def prepare():
    from vllm_fl.ops.fused_moe import router as fl_router

    original = biased.fused_topk_bias
    required = {
        "hidden_states",
        "gating_output",
        "scoring_func",
        "e_score_correction_bias",
        "topk",
        "renormalize",
        "indices_type",
        "input_tokens",
        "hash_indices_table",
        "routed_scaling_factor",
        "bias_vl",
        "image_sentinel_lo",
    }
    if not required.issubset(signature(original).parameters):
        raise RuntimeError("unsupported native biased-router contract")
    if not required.issubset(signature(fl_router.fused_topk_bias).parameters):
        raise RuntimeError("unsupported FL biased-router alias contract")
    if getattr(original, "_fl_stable_sqrtsoftplus", False):
        return [
            (module, "fused_topk_bias", original) for module in (grouped, fl_router)
        ]

    @wraps(original)
    def route(
        hidden_states,
        gating_output,
        scoring_func,
        e_score_correction_bias,
        topk,
        renormalize,
        indices_type=None,
        input_tokens=None,
        hash_indices_table=None,
        routed_scaling_factor=1.0,
        bias_vl=None,
        image_sentinel_lo=0,
    ):
        shape_scope = (
            hidden_states.shape[-1],
            gating_output.shape[-1],
            topk,
            image_sentinel_lo,
        ) in ((5120, 384, 6, 129264), (5120, 128, 3, 0))
        if (
            not shape_scope
            or scoring_func != "sqrtsoftplus"
            or hash_indices_table is not None
        ):
            return original(
                hidden_states,
                gating_output,
                scoring_func,
                e_score_correction_bias,
                topk,
                renormalize,
                indices_type,
                input_tokens,
                hash_indices_table,
                routed_scaling_factor,
                bias_vl,
                image_sentinel_lo,
            )
        import torch

        if hidden_states.shape[0] != gating_output.shape[0]:
            raise ValueError("routing hidden/logit rows must match")
        logger.info_once(
            "V4.1 stable sqrt-softplus routing selected through public FGV dispatch"
        )
        return resolve_op("sqrtsoftplus_topk")(
            gating_output,
            e_score_correction_bias,
            topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            input_ids=input_tokens,
            bias_vl=bias_vl,
            image_sentinel_lo=image_sentinel_lo,
            indices_dtype=torch.int32 if indices_type is None else indices_type,
            is_padding=biased._get_padding_mask(gating_output.shape[0]),
        )

    route._fl_stable_sqrtsoftplus = True
    return [
        (module, "fused_topk_bias", route) for module in (biased, grouped, fl_router)
    ]
