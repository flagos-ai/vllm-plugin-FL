# SPDX-License-Identifier: Apache-2.0
"""Padding-aware Hy4 Flash routing through the plugin operator registry."""

from vllm_fl.dispatch import CachedOp

_grouped_topk = CachedOp("grouped_topk")


def grouped_topk(
    scores,
    n_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    bias,
    scoring_func=0,
    is_padding=None,
):
    return _grouped_topk(
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
        is_padding=is_padding,
    )
