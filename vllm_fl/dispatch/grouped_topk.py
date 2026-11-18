# SPDX-License-Identifier: Apache-2.0
"""Dispatch grouped TopK with its optional padding capability."""

from vllm_fl.dispatch import CachedOp, get_default_manager
from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import (
    grouped_topk_supports_padding,
)

_grouped_topk = CachedOp("grouped_topk")


def _supports_padding(impl):
    # Only the FlagGems wrapper declares the mask contract; native ABI8 and
    # other vendor wrappers in this branch do not accept that argument.
    return impl.impl_id == "default.flagos" and grouped_topk_supports_padding()


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
    args = (
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )
    if is_padding is None:
        return _grouped_topk(*args)
    return get_default_manager().call_compatible(
        "grouped_topk", _supports_padding, *args, is_padding=is_padding
    )
