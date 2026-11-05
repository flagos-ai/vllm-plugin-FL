# SPDX-License-Identifier: Apache-2.0
"""Default-OFF model calls; all new numerical code lives in FlagGems-vllm."""

from vllm_fl.utils import use_flaggems_op


def maybe_ihc_pair_pre(owner, x):
    if not use_flaggems_op("ihc_pre_pair", default=False):
        return None
    from flaggems_vllm import can_ihc_pair, ihc_pre_pair

    args = (
        x,
        owner.hc_fn.weight,
        owner.hc_scale,
        owner.hc_base,
        owner.magnitude,
        owner.hc_eps,
        owner.layernorm_epsilon,
    )
    if not can_ihc_pair(*args):
        return None
    return ihc_pre_pair(*args)


def maybe_ihc_pair_post_pre(
    x, residual, post, weight, scale, base, magnitude, hc_eps, norm_eps
):
    if not use_flaggems_op("ihc_post_pre_pair", default=False):
        return None
    from flaggems_vllm import can_ihc_pair, ihc_post_pre_pair

    if not can_ihc_pair(
        x,
        weight,
        scale,
        base,
        magnitude,
        hc_eps,
        norm_eps,
        residual=residual,
        post=post,
    ):
        return None
    return ihc_post_pre_pair(
        x, residual, post, weight, scale, base, magnitude, hc_eps, norm_eps
    )
