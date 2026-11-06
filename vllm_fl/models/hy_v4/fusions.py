# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Shared Hy4 fusion adapters through public operator policy and APIs."""

from vllm_fl.utils import use_flaggems_op


def triton_ihc_pre(x, weight, scale, base, magnitude, hc_eps, norm_eps):
    from flaggems_vllm import ihc_pre

    return ihc_pre(x, weight, scale, base, magnitude, hc_eps, norm_eps)


def triton_ihc_post(x, residual, post):
    from flaggems_vllm import ihc_post

    return ihc_post(x, residual, post)


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


def maybe_mrope_direct_cache(rotary, positions, q_pe, k_pe):
    if not use_flaggems_op("mrope_strided_cache", default=False):
        return None
    if (
        positions.ndim != 2
        or (rotary.head_size != 64)
        or (rotary.rotary_dim != 64)
        or (rotary.is_neox_style is not False)
    ):
        return None
    sections = rotary.mrope_section
    interleaved = rotary.mrope_interleaved
    cache = rotary.cos_sin_cache
    from flaggems_vllm.ops.mrope_strided_cache import (
        can_mrope_strided_cache,
        mrope_strided_cache,
    )

    if not can_mrope_strided_cache(q_pe, k_pe, cache, positions, sections, interleaved):
        return None
    return mrope_strided_cache(q_pe, k_pe, cache, positions, sections, interleaved)
