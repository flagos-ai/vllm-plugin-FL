# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.24 MHC reference with the GLM-required final RMSNorm."""

import torch


def rms_norm(x, weight, eps):
    if weight is None:
        return x
    value = x.float()
    return (
        value
        * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        * weight.float()
    ).to(x.dtype)


def mhc_pre(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    norm_weight=None,
    norm_eps=0.0,
):
    from vllm.model_executor.layers.mhc import MHCPreOp

    post, comb, x = MHCPreOp.forward_native(
        None,
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )
    return post, comb, rms_norm(x, norm_weight, norm_eps)


def mhc_post(x, residual, post_layer_mix, comb_res_mix):
    from vllm.model_executor.layers.mhc import MHCPostOp

    return MHCPostOp.forward_native(None, x, residual, post_layer_mix, comb_res_mix)


def mhc_fused_post_pre(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=0.0,
):
    updated = mhc_post(x, residual, post_layer_mix, comb_res_mix)
    return (
        updated,
        *mhc_pre(
            updated,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        ),
    )
