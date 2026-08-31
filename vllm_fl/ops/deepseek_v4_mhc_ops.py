# Copyright (c) 2026 BAAI. All rights reserved.
"""Opaque custom-op wrappers for the DeepSeek-V4 mHC tilelang kernels.

torch.compile (VLLM_FL_DSV4_TORCH_COMPILE=1) cannot trace tilelang's JIT
dispatch (`_infer_jit_mode` -> inspect/importlib). The vendor fork solved
this by registering mhc_pre/mhc_post as torch custom ops; do the same here
around the upstream tilelang wrappers so dynamo sees opaque calls.
"""

import torch

from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_fused_post_pre_tilelang,
    mhc_post_tilelang,
    mhc_pre_tilelang,
)
from vllm.utils.torch_utils import direct_register_custom_op


def _fl_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = mhc_pre_tilelang(
        residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
        n_splits=n_splits, norm_weight=norm_weight, norm_eps=norm_eps,
    )
    return tuple(t.contiguous() for t in out)


def _fl_mhc_pre_fake(
    residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
    hc_post_mult_value, sinkhorn_repeat, n_splits, norm_weight, norm_eps,
):
    hc_mult, hidden = residual.shape[-2], residual.shape[-1]
    lead = residual.shape[:-2]
    post = torch.empty(*lead, hc_mult, dtype=torch.float32, device=residual.device)
    res = torch.empty(
        *lead, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
    )
    x = torch.empty(*lead, hidden, dtype=residual.dtype, device=residual.device)
    return post, res, x


def _fl_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix).contiguous()


def _fl_mhc_post_fake(x, residual, post_layer_mix, comb_res_mix):
    return torch.empty_like(residual)


def _fl_mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    tile_n: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = mhc_fused_post_pre_tilelang(
        x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat, n_splits=n_splits, tile_n=tile_n,
        norm_weight=norm_weight, norm_eps=norm_eps,
    )
    return tuple(t.contiguous() for t in out)


def _fl_mhc_fused_post_pre_fake(
    x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
    rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
    sinkhorn_repeat, n_splits, tile_n, norm_weight, norm_eps,
):
    hc_mult, hidden = residual.shape[-2], residual.shape[-1]
    lead = residual.shape[:-2]
    new_res = torch.empty_like(residual)
    post = torch.empty(*lead, hc_mult, dtype=torch.float32, device=residual.device)
    res = torch.empty(
        *lead, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
    )
    xi = torch.empty(*lead, hidden, dtype=residual.dtype, device=residual.device)
    return new_res, post, res, xi


direct_register_custom_op(
    op_name="fl_mhc_pre",
    op_func=_fl_mhc_pre,
    fake_impl=_fl_mhc_pre_fake,
)
direct_register_custom_op(
    op_name="fl_mhc_post",
    op_func=_fl_mhc_post,
    fake_impl=_fl_mhc_post_fake,
)
direct_register_custom_op(
    op_name="fl_mhc_fused_post_pre",
    op_func=_fl_mhc_fused_post_pre,
    fake_impl=_fl_mhc_fused_post_pre_fake,
)


def _fl_hc_head_fused(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    return hc_head_fused_kernel_tilelang(
        hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps
    ).contiguous()


def _fl_hc_head_fused_fake(hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps):
    num_tokens, hc_mult, hidden = hs_flat.shape
    return torch.empty(
        num_tokens, hidden, dtype=torch.bfloat16, device=hs_flat.device
    )


direct_register_custom_op(
    op_name="fl_hc_head_fused",
    op_func=_fl_hc_head_fused,
    fake_impl=_fl_hc_head_fused_fake,
)


def hc_head_fused_opaque(hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps):
    return torch.ops.vllm.fl_hc_head_fused(
        hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps
    )


def mhc_pre_opaque(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                   hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                   n_splits=1, norm_weight=None, norm_eps=1e-6):
    return torch.ops.vllm.fl_mhc_pre(
        residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits,
        norm_weight, norm_eps,
    )


def mhc_post_opaque(x, residual, post_layer_mix, comb_res_mix):
    return torch.ops.vllm.fl_mhc_post(x, residual, post_layer_mix, comb_res_mix)


def mhc_fused_post_pre_opaque(x, residual, post_layer_mix, comb_res_mix, fn,
                              hc_scale, hc_base, rms_eps, hc_pre_eps,
                              hc_sinkhorn_eps, hc_post_mult_value,
                              sinkhorn_repeat, n_splits=1, tile_n=1,
                              norm_weight=None, norm_eps=1e-6):
    return torch.ops.vllm.fl_mhc_fused_post_pre(
        x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat, n_splits, tile_n, norm_weight, norm_eps,
    )
