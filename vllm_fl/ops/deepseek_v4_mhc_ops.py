# Copyright (c) 2026 BAAI. All rights reserved.
"""Opaque custom-op wrappers for the DeepSeek-V4 mHC tilelang kernels.

torch.compile (VLLM_FL_DSV4_TORCH_COMPILE=1) cannot trace tilelang's JIT
dispatch (`_infer_jit_mode` -> inspect/importlib). The vendor fork solved
this by registering mhc_pre/mhc_post as torch custom ops; do the same here
around the upstream tilelang wrappers so dynamo sees opaque calls.
"""

import contextlib
import os

import torch

from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_fused_post_pre_tilelang,
    mhc_post_tilelang,
    mhc_pre_tilelang,
)
from vllm.utils.torch_utils import direct_register_custom_op

# --- mHC prenorm GEMM: route to PPU deep_gemm instead of tilelang -------------
#
# Upstream mhc_pre_tilelang / mhc_fused_post_pre_tilelang ALREADY have a
# deep_gemm path for the prenorm GEMM; it is gated on is_deep_gemm_supported(),
# which is False on PPU because PlatformFL.support_deep_gemm() only returns True
# for nvidia. So FL silently falls back to the tilelang kernel. Measured on
# 8x PPU-ZW810E, ctx 16384 prefill, 11 clean chunk steps per stack:
#
#     deep_gemm::sm80_tf32_hc_prenorm_gemm_impl   146.55 ms/step  (T-Head)
#     hc_prenorm_gemm_block_m_tilelang_kernel     332.30 ms/step  (FL)
#
# i.e. the tilelang kernel is 2.27x slower and accounts for +185.75 ms/chunk of
# the +453 ms (+15.5%) prefill-chunk gap vs T-Head. The block_m variant is
# selected by `x.shape[0] >= 1024`, which is why this hits prefill, not decode.
#
# The T-Head fork instead branches directly (model_executor/layers/mhc.py:281):
#     if current_platform.is_ppu():
#         from vllm.utils.ppu_deep_gemm import tf32_hc_prenorm_gemm
# and critically ALSO forces n_splits = 1 on PPU (mhc.py:242) where upstream
# would compute compute_num_split(...) > 1. Both are required: enabling the
# deep_gemm branch without pinning n_splits feeds the PPU kernel a split-k it
# does not expect.
#
# Scope: patched only for the duration of the two mHC calls, via the plugin's
# own opaque-op wrappers. Both symbols are imported INSIDE the upstream
# functions (mhc/tilelang.py:165 and :408), so a scoped patch reaches them,
# while the module-level importers of is_deep_gemm_supported (fp8.py:88,
# deep_gemm_moe.py:43, scaled_mm/deep_gemm.py:20) bound the original object at
# import time and are untouched. Not relying on that import order alone is why
# this is scoped rather than a global flip of support_deep_gemm().
#
# mHC runs inside an opaque custom op, i.e. outside the compiled graph, so the
# per-call patch cannot invalidate Dynamo guards — unlike an earlier attempt at
# patching torch.repeat_interleave per step, which cost 1.1% for that reason.
#
# Set VLLM_FL_MHC_DEEPGEMM=0 to restore the upstream tilelang path.
_MHC_DEEPGEMM = os.environ.get("VLLM_FL_MHC_DEEPGEMM", "1") == "1"


@contextlib.contextmanager
def _ppu_deepgemm_prenorm():
    """Make upstream's mHC prenorm take its deep_gemm branch with n_splits=1."""
    if not _MHC_DEEPGEMM:
        yield
        return
    import vllm.model_executor.kernels.mhc.tilelang_kernels as tk
    import vllm.utils.deep_gemm as udg

    orig_supported = udg.is_deep_gemm_supported
    orig_split = tk.compute_num_split
    udg.is_deep_gemm_supported = lambda: True
    tk.compute_num_split = lambda *a, **k: 1
    try:
        yield
    finally:
        udg.is_deep_gemm_supported = orig_supported
        tk.compute_num_split = orig_split


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
    with _ppu_deepgemm_prenorm():
        out = mhc_pre_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits=n_splits,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
    return tuple(t.contiguous() for t in out)


def _fl_mhc_pre_fake(
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
    norm_weight,
    norm_eps,
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
    with _ppu_deepgemm_prenorm():
        out = mhc_fused_post_pre_tilelang(
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
            n_splits=n_splits,
            tile_n=tile_n,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
    return tuple(t.contiguous() for t in out)


def _fl_mhc_fused_post_pre_fake(
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
    n_splits,
    tile_n,
    norm_weight,
    norm_eps,
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
    return torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=hs_flat.device)


direct_register_custom_op(
    op_name="fl_hc_head_fused",
    op_func=_fl_hc_head_fused,
    fake_impl=_fl_hc_head_fused_fake,
)


def hc_head_fused_opaque(hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps):
    return torch.ops.vllm.fl_hc_head_fused(
        hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps
    )


def mhc_pre_opaque(
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
    norm_eps=1e-6,
):
    return torch.ops.vllm.fl_mhc_pre(
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
        norm_weight,
        norm_eps,
    )


def mhc_post_opaque(x, residual, post_layer_mix, comb_res_mix):
    return torch.ops.vllm.fl_mhc_post(x, residual, post_layer_mix, comb_res_mix)


def mhc_fused_post_pre_opaque(
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
    norm_eps=1e-6,
):
    return torch.ops.vllm.fl_mhc_fused_post_pre(
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
        n_splits,
        tile_n,
        norm_weight,
        norm_eps,
    )
