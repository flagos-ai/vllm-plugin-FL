# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.


# ---------------------------------------------------
# Enabled by the typed MoE policy in vllm_fl.dsv4_config.
# ---------------------------------------------------

import torch
import contextlib
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer
from vllm.logger import logger

mctlass_op = None
mctlass_scaled_gemm = None
with contextlib.suppress(ImportError):
    if mctlass_op is None:
        import mctlassEx

        mctlass_op = mctlassEx.mctlassExHandleWrapper()

        try:
            from mctlassEx import ScaledGEMM

            mctlass_scaled_gemm = ScaledGEMM()
        except ImportError:
            logger.warning(
                "Failed to import ScaledGEMM from mctlass. "
                "scaled_mm_azp not support for now"
            )


# w8a8 scaled mm
def mctlassEx_w8a8_scaled_mm_azp(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    azp_adj: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
) -> torch.Tensor:
    if bias is not None and bias.dim() == 1:
        bias = bias.unsqueeze(0)

    if azp is not None or azp_adj is not None or bias is not None:
        assert mctlass_scaled_gemm is not None, (
            "scaled_mm with azp and bias is not supported for current mctlass version"
        )
        _, K = a.shape
        M, N = out.shape
        mctlass_scaled_gemm(
            [M, N, K], a, b, out, scale_a, scale_b.T, bias, azp_adj=azp_adj, azp=azp
        )
    else:
        assert mctlass_op is not None, "mctlassOp is not imported correctly"
        stream = torch.cuda.current_stream().cuda_stream
        mctlass_op.mctlass_w8a8_scaled_mm_azp(
            a, b, out, scale_a, scale_b.T, bias, azp_adj, azp, stream
        )
    return out


def mctlassEx_w8a8_scaled_mm_azp_fake(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    azp_adj: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
) -> torch.Tensor:
    return out


direct_register_custom_op(
    op_name="mctlassEx_w8a8_scaled_mm_azp",
    op_func=mctlassEx_w8a8_scaled_mm_azp,
    mutates_args=["out"],
    fake_impl=mctlassEx_w8a8_scaled_mm_azp_fake,
    tags=(
        ()
        if is_torch_equal_or_newer("2.7.0")
        else (torch.Tag.needs_fixed_stride_order,)
    ),
)


# w8a8 fused moe
def mctlassEx_fused_moe_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM: int,
    topk: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    # TODO: need mctlass to fix it
    stream = torch.cuda.current_stream().cuda_stream
    c1 = c.view(-1, c.size(-1)).contiguous()
    assert mctlass_op is not None, "mctlassOp is not imported correctly"
    mctlass_op.mctlass_fuse_moe_gemm(
        a,
        b,
        c1,
        a_scales,
        b_scales,
        topk_weights,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        EM,
        topk,
        mul_routed_weight,
        stream,
    )
    return c1.reshape(c.shape)


def mctlassEx_fused_moe_gemm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM: int,
    topk: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    return c


direct_register_custom_op(
    op_name="mctlassEx_fused_moe_gemm",
    op_func=mctlassEx_fused_moe_gemm,
    mutates_args=["c"],
    fake_impl=mctlassEx_fused_moe_gemm_fake,
    tags=(
        ()
        if is_torch_equal_or_newer("2.7.0")
        else (torch.Tag.needs_fixed_stride_order,)
    ),
)


# w4a8 fused moe


# -------------------------------------------------
# Note:
#
# This is different from `cutlass_scaled_mm` in `_cutlass_ops.py`.
# It invokes mctlassEx python API directly.
# -------------------------------------------------
def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype

    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])
    a = a.view(-1, a.shape[-1])

    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)

    torch.ops.vllm.mctlassEx_w8a8_scaled_mm_azp(out, a, b, scale_a, scale_b, bias)

    return out.view(*target_shape)


# -------------------------------------------------
# Note:
#
# This is different from `cutlass_scaled_mm_azp` in `_cutlass_ops.py`.
# It invokes mctlassEx python API directly.
# -------------------------------------------------
def cutlass_scaled_mm_azp(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    azp_adj: torch.Tensor,
    azp: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype

    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])
    a = a.view(-1, a.shape[-1])
    assert azp is None or azp.numel() == a.shape[0]

    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    torch.ops.vllm.mctlassEx_w8a8_scaled_mm_azp(
        out, a, b, scale_a, scale_b, bias, azp_adj, azp
    )

    return out.view(*target_shape)


# -------------------------------------------------
# Note:
#
# This is different from `cutlass_moe_mm_w8a8_get_kernel_m` in `_cutlass_ops.py`.
# It invokes mctlassEx python API directly.
# -------------------------------------------------
_MCTLASS_KM_CACHE = {}


def cutlass_moe_mm_w8a8_get_kernel_m(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, topk: int
) -> int:
    assert mctlass_op is not None, "mctlassOp is not imported correctly"
    # PATCH(km_cache): mctlass_fuse_moe_get_kernel_m only uses tensor SHAPES
    # (verified: km stable wrt values/dtype; pure M-ladder). The original code
    # cast the real bf16 activation/weight to int8 on EVERY call just to query
    # BLOCK_SIZE_M, costing a [M,H] plus a [E,N,H] direct_copy per layer
    # (~7us decode, 78-149us prefill). Query with a shape-matched empty int8
    # activation, reuse the already-int8 W8A8 weight, and cache by shape:
    # semantics of kernel selection are unchanged and no full-size weight
    # scratch allocation is introduced on a cache miss.
    key = (tuple(a.shape), tuple(b.shape), tuple(c.shape), topk)
    km = _MCTLASS_KM_CACHE.get(key)
    if km is None:
        qa = torch.empty(a.shape, dtype=torch.int8, device=a.device)
        if b.dtype != torch.int8:
            raise TypeError(
                "cutlass_moe_mm_w8a8_get_kernel_m expects INT8 W8A8 weights; "
                f"got {b.dtype}"
            )
        c1 = c.view(-1, c.size(-1)).contiguous()
        km = mctlass_op.mctlass_fuse_moe_get_kernel_m(qa, b, c1, topk)
        _MCTLASS_KM_CACHE[key] = km
    return km


# -------------------------------------------------
# Note:
#
# This is different from `cutlass_moe_mm_w8a8` in `_cutlass_ops.py`.
# It invokes mctlassEx python API directly.
# -------------------------------------------------
def cutlass_moe_mm_w8a8(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    moe_weight: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM: int,
    topk: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    torch.ops.vllm.mctlassEx_fused_moe_gemm(
        a,
        b,
        c,
        a_scales,
        b_scales,
        moe_weight,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        EM,
        topk,
        mul_routed_weight,
    )


# GEMM


# Compatibility names only; original implementations above are unchanged.
scaled_mm = cutlass_scaled_mm
scaled_mm_azp = cutlass_scaled_mm_azp
kernel_m = cutlass_moe_mm_w8a8_get_kernel_m
