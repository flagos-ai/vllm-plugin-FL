# Copyright (c) 2026 BAAI. All rights reserved.
"""Large-M mHC pre + FlagGems-compatible RMSNorm, behind a graph boundary.

Keep the existing optional-norm MHC API unchanged: its rounding contract is
different. This entry preserves both BF16 roundings of the standalone chain.
"""

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def prefill_mhc_pre_norm(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    weight: torch.Tensor,
    rms_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_mult: float,
    sinkhorn_repeat: int,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .mhc import _get_aiter_mhc_ops, mhc_pre_hygon

    eligible = (
        residual.is_cuda and residual.dtype == torch.bfloat16
        and residual.ndim == 3 and residual.shape[1:] == (4, 4096)
        and 1024 <= residual.shape[0] <= 8192 and residual.is_contiguous()
        and fn.shape == (24, 16384) and weight.shape == (4096,)
        and scale.shape == (3,) and base.shape == (24,)
        and weight.dtype == torch.bfloat16
        and all(t.dtype == torch.float32 for t in (fn, scale, base))
        and all(t.is_contiguous() and t.device == residual.device
                for t in (fn, scale, base, weight))
        and _get_aiter_mhc_ops() is not None
    )
    if not eligible:
        import flag_gems

        post, comb, y = mhc_pre_hygon(
            residual, fn, scale, base, rms_eps, pre_eps,
            sinkhorn_eps, post_mult, sinkhorn_repeat,
        )
        return post, comb, flag_gems.rms_norm(
            y, (y.shape[-1],), weight, norm_eps)

    from aiter.ops.tilelang.mhc.pre_big_fuse import get_block_info
    from aiter.ops.tilelang.mhc.pre_norm_fn_splitk_kernel import (
        mhc_pre_gemm_sqrsum_splitk_kernel,
    )
    from .prefill_mhc_kernel import _mhc_pre_norm_wave

    m = residual.shape[0]
    cu_count = torch.cuda.get_device_properties(residual.device).multi_processor_count
    block = get_block_info((m + 31) // 32 * 32, 16384, cu_count)
    splits = block.n_splits_pre
    kernel, _ = mhc_pre_gemm_sqrsum_splitk_kernel(
        24, 16384, split_k=splits, token_block=block.token_block,
        hidden_block=block.hidden_block, round_fn_to_tf32=True,
    )
    partial = torch.empty((splits, m, 24), device=residual.device, dtype=torch.float32)
    square_sum = torch.empty((splits, m), device=residual.device, dtype=torch.float32)
    post = torch.empty((m, 4), device=residual.device, dtype=torch.float32)
    comb = torch.empty((m, 16), device=residual.device, dtype=torch.float32)
    y = torch.empty((m, 4096), device=residual.device, dtype=torch.bfloat16)
    kernel(residual.view(m, 16384), fn, partial, square_sum)
    _mhc_pre_norm_wave(
        4096, norm_eps, rms_eps, pre_eps, sinkhorn_eps,
        post_mult, sinkhorn_repeat, n_splits=splits,
    )(partial, square_sum, scale, base, residual, weight, post, comb, y)
    return post.view(m, 4, 1), comb.view(m, 4, 4), y


def _fake(residual, fn, scale, base, weight, rms_eps, pre_eps,
          sinkhorn_eps, post_mult, sinkhorn_repeat, norm_eps):
    shape = residual.shape[:-2]
    h, d = residual.shape[-2:]
    return (
        residual.new_empty((*shape, h, 1), dtype=torch.float32),
        residual.new_empty((*shape, h, h), dtype=torch.float32),
        residual.new_empty((*shape, d)),
    )


direct_register_custom_op(
    op_name="fl_hygon_dv4_prefill_mhc_pre_norm",
    op_func=prefill_mhc_pre_norm,
    mutates_args=[],
    fake_impl=_fake,
)
