# Copyright (c) 2026 BAAI. All rights reserved.
"""Hygon indexer bridge independent of old FlagGems' 8-argument _C schema."""
import logging

import torch

logger = logging.getLogger(__name__)


@torch.library.custom_op(
    "vllm_fl::hygon_indexer_group_quant",
    mutates_args=("output_q", "output_s"),
)
def _group_quant(
    x: torch.Tensor, output_q: torch.Tensor, output_s: torch.Tensor,
    group_size: int, eps: float, fp8_min: float, fp8_max: float,
    use_ue8m0: bool,
) -> None:
    # Reuse vLLM's source-visible Triton fallback, not the incompatible _C op.
    import triton
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    if x.numel() == 0:
        return
    block = triton.next_power_of_2(group_size)
    kwargs = dict(fp8_min=fp8_min, fp8_max=fp8_max, use_ue8m0=use_ue8m0,
                  BLOCK=block, num_warps=min(max(block // 256, 1), 8),
                  num_stages=1)
    grid = (x.numel() // group_size,)
    if output_s.stride(0) == 1 and output_s.stride(1) != 1:
        fp8_utils._per_token_group_quant_fp8_colmajor[grid](
            x, output_q, output_s, group_size, x.shape[1], x.stride(0),
            output_s.stride(1), eps, **kwargs)
    else:
        fp8_utils._per_token_group_quant_fp8[grid](
            x, output_q, output_s, group_size, x.shape[1], x.stride(0),
            eps, **kwargs)


@_group_quant.register_fake
def _group_quant_fake(x, output_q, output_s, group_size, eps, fp8_min,
                     fp8_max, use_ue8m0):
    return None


def per_token_group_quant_fp8(
    x, group_size, eps=1e-10, dtype=None, column_major_scales=False,
    tma_aligned_scales=False, out_q=None, use_ue8m0=None,
):
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    if x.ndim != 2 or x.stride(-1) != 1 or group_size <= 0:
        raise ValueError("Hygon indexer quant requires a 2D group-contiguous input")
    if x.shape[1] % group_size or eps <= 0:
        raise ValueError("Invalid indexer group size or eps")
    dtype = fp8_utils.current_platform.fp8_dtype() if dtype is None else dtype
    if dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        raise ValueError("Hygon indexer bridge requires E4M3 output")
    if use_ue8m0 is None:
        use_ue8m0 = fp8_utils.is_deep_gemm_e8m0_used()
    if out_q is None:
        out_q = torch.empty(x.shape, device=x.device, dtype=dtype)
    elif (out_q.shape != x.shape or out_q.dtype != dtype
          or out_q.device != x.device or not out_q.is_contiguous()):
        raise ValueError("Invalid indexer out_q shape/dtype/device/layout")
    rows, groups = x.shape[0], x.shape[1] // group_size
    if column_major_scales:
        leading = ((rows + 3) // 4) * 4 if tma_aligned_scales else rows
        scales = torch.empty_strided((rows, groups), (1, leading),
                                    device=x.device, dtype=torch.float32)
    else:
        scales = torch.empty((rows, groups), device=x.device, dtype=torch.float32)
    fp8_min, fp8_max = fp8_utils.get_fp8_min_max()
    _group_quant(x, out_q, scales, group_size, eps, fp8_min, fp8_max,
                 use_ue8m0)
    return out_q, scales


def patch_indexer_group_quant():
    from vllm.model_executor.models import deepseek_v2

    # Only the shared GLM/DSA indexer call site; other quantizers are untouched.
    deepseek_v2.per_token_group_quant_fp8 = per_token_group_quant_fp8
    logger.info("Patched Hygon DSA indexer group quant with stride-aware Triton bridge")
