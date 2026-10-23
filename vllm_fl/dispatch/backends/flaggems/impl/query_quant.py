# SPDX-License-Identifier: Apache-2.0
"""Opaque compile boundary for the public FlagGems query quantization API."""

import torch


def _fp8_query_dtype():
    """Force float8_e4m3fn output when FlagGems falls back to float32.

    FlagGems selects ``SUPPORTED_FP8_DTYPE = float8_e4m3fn`` only on devices it
    reports as Hopper-or-newer; on older capabilities it stores the
    quantized values in float32 instead. The HY4 sparse indexer hands this
    tensor to the FP8 MQA-logits kernel, which requires an E4M3FN query, so
    the capability-based fallback breaks the sparse path on devices such as
    MetaX C550 (reports 8.0) even though they do support the dtype.

    Returns None (FlagGems default) when the dtype is unavailable.
    """
    from flag_gems.ops.per_token_group_quant_fp8 import SUPPORTED_FP8_DTYPE

    if SUPPORTED_FP8_DTYPE is torch.float32:
        return getattr(torch, "float8_e4m3fn", None)
    return None


@torch.library.custom_op("vllm_fl::flaggems_per_token_group_quant_fp8", mutates_args=())
def flaggems_per_token_group_quant_fp8(
    x: torch.Tensor, group_size: int, scale_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    import flag_gems

    return flag_gems.per_token_group_quant_fp8(
        x, group_size, dtype=_fp8_query_dtype(), scale_ue8m0=scale_ue8m0
    )


@flaggems_per_token_group_quant_fp8.register_fake
def _fake_query_quant(x, group_size, scale_ue8m0):
    from flag_gems.ops.per_token_group_quant_fp8 import SUPPORTED_FP8_DTYPE

    assert group_size > 0 and x.shape[-1] % group_size == 0
    assert x.stride(-1) == 1
    quantized = torch.empty_like(x, dtype=_fp8_query_dtype() or SUPPORTED_FP8_DTYPE)
    scales = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )
    return quantized, scales
