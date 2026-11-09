# SPDX-License-Identifier: Apache-2.0
"""Opaque compile boundary for the public FlagGems query quantization API."""

import torch


@torch.library.custom_op("vllm_fl::flaggems_per_token_group_quant_fp8", mutates_args=())
def flaggems_per_token_group_quant_fp8(
    x: torch.Tensor, group_size: int, scale_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    import flag_gems

    return flag_gems.per_token_group_quant_fp8(x, group_size, scale_ue8m0=scale_ue8m0)


@flaggems_per_token_group_quant_fp8.register_fake
def _fake_query_quant(x, group_size, scale_ue8m0):
    from flag_gems.ops.per_token_group_quant_fp8 import SUPPORTED_FP8_DTYPE

    assert group_size > 0 and x.shape[-1] % group_size == 0
    assert x.stride(-1) == 1
    quantized = torch.empty_like(x, dtype=SUPPORTED_FP8_DTYPE)
    scales = torch.empty(
        x.shape[:-1] + (x.shape[-1] // group_size,),
        device=x.device,
        dtype=torch.float32,
    )
    return quantized, scales
