# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.ops.triton_merge_attn_states import (
    merge_attn_states as _triton_merge_attn_states,
)


def _supported_dtypes(o: torch.Tensor) -> bool:
    """Custom CUDA kernel does not support FP8 dtype."""
    return o.dtype in [torch.float32, torch.half, torch.bfloat16]


def _supported_headdim(o: torch.Tensor) -> bool:
    """Custom CUDA kernel load/store 128b(16 bytes) per memory issue.
    The headdim must be multiple of pack_size (float32 -> 4, half/bf16 -> 8).
    """
    headdim = o.shape[2]  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    if o.dtype == torch.float32:
        return headdim % 4 == 0
    return headdim % 8 == 0


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
) -> None:
    if _supported_dtypes(output) and _supported_headdim(output):
        try:
            from vllm._custom_ops import merge_attn_states as _native_merge

            return _native_merge(
                output, prefix_output, prefix_lse,
                suffix_output, suffix_lse, output_lse
            )
        except (ImportError, AttributeError):
            pass

    return _triton_merge_attn_states(
        output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse
    )
