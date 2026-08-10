# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
) -> None:
    """Merge partial attention outputs from prefix and suffix using LSE rescaling.

    Prefers mcoplib CUDA kernel when dtype/headdim constraints are met,
    falls back to Triton implementation otherwise.
    """
    if output.dtype not in (torch.float32, torch.half, torch.bfloat16):
        assert output_scale is not None, (
            f"output_scale is required when output is {output.dtype}"
        )

    def _supported_dtypes(t: torch.Tensor) -> bool:
        return t.dtype in (torch.float32, torch.half, torch.bfloat16)

    def _supported_headdim(t: torch.Tensor) -> bool:
        headdim = t.shape[2]
        if t.dtype == torch.float32:
            return headdim % 4 == 0
        return headdim % 8 == 0

    if _supported_dtypes(prefix_output) and _supported_headdim(prefix_output):
        from vllm._custom_ops import merge_attn_states as _cuda_merge

        return _cuda_merge(
            output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse,
            prefill_tokens_with_context,
            output_scale,
        )
    else:
        from vllm.v1.attention.ops.triton_merge_attn_states import (
            merge_attn_states as _triton_merge,
        )

        return _triton_merge(
            output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_lse,
            prefill_tokens_with_context,
            output_scale,
        )
