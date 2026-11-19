# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve native MoE reduction while avoiding illegal CUDA grid-Y candidates.

FlagGems-vllm ``ops/moe_sum.py`` at pinned b5c2c224 uses grid Y =
ceil(num_tokens / TOKENS) and autotunes candidates with TOKENS=1. CUDA's Y
limit is 65535. Balance large batches into legal row views, retaining the
original reducer, FP32 expert accumulation order, dtype, strides and output.
Every large-batch slice has at least 32768 rows, so its original MT reduction
path/token bucket is retained, rather than switching a tiny final tail to the
pairwise reducer. This wrapper changes neither EP dispatch nor routing weights.
"""

CUDA_GRID_Y_MAX = 65535


def _balanced_row_ranges(num_tokens):
    count = (num_tokens + CUDA_GRID_Y_MAX - 1) // CUDA_GRID_Y_MAX
    base, remainder = divmod(num_tokens, count)
    start = 0
    for index in range(count):
        end = start + base + (index < remainder)
        yield start, end
        start = end


def safe_moe_sum(gems, input, output):
    """Write native reduction into the supplied output, using metadata-only views.

    As for the original reducer, warm up each shape family before CUDA graph
    capture. No host read of CUDA data or additional tensor storage is needed.
    """
    # Delegate malformed dimensions too, preserving the upstream validation.
    if input.dim() != 3 or output.dim() != 2:
        return gems.moe_sum(input, output)
    num_tokens = input.shape[0]
    if num_tokens <= CUDA_GRID_Y_MAX:
        return gems.moe_sum(input, output)
    # Validate global shape before slicing, so an oversized output cannot be
    # silently accepted when each individual row view has the expected shape.
    assert tuple(output.shape) == (num_tokens, input.shape[2]), (
        f"moe_sum: output shape {tuple(output.shape)} mismatch, "
        f"expected ({num_tokens}, {input.shape[2]}) from input shape"
    )
    for start, end in _balanced_row_ranges(num_tokens):
        gems.moe_sum(input[start:end], output[start:end])


def masked_moe_sum(gems, input, route_mask, output):
    """Fuse dummy-route masking into the same pinned FP32 accumulation order."""
    if (
        input.is_cuda
        and input.dim() == 3
        and output.dim() == 2
        and input.is_contiguous()
        and output.is_contiguous()
        and route_mask.is_contiguous()
        and tuple(route_mask.shape) == tuple(input.shape[:2])
        and input.shape[1] <= 16
        and input.shape[1] >= 1
        and input.shape[2] % (16 // input.element_size()) == 0
    ):
        from .mxfp4_marlin_masked_reduce import masked_sum

        assert tuple(output.shape) == (input.shape[0], input.shape[2])
        assert (
            input.dtype == output.dtype
            and input.device == output.device == route_mask.device
        )
        import torch

        assert route_mask.dtype == torch.bool
        assert input.dtype in (torch.bfloat16, torch.float16, torch.float32)
        return masked_sum(input, route_mask, output)
    input.masked_fill_(~route_mask.unsqueeze(-1), 0)
    return safe_moe_sum(gems, input, output)
