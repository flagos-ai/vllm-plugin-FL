# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vLLM v0.32.0:
# https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu/sample/gumbel.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""V2 temperature scaling with an immutable source and original output view."""

from vllm.triton_utils import tl, triton


@triton.jit
def _temperature_kernel(
    input_logits_ptr,
    input_logits_stride,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(
        input_logits_ptr + token_idx * input_logits_stride + block, mask=mask
    )
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)
