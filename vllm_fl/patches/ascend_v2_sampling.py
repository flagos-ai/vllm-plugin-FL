# Copyright (c) 2026 BAAI. All rights reserved.
"""Preserve V2 sampling transforms with bounded tiles and read-only logits.

The vLLM 0.32 CUDA launcher uses 8192 entries per tile, exceeding Ascend UB.
Its in-place kernel also reapplies the first vocabulary tile on large Ascend
inputs. A separate read-only logits source retains the upstream mathematics
while preserving the caller's output buffer and all penalty state.
"""

import importlib

import torch

_PENALTIES_MODULE = "vllm.v1.worker.gpu.sample.penalties"
_ASCEND_KERNEL_MODULE = "vllm_fl.patches.ascend_v2_penalties_kernel"
_TEMPERATURE_MODULE = "vllm.v1.worker.gpu.sample.gumbel"
_SAMPLING_STATES_MODULE = "vllm.v1.worker.gpu.sample.states"
_ASCEND_TEMPERATURE_KERNEL_MODULE = "vllm_fl.patches.ascend_v2_temperature_kernel"
_ASCEND_GUMBEL_MODULE = "vllm_fl.patches.ascend_v2_gumbel"
ASCEND_PENALTIES_BLOCK_SIZE = 1024
ASCEND_TEMPERATURE_BLOCK_SIZE = 1024


def apply_ascend_v2_penalties(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    """Apply upstream V2 penalties with independent input/output row strides."""
    num_tokens, vocab_size = logits.shape
    if num_tokens == 0 or vocab_size == 0:
        return
    penalties = importlib.import_module(_ASCEND_KERNEL_MODULE)
    # Allocate and launch on the current NPU stream. Keep the source alive
    # through enqueueing; the allocator orders its reuse after the kernel.
    source = logits.clone()
    block_size = ASCEND_PENALTIES_BLOCK_SIZE
    num_blocks = (vocab_size + block_size - 1) // block_size
    penalties._penalties_kernel[(num_tokens, num_blocks)](
        source,
        source.stride(0),
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        vocab_size,
        BLOCK_SIZE=block_size,
    )


def apply_ascend_v2_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    """Scale each row once, reading immutable input on the current stream."""
    num_tokens, vocab_size = logits.shape
    if num_tokens == 0 or vocab_size == 0:
        return
    kernels = importlib.import_module(_ASCEND_TEMPERATURE_KERNEL_MODULE)
    source = logits.clone()
    block_size = ASCEND_TEMPERATURE_BLOCK_SIZE
    num_blocks = (vocab_size + block_size - 1) // block_size
    kernels._temperature_kernel[(num_tokens, num_blocks)](
        source,
        source.stride(0),
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=block_size,
    )


def install_ascend_v2_sampling_shims() -> bool:
    """Replace NPU V2 transform launchers; keep the sampler and state intact."""
    from vllm.platforms import current_platform

    if current_platform.device_type != "npu":
        return False
    penalties = importlib.import_module(_PENALTIES_MODULE)
    penalties.apply_penalties = apply_ascend_v2_penalties
    temperature = importlib.import_module(_TEMPERATURE_MODULE)
    temperature.apply_temperature = apply_ascend_v2_temperature
    gumbel = importlib.import_module(_ASCEND_GUMBEL_MODULE)
    # Existing JIT callers resolve these globals at compilation, so importing
    # gumbel_noised_argmax earlier does not retain the unsigned-shift defect.
    for name in (
        "_murmur3_rotl32",
        "_murmur3_mix",
        "_murmur3_fmix32",
        "murmur3_hash32",
        "murmur3_uniform32",
        "_uniform64_from_random53",
        "murmur3_uniform64",
    ):
        setattr(temperature, name, getattr(gumbel, name))
    # SamplingStates imports this function directly at module initialization.
    states = importlib.import_module(_SAMPLING_STATES_MODULE)
    states.apply_temperature = apply_ascend_v2_temperature
    return True
