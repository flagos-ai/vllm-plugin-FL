# GCU300 reshape_and_cache_flash: PyTorch native implementation.
#
# The upstream vLLM triton_reshape_and_cache_flash kernel has GCU
# compatibility issues:
#   1. torch.cuda.get_device_capability() crashes (not CUDA)
#   2. tl.load(slot_mapping).to(tl.int64) — int64 in triton kernel
#      requires TORCH_GCU_ENABLE_INT64_AND_UINT64=1 env var, and even
#      then the int64 division/modulo inside the kernel may not work
#      correctly on GCU300.
#
# The PyTorch native implementation below is correct, portable, and
# CUDA Graph compatible (no data-dependent control flow, no dynamic
# shapes, no CPU-GPU synchronization).
#
# When flag_gems is enabled (USE_FLAGGEMS=1), ATen operators (clamp,
# div, remainder, index, etc.) are replaced by Triton kernels that
# support int64 with ENABLE_I64_CHECK=0 env var.

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def reshape_and_cache_flash(
    key: torch.Tensor,         # [num_tokens, num_kv_heads, head_size]
    value: torch.Tensor,       # [num_tokens, num_kv_heads, head_size]
    key_cache: torch.Tensor,   # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor, # [num_blocks, block_size, num_kv_heads, head_size]
    slot_mapping: torch.Tensor,  # [num_tokens]  int64
    kv_cache_dtype: str,       # e.g. "auto", "fp8"  (unused for non-quantized)
    k_scale: torch.Tensor,     # scalar or per-token scale  (unused for non-quantized)
    v_scale: torch.Tensor,     # scalar or per-token scale  (unused for non-quantized)
) -> None:
    """Write per-token K/V into the paged KV cache (PyTorch native).

    Writes each token's key/value tensor into the paged KV cache at the
    flat slot position given by *slot_mapping*.  Tokens whose slot is
    negative (padding / speculative-draft rejects) are redirected to
    block 0 (NULL_BLOCK_ID, reserved for padding in vLLM), which is
    a harmless no-op write.

    CUDA Graph compatible:
      - No data-dependent control flow (no if/return based on tensor values)
      - No dynamic shapes (no boolean masking)
      - No CPU-GPU synchronization (no .item()/.cpu()/.tolist())
    """
    block_size = key_cache.size(1)

    # Clamp negative slots to 0 (NULL_BLOCK_ID). Block 0 is reserved
    # for padding in vLLM, so writing padding K/V there is harmless.
    # This avoids boolean indexing (dynamic shape) and CPU-GPU sync.
    slots = torch.clamp(slot_mapping, min=0)

    # Decompose flat slot → (block_idx, offset_within_block)
    block_idx = slots // block_size
    block_offset = slots % block_size

    # In-place write into paged cache via advanced indexing
    key_cache[block_idx, block_offset] = key
    value_cache[block_idx, block_offset] = value
