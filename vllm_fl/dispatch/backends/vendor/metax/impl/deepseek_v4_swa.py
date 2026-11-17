# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

# Adapted from vLLM sparse_swa for MetaX Triton.

from vllm.triton_utils import tl, triton


@triton.jit
def _compute_swa_indices_and_lens_kernel(
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    window_size,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if is_valid == 0:
        tl.store(swa_lens_ptr + token_idx, 0)
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    pos = prefix_len + token_idx - query_start
    start_pos = tl.maximum(pos - window_size + 1, 0)
    end_pos = pos + 1
    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + token_idx, swa_len)

    for i in range(0, window_size, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + token_idx * swa_indices_stride + offset,
            slot_ids,
            mask=offset < window_size,
        )


def apply_metax_swa_patch() -> None:
    from flag_gems.fused.flash_mla_with_kvcache import get_mla_metadata

    from vllm.v1.attention.backends.mla import sparse_swa

    sparse_swa._compute_swa_indices_and_lens_kernel = (
        _compute_swa_indices_and_lens_kernel
    )
    sparse_swa.get_mla_metadata = get_mla_metadata
