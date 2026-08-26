# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: ignore-errors

"""Install vLLM-Ascend's Mamba state-copy support in upstream vLLM.

The upstream kernel uses a pointer layout that the Ascend Triton compiler
cannot lower for an aligned Mamba cache copy.  vLLM-Ascend supplies an Ascend
kernel and launches it with a larger block size; FL owns a local copy so the
runtime does not depend on the vllm-ascend package.
"""

from typing import Any

import torch
from vllm.v1.worker import mamba_utils

from vllm_fl.dispatch.backends.vendor.ascend.impl.triton.batch_memcpy import (
    batch_memcpy_kernel,
)


def _batch_memcpy_ascend(src_ptrs, dst_ptrs, sizes) -> None:
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    # Match vLLM-Ascend: 8192 improves copy throughput and avoids the upstream
    # kernel path that fails Ascend Triton lowering at a prefix-cache boundary.
    block_size = 8192
    batch_memcpy_kernel[grid](
        src_ptrs,
        dst_ptrs,
        sizes,
        BLOCK_SIZE=block_size,
    )


def _tensor_view_from_data_ptr(
    state: torch.Tensor,
    start_addr: int,
    num_elements: int,
) -> torch.Tensor:
    byte_offset = start_addr - state.data_ptr()
    element_size = state.element_size()
    if byte_offset < 0 or byte_offset % element_size != 0:
        raise RuntimeError("Invalid Mamba state copy pointer.")

    element_offset = byte_offset // element_size
    flat_state = state.view(-1)
    if element_offset + num_elements > flat_state.numel():
        raise RuntimeError("Mamba state copy range exceeds tensor storage.")
    return flat_state.narrow(0, element_offset, num_elements)


def _get_tensor_copy_pairs(
    copy_bufs: mamba_utils.MambaCopyBuffers,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if copy_bufs.offset == 0 or not hasattr(copy_bufs, "_tensor_copy_pairs"):
        copy_bufs._tensor_copy_pairs = []
    return copy_bufs._tensor_copy_pairs


def _collect_mamba_copy_meta_torch(
    copy_bufs: mamba_utils.MambaCopyBuffers,
    kv_cache_config,
    mamba_state_copy_funcs,
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    tensor_copy_pairs = _get_tensor_copy_pairs(copy_bufs)
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[
            mamba_group_id
        ].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            for state, state_copy_func in zip(
                kv_caches,
                mamba_state_copy_funcs,
            ):
                copy_spec = state_copy_func(
                    state,
                    block_ids,
                    src_block_idx,
                    accept_token_bias + 1,
                )
                src_state = _tensor_view_from_data_ptr(
                    state,
                    copy_spec.start_addr,
                    copy_spec.num_elements,
                )
                dst_state = _tensor_view_from_data_ptr(
                    state,
                    state[dest_block_id].data_ptr(),
                    copy_spec.num_elements,
                )
                tensor_copy_pairs.append((src_state, dst_state))
                sizes_np[offset] = (
                    copy_spec.num_elements * state.element_size()
                )
                offset += 1

    copy_bufs.offset = offset


def _do_mamba_copy_block_torch(
    copy_bufs: mamba_utils.MambaCopyBuffers,
) -> None:
    count = copy_bufs.offset
    if count == 0:
        if hasattr(copy_bufs, "_tensor_copy_pairs"):
            copy_bufs._tensor_copy_pairs = []
        return

    tensor_copy_pairs = getattr(copy_bufs, "_tensor_copy_pairs", None)
    if tensor_copy_pairs is None or len(tensor_copy_pairs) != count:
        raise RuntimeError("Mamba tensor copy metadata is incomplete.")

    for src_state, dst_state in tensor_copy_pairs:
        # clone keeps overlapping source/destination ranges well-defined.
        dst_state.copy_(src_state.clone())
    copy_bufs._tensor_copy_pairs = []


def apply_mamba_utils_patch() -> None:
    mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel
    mamba_utils.batch_memcpy = _batch_memcpy_ascend
