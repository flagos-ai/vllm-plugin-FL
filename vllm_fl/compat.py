# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Compatibility shims for vLLM symbols that upstream removed/renamed.

These functions lived in the FlagOS fork of vLLM (in
``vllm.v1.worker.gpu.attn_utils``) but were dropped from upstream vLLM 0.32.
They are self-contained and are re-exported here so ``vllm_fl`` can keep
working against an upstream ``v0.32.0`` build.
"""

from __future__ import annotations

from math import prod

import torch

from vllm.utils.torch_utils import get_dtype_size


def _reshape_attention_kv_cache(
    kv_raw_tensor: torch.Tensor,
    kv_cache_spec,
    kv_cache_shape: tuple[int, ...],
    kv_cache_stride_order: tuple[int, ...],
    num_blocks: int,
    packing: tuple[int, int] | None,
) -> torch.Tensor:
    """Reshape a raw KV-cache tensor into the attention backend's layout.

    Mirrors ``_reshape_attention_kv_cache`` from the FlagOS vLLM fork
    (``vllm/v1/worker/gpu/attn_utils.py``).
    """
    permuted_kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
    inv_order = [
        kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
    ]
    dtype = kv_cache_spec.dtype

    if packing is not None:
        offset, block_stride = packing
        assert inv_order[0] == 0
        page_bytes = prod(kv_cache_shape[1:]) * get_dtype_size(dtype)
        kv_cache = (
            kv_raw_tensor.view(-1, block_stride)[:, offset : offset + page_bytes]
            .view(dtype)
            .view(kv_cache_shape)
        )
    elif getattr(kv_cache_spec, "page_size_padded", None) is not None:
        # Use a strided view to skip the padding between physical pages.
        assert kv_cache_shape[0] == num_blocks, (
            "Padded KV pages require a num-blocks-first KV cache layout (got "
            f"shape {kv_cache_shape} with num_blocks={num_blocks}); "
            "kv-first layouts are not supported."
        )
        dtype_size = get_dtype_size(kv_cache_spec.dtype)
        page_stride = kv_cache_spec.page_size_bytes // dtype_size

        num_blocks_dim = inv_order[0]
        strides = list(torch.empty(permuted_kv_cache_shape).stride())
        strides[num_blocks_dim] = page_stride

        kv_cache = torch.as_strided(
            kv_raw_tensor.view(dtype),
            size=permuted_kv_cache_shape,
            stride=tuple(strides),
        )
    else:
        # No padding — safe to use a contiguous view.
        kv_cache = kv_raw_tensor.view(dtype).view(permuted_kv_cache_shape)

    return kv_cache.permute(*inv_order)
