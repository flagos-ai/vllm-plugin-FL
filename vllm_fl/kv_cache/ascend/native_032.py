# SPDX-License-Identifier: Apache-2.0
"""Ascend layer views over vLLM v0.32.0's native KV allocation.

The manager/connector keeps its original ``[B, H, N, C]`` tensors. Only the
views bound to Ascend layers change: ordinary attention consumes a K/V pair,
and Mamba consumes one typed tensor per state. No storage is copied or repacked.
See upstream ``vllm/v1/kv_cache_interface.py:create_kv_cache_views`` at v0.32.0
(``5df63a7203cdb9328d824a27b682e77c7b9928be``).
"""

from math import prod

import torch

from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode, MambaSpec


def _attention_views(cache: torch.Tensor, spec: AttentionSpec):
    if (
        spec.head_size_v != spec.head_size
        or spec.tokens_per_state != 1
        or spec.num_heads != spec.num_kv_heads
        or spec.state_content_bytes is not None
        or spec.kv_quant_mode != KVQuantMode.NONE
    ):
        raise NotImplementedError(
            "Ascend native KV binding requires ordinary, unquantized K/V "
            "attention with equal key/value head sizes"
        )
    if (
        cache.ndim != 4
        or cache.shape[1] != spec.num_kv_heads
        or cache.shape[-1] != 2 * spec.head_size
        or cache.dtype != spec.dtype
    ):
        raise ValueError(
            "Ascend attention cache must be a typed [B, H, N, 2 * head_size] "
            "view matching its AttentionSpec"
        )
    # transpose/split preserve the allocator's layer offset and all physical
    # strides, including padded or interleaved blocks and shared layer aliases.
    return cache.transpose(1, 2).split(spec.head_size, dim=-1)


def _mamba_views(cache: torch.Tensor, spec: MambaSpec):
    if (
        cache.dtype != torch.int8
        or cache.ndim != 4
        or cache.shape[1:3] != (1, 1)
        or cache.stride(-1) != 1
    ):
        raise ValueError("Ascend Mamba cache must be an int8 [B, 1, 1, C] page view")
    if not spec.shapes or len(spec.shapes) != len(spec.dtypes):
        raise ValueError("MambaSpec must contain matching, nonempty shapes and dtypes")

    pages = cache[:, 0, 0, :]
    states = []
    offset = 0
    for shape, dtype in zip(spec.shapes, spec.dtypes):
        if not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError("Mamba state dimensions must be positive integers")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("Mamba state dtypes must be torch.dtype values")
        itemsize = dtype.itemsize
        nbytes = prod(shape) * itemsize
        end = offset + nbytes
        if end > pages.shape[1]:
            raise ValueError("Mamba state shapes exceed the cache page payload")
        if (pages.storage_offset() + offset) % itemsize or pages.stride(0) % itemsize:
            raise ValueError(
                "Mamba state offset and block stride must align to its dtype"
            )
        # Slice within EACH page before reinterpreting bytes. Flattening the
        # allocation would consume padding/other layers between consecutive pages.
        state = pages[:, offset:end].view(dtype).view(cache.shape[0], *shape)
        # view() may normalize a singleton B stride. Retain the allocator's
        # explicit page stride even for a one-block pool.
        state = state.as_strided(
            state.shape, (pages.stride(0) // itemsize, *state.stride()[1:])
        )
        states.append(state)
        offset = end
    if offset != pages.shape[1]:
        raise ValueError("Mamba cache page payload does not match its state shapes")
    return tuple(states)


def adapt_layer_kv_caches(kernel_caches: dict, attn_groups: list) -> dict:
    """Return Ascend layer views without changing manager tensors or storage.

    ``attn_groups`` is the flat list of native 0.32 ``AttentionGroup`` objects;
    each has a concrete ``kv_cache_spec`` and ``layer_names``. Specifications
    with opaque/quantized attention storage need their own explicit adapter.
    """
    adapted = {}
    for group in attn_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, AttentionSpec):
            make_views = _attention_views
        elif isinstance(spec, MambaSpec):
            make_views = _mamba_views
        else:
            raise NotImplementedError(
                f"Ascend native KV binding does not support {type(spec).__name__}"
            )
        for name in group.layer_names:
            # Native groups may include encoder-only layers with no allocation.
            if name not in kernel_caches:
                continue
            if name in adapted:
                raise ValueError(f"Layer {name!r} occurs in multiple KV cache groups")
            cache = kernel_caches[name]
            if not isinstance(cache, torch.Tensor):
                raise TypeError(f"Layer {name!r} KV cache must be a native tensor view")
            adapted[name] = make_views(cache, spec)
    missing = kernel_caches.keys() - adapted.keys()
    if missing:
        raise ValueError(f"KV caches have no attention group: {sorted(missing)}")
    return adapted
