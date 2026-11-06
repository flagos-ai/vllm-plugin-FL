# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-Ascend v0.24.0rc1 hybrid cache configuration and views.
"""Dense Ascend hybrid cache views over the scheduler's shared byte pools.

Each pool stores all convolution states, then equally sized SSM/K slabs and
the V slab. Cache groups retain ownership of their scheduler-assigned block
IDs; the views must not allocate independent storage or change those IDs.
"""

import math
from dataclasses import replace

import torch

from vllm.model_executor.models import ModelRegistry
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)


def uses_dense_hybrid_cache(model_config) -> bool:
    """Limit this layout to the Qwen3.5/3.6 GDN family validated here."""
    return model_config.is_hybrid and getattr(
        model_config.hf_text_config, "model_type", None
    ) in ("qwen3_5_text", "qwen3_5_moe_text")


def align_hybrid_block_size(vllm_config) -> None:
    """Align at the upstream deferred platform hook, before cache planning."""
    cache = vllm_config.cache_config
    model = vllm_config.model_config
    model_cls, _ = ModelRegistry.resolve_model_cls(
        model.architecture, model_config=model
    )
    shapes = model_cls.get_mamba_state_shape_from_config(vllm_config)
    dtypes = model_cls.get_mamba_state_dtype_from_config(vllm_config)
    sizes = [math.prod(s) * get_dtype_size(d) for s, d in zip(shapes, dtypes)]
    # This layout uses convolution followed by the recurrent state. Other
    # state families need their own physical layout, not a guessed byte split.
    if model.use_mla or len(sizes) != 2 or sizes[0] >= sizes[1]:
        raise NotImplementedError(
            "Ascend dense hybrid cache requires convolution + GDN state"
        )
    dtype = (
        model.dtype
        if cache.cache_dtype == "auto"
        else STR_DTYPE_TO_TORCH_DTYPE[cache.cache_dtype]
    )
    key_token_bytes = (
        model.get_head_size()
        * model.get_num_kv_heads(vllm_config.parallel_config)
        * get_dtype_size(dtype)
    )
    block_size = 128 * cdiv(sizes[1], 128 * key_token_bytes)
    if block_size * key_token_bytes != sizes[1]:
        raise ValueError("Cannot align Ascend key and GDN state pages")
    if cache.user_specified_block_size and cache.block_size > block_size:
        raise ValueError(
            "Ascend dense hybrid cache requires block_size="
            + str(block_size)
            + " to preserve shared K/SSM block ownership"
        )
    cache.block_size = block_size
    cache.mamba_page_size_padded = 2 * sizes[1] + sizes[0]
    cache.mamba_block_size = (
        block_size if cache.mamba_cache_mode == "align" else model.max_model_len
    )


def pad_attention_specs(specs, page_size: int):
    """Use the global hybrid page size, including on attention-only PP stages."""
    return {
        name: replace(spec, page_size_padded=page_size)
        if isinstance(spec, AttentionSpec) and spec.page_size_bytes < page_size
        else spec
        for name, spec in specs.items()
    }


def dense_attention_view(raw, spec, shape):
    """Keep planar K/V at the pool tail, without per-forward cache copies."""
    payload = math.prod(shape) * get_dtype_size(spec.dtype)
    offset = raw.nbytes - payload
    if raw.element_size() != 1 or offset < 0 or offset % get_dtype_size(spec.dtype):
        raise ValueError("Invalid Ascend hybrid attention backing")
    return raw[offset:].view(spec.dtype).view(shape)


def dense_mamba_views(raw, shapes, dtypes, num_blocks):
    """Group each state type densely so NPU gathers only requested rows."""
    if raw.element_size() != 1:
        raise ValueError("Ascend hybrid cache requires byte backing")
    states = []
    offset = 0
    for shape, dtype in zip(shapes, dtypes):
        end = offset + num_blocks * math.prod(shape) * get_dtype_size(dtype)
        if offset % get_dtype_size(dtype) or end > raw.nbytes:
            raise ValueError("Invalid Ascend hybrid state backing")
        states.append(raw[offset:end].view(dtype).view(num_blocks, *shape))
        offset = end
    return states


def allocate_hybrid_kv_cache(
    config: KVCacheConfig,
    device: torch.device,
    attention_backends: dict,
    kernel_block_sizes: list[int],
    cache_dtype: str,
) -> dict[str, torch.Tensor]:
    if not config.kv_cache_tensors:
        return {}
    if any(tensor.host_resident for tensor in config.kv_cache_tensors):
        raise NotImplementedError(
            "Ascend hybrid KV cache requires device-resident pools"
        )
    size = max(tensor.size for tensor in config.kv_cache_tensors)
    backing = torch.zeros(size, dtype=torch.uint8, device=device)
    raw_caches = {}
    for tensor in config.kv_cache_tensors:
        for layer_index, layer_name in enumerate(tensor.layers):
            offset = tensor.offset + layer_index * tensor.layer_stride
            raw_caches[layer_name] = (tensor, offset)

    caches = {}
    for group_index, group in enumerate(config.kv_cache_groups):
        for layer_name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[layer_name]
            tensor, offset = raw_caches[layer_name]
            if tensor.block_stride != spec.page_size_bytes:
                raise ValueError("Ascend hybrid KV cache requires planar layer pages")
            num_blocks = config.num_blocks_of(tensor)
            byte_count = num_blocks * spec.page_size_bytes
            raw = backing[offset : offset + byte_count]
            if raw.numel() != byte_count:
                raise ValueError("Ascend hybrid KV cache layer exceeds backing storage")
            if isinstance(spec, AttentionSpec):
                kernel_block_size = kernel_block_sizes[group_index]
                if spec.block_size % kernel_block_size:
                    raise ValueError(
                        "Kernel block size must divide the cache block size"
                    )
                shape = attention_backends[layer_name].get_kv_cache_shape(
                    num_blocks * spec.block_size // kernel_block_size,
                    kernel_block_size,
                    spec.num_kv_heads,
                    spec.head_size,
                    cache_dtype,
                )
                caches[layer_name] = dense_attention_view(raw, spec, shape)
            elif isinstance(spec, MambaSpec):
                caches[layer_name] = raw.view(num_blocks, 1, 1, spec.page_size_bytes)
            else:
                raise TypeError(
                    f"Unsupported Ascend hybrid KV cache spec: {type(spec)}"
                )
    return caches
