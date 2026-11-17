# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM-Ascend v0.24.0rc1 hybrid cache configuration and views.
"""Dense Ascend hybrid cache views over the scheduler's shared byte pools.

Each pool stores all convolution states, then equally sized SSM/K slabs and
the V slab. Cache groups retain ownership of their scheduler-assigned block
IDs; the views must not allocate independent storage or change those IDs.
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch

from vllm.model_executor.models import ModelRegistry
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
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


def _deduplicate_block_planes(planes):
    """Return byte views with block IDs at dim 0, retaining storage aliases."""
    unique = {}
    for plane in planes:
        byte_plane = plane.view(torch.uint8).view(plane.shape[0], -1)
        key = (
            byte_plane.device,
            byte_plane.data_ptr(),
            tuple(byte_plane.shape),
            tuple(byte_plane.stride()),
        )
        unique.setdefault(key, byte_plane)
    return list(unique.values())


def _layer_specs(kv_cache_config):
    specs = {}
    group_ids = {}
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            specs[name] = spec
            group_ids[name] = group_id
    return specs, group_ids


def _attention_kernel_view(manager, spec, kernel_block_size):
    """Map the manager's BHNC axes to native planar K/V without a copy."""
    if (
        manager.ndim != 4
        or manager.shape[1] != 2
        or manager.shape[2] != spec.block_size
        or manager.shape[3] != spec.num_kv_heads * spec.head_size
        or spec.tokens_per_state != 1
        or spec.head_size_v != spec.head_size
    ):
        raise NotImplementedError("Unsupported Ascend manager attention view")
    ratio, remainder = divmod(spec.block_size, kernel_block_size)
    if remainder or ratio < 1:
        raise ValueError("Ascend kernel block size must divide its manager block")
    return manager.permute(1, 0, 2, 3).view(
        2,
        manager.shape[0] * ratio,
        kernel_block_size,
        spec.num_kv_heads,
        spec.head_size,
    )


def allocate_ascend_kv_cache(
    kv_cache_config,
    device,
    layout,
    kernel_block_sizes,
    attn_groups,
    use_dense,
):
    """Allocate manager, native kernel, and block-operation views once.

    Dense hybrid slabs have different block strides for convolution and SSM.
    They deliberately have no fictitious single Mamba manager view. Their raw
    pages are only bind inputs for the Ascend GDN layer; block operations use
    the explicit byte planes returned here.
    """
    if not kv_cache_config.kv_cache_tensors:
        return {}, {}, []
    if any(tensor.host_resident for tensor in kv_cache_config.kv_cache_tensors):
        raise NotImplementedError("Ascend cache does not support host-resident pools")

    specs, group_ids = _layer_specs(kv_cache_config)
    has_attention = any(isinstance(spec, AttentionSpec) for spec in specs.values())
    has_mamba = any(isinstance(spec, MambaSpec) for spec in specs.values())
    expected_layout = "LBNHC" if use_dense else "LHBNC"
    if (use_dense or has_attention) and layout.name != expected_layout:
        raise NotImplementedError(
            f"Ascend {'dense hybrid' if use_dense else 'planar'} cache requires "
            f"{expected_layout}, got {layout.name}"
        )
    if not use_dense and has_attention and has_mamba:
        raise NotImplementedError(
            "Ascend mixed attention/Mamba caches require the dense GDN layout"
        )
    bind_views = {}
    manager_views = {}
    planes = []

    if not use_dense:
        from vllm.v1.worker.utils import allocate_kv_cache

        # LHBNC keeps K and V planar. Subdividing B before allocation would
        # fail upstream's dense-page check; split only the native kernel view.
        manager_views = allocate_kv_cache(
            kv_cache_config, device, layout, kernel_block_sizes=None
        )
        for name, manager in manager_views.items():
            spec = specs[name]
            if isinstance(spec, AttentionSpec):
                kernel_size = kernel_block_sizes[group_ids[name]]
                bind_views[name] = _attention_kernel_view(manager, spec, kernel_size)
                # Each plane is individually contiguous across its blocks;
                # flattening the whole BHNC manager would allocate a copy.
                planes.extend((manager[:, 0], manager[:, 1]))
            elif isinstance(spec, MambaSpec):
                bind_views[name] = manager
                planes.append(manager)
            else:
                raise NotImplementedError(f"Unsupported Ascend KV spec: {type(spec)}")
    else:
        sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
        if len(sizes) != 1:
            raise ValueError("Ascend dense tensors must share one backing allocation")
        backing = torch.zeros(sizes.pop(), dtype=torch.int8, device=device)
        pool_layouts = {}
        for tensor in kv_cache_config.kv_cache_tensors:
            num_blocks = kv_cache_config.num_blocks_of(tensor)
            for layer_index, name in enumerate(tensor.layers):
                spec = specs[name]
                page_bytes = spec.page_size_bytes
                pool_bytes = num_blocks * page_bytes
                if tensor.block_stride != page_bytes or (
                    len(tensor.layers) > 1 and tensor.layer_stride < pool_bytes
                ):
                    raise NotImplementedError(
                        "Ascend dense hybrid cache requires layer-compact pools"
                    )
                offset = tensor.offset + layer_index * tensor.layer_stride
                if offset < 0 or offset + pool_bytes > backing.numel():
                    raise ValueError("Ascend dense layer pool exceeds its allocation")
                raw = backing[offset : offset + pool_bytes]
                if isinstance(spec, AttentionSpec):
                    state_bytes = (
                        spec.block_size
                        * spec.num_kv_heads
                        * spec.head_size
                        * get_dtype_size(spec.dtype)
                    )
                    conv_bytes = page_bytes - 2 * state_bytes
                    if conv_bytes <= 0:
                        raise ValueError(
                            "Ascend dense attention requires a convolution prefix"
                        )
                    kernel_size = kernel_block_sizes[group_ids[name]]
                    ratio, remainder = divmod(spec.block_size, kernel_size)
                    if remainder or ratio < 1:
                        raise ValueError(
                            "Ascend kernel block size must divide its manager block"
                        )
                    shape = (
                        2,
                        num_blocks * ratio,
                        kernel_size,
                        spec.num_kv_heads,
                        spec.head_size,
                    )
                    kernel = dense_attention_view(raw, spec, shape)
                    bind_views[name] = kernel
                    manager_views[name] = kernel.view(
                        2, num_blocks, spec.block_size, -1
                    ).permute(1, 0, 2, 3)
                elif isinstance(spec, MambaSpec):
                    state_sizes = [
                        math.prod(shape) * get_dtype_size(dtype)
                        for shape, dtype in zip(spec.shapes, spec.dtypes)
                    ]
                    if len(state_sizes) != 2 or state_sizes[0] >= state_sizes[1]:
                        raise NotImplementedError(
                            "Ascend dense hybrid cache requires convolution + GDN state"
                        )
                    conv_bytes, state_bytes = state_sizes
                    if page_bytes != conv_bytes + 2 * state_bytes:
                        raise ValueError(
                            "Ascend dense GDN padding must match its K/V slabs"
                        )
                    bind_views[name] = raw.view(num_blocks, 1, 1, page_bytes)
                else:
                    raise NotImplementedError(
                        f"Unsupported Ascend KV spec: {type(spec)}"
                    )

                pool_layout = (pool_bytes, num_blocks, conv_bytes, state_bytes)
                for other_offset, other_layout in pool_layouts.items():
                    other_end = other_offset + other_layout[0]
                    if (
                        offset < other_end
                        and other_offset < offset + pool_bytes
                        and (offset != other_offset or pool_layout != other_layout)
                    ):
                        raise ValueError(
                            "Overlaid Ascend dense pools must share the same "
                            "convolution/SSM/V block layout"
                        )
                pool_layouts[offset] = pool_layout

                # Include the reserved V plane even for a Mamba-only PP stage.
                # Attention-only stages derive their convolution prefix above.
                conv_end = num_blocks * conv_bytes
                state_end = conv_end + num_blocks * state_bytes
                planes.extend(
                    (
                        raw[:conv_end].view(num_blocks, conv_bytes),
                        raw[conv_end:state_end].view(num_blocks, state_bytes),
                        raw[state_end:].view(num_blocks, state_bytes),
                    )
                )

    for group in attn_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, AttentionSpec) and group.kernel_block_size is not None:
            kernel_size = kernel_block_sizes[group.kv_cache_group_id]
            group.kernel_block_stride = spec.block_size // kernel_size
    return bind_views, manager_views, _deduplicate_block_planes(planes)


def copy_block_planes(planes, copies):
    """Snapshot every source row before writing through integer or byte views."""
    if not planes or not copies:
        return
    planes = _deduplicate_block_planes(planes)
    indices_by_device = {}
    snapshots = []
    for plane in planes:
        aligned = (
            plane.storage_offset() % 4 == 0
            and plane.stride(0) % 4 == 0
            and plane.shape[1] % 4 == 0
        )
        # Raw int32 preserves every bit while avoiding the NPU uint8
        # index_copy_ bug that corrupts untouched rows of offset views.
        target = plane.view(torch.int32) if aligned else plane
        if plane.device not in indices_by_device:
            indices_by_device[plane.device] = torch.tensor(
                copies, dtype=torch.long, device=plane.device
            ).unbind(dim=1)
        source, destination = indices_by_device[plane.device]
        snapshots.append((target, destination, target[source], aligned))
    for target, destination, snapshot, aligned in snapshots:
        if aligned:
            target.index_copy_(0, destination, snapshot)
        else:
            # Copy only the requested rows, never clone a whole cache plane.
            for row, (_, block_id) in enumerate(copies):
                target[block_id].copy_(snapshot[row])


def zero_block_planes(planes, block_ids):
    """Zero rows in place with supported NPU integer operations."""
    if not planes or not block_ids:
        return
    indices_by_device = {}
    for plane in _deduplicate_block_planes(planes):
        aligned = (
            plane.storage_offset() % 4 == 0
            and plane.stride(0) % 4 == 0
            and plane.shape[1] % 4 == 0
        )
        if aligned:
            if plane.device not in indices_by_device:
                indices_by_device[plane.device] = torch.tensor(
                    block_ids, dtype=torch.long, device=plane.device
                )
            plane.view(torch.int32).index_fill_(0, indices_by_device[plane.device], 0)
        else:
            zeros = torch.zeros(plane.shape[1], dtype=torch.uint8, device=plane.device)
            for block_id in block_ids:
                plane[block_id].copy_(zeros)
