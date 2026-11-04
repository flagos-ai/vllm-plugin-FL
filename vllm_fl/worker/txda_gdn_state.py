# Copyright (c) 2026 BAAI. All rights reserved.
"""Limit TXDA GDN state indexing to the selected rows of its real cache.

The upstream GDN core remains intact. Its cache gather/scatter otherwise uses
the TXDA advanced-index fallback, which transfers the whole padded allocation.
Only the exact state tensor and the audited leading-row pattern are adapted.
"""

import functools

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.logger import init_logger

logger = init_logger("vllm.fl.txda.gdn_state")


def _supports_row_copy(cache: torch.Tensor) -> bool:
    if (
        type(cache) is not torch.Tensor
        or cache.layout != torch.strided
        or cache.ndim != 4
        or cache.dtype not in (torch.float32, torch.bfloat16)
        or cache.requires_grad
    ):
        return False
    # Proven layouts have contiguous state rows and optional padding between
    # disjoint pages. Preserve the native path for overlapping/unknown views.
    row_size = 1
    for dimension in range(cache.ndim - 1, 0, -1):
        if cache.shape[dimension] <= 0:
            return False
        if cache.shape[dimension] > 1 and cache.stride(dimension) != row_size:
            return False
        row_size *= cache.shape[dimension]
    return cache.stride(0) >= row_size


def _validated_ids(cache: torch.Tensor, index: torch.Tensor) -> list[int]:
    rows = cache.shape[0]
    ids = index.detach().cpu().tolist()
    normalized = []
    for value in ids:
        if not -rows <= value < rows:
            raise IndexError(f"GDN state row index {value} is outside the cache")
        normalized.append(value + rows if value < 0 else value)
    return normalized


class TxdaStateCacheIndexMode(TorchDispatchMode):
    """Intercept only unique leading-index operations on one retained tensor."""

    def __init__(self, target: torch.Tensor, on_copy=None):
        super().__init__()
        self.target = target
        self.on_copy = on_copy
        self.gather_hits = 0
        self.scatter_hits = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        gather = func is torch.ops.aten.index.Tensor
        scatter = func is torch.ops.aten.index_put_.default
        cache = args[0] if args else kwargs.get("self")
        if not (gather or scatter) or cache is not self.target:
            return func(*args, **kwargs)
        if not _supports_row_copy(cache):
            return func(*args, **kwargs)
        indices = args[1] if len(args) > 1 else kwargs.get("indices")
        if not isinstance(indices, (tuple, list)) or len(indices) != 1:
            return func(*args, **kwargs)
        index = indices[0]
        if (
            type(index) is not torch.Tensor
            or index.ndim != 1
            or index.dtype not in (torch.int32, torch.int64)
            or (index.device.type != "cpu" and index.device != cache.device)
        ):
            return func(*args, **kwargs)
        accumulate = args[3] if len(args) > 3 else kwargs.get("accumulate", False)
        if scatter and accumulate:
            return func(*args, **kwargs)
        # Validate every row before any mutation. Duplicate IDs retain ATen's
        # own write-order behavior rather than assigning a new ordering here.
        ids = _validated_ids(cache, index)
        if len(set(ids)) != len(ids):
            return func(*args, **kwargs)
        if gather:
            output = torch.empty(
                (len(ids), *cache.shape[1:]), dtype=cache.dtype, device=cache.device
            )
            for position, row in enumerate(ids):
                output[position : position + 1].copy_(cache[row : row + 1])
            self.gather_hits += 1
            result = output
        else:
            values = args[2] if len(args) > 2 else kwargs.get("values")
            if type(values) is not torch.Tensor or values.requires_grad:
                return func(*args, **kwargs)
            # Python's single-row setter strips the leading size-one axis
            # before invoking index_put_. Restore only that exact row shape;
            # general broadcasting remains on the original ATen path.
            if len(ids) == 1 and tuple(values.shape) == tuple(cache.shape[1:]):
                values = values.unsqueeze(0)
            if (
                values.device != cache.device
                or values.dtype != cache.dtype
                or tuple(values.shape) != (len(ids), *cache.shape[1:])
                or not _supports_row_copy(values)
            ):
                return func(*args, **kwargs)
            # Even disjoint views of one allocation take the native route.
            # This preserves ATen overlap errors and avoids changing values
            # that a later copied row would still need to read.
            if (
                values.untyped_storage().data_ptr()
                == cache.untyped_storage().data_ptr()
            ):
                return func(*args, **kwargs)
            for position, row in enumerate(ids):
                cache[row : row + 1].copy_(values[position : position + 1])
            self.scatter_hits += 1
            result = cache
        if ids and self.on_copy is not None:
            self.on_copy("gather" if gather else "scatter", len(ids))
        return result


def _is_txda_state_cache(cache) -> bool:
    # Fake tensors and tensor subclasses must preserve their original dispatch.
    return type(cache) is torch.Tensor and cache.device.type == "txda"


def patch_txda_gdn_state() -> bool:
    """Wrap only the upstream core, without replacing its implementation."""
    from vllm.platforms import current_platform

    if current_platform.device_type != "txda":
        return False
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    original = QwenGatedDeltaNetAttention._forward_core
    if getattr(original, "_fl_txda_state_index", False):
        return True

    @functools.wraps(original)
    def forward_core(self, *args, **kwargs):
        caches = getattr(self, "kv_cache", None)
        cache = (
            caches[1] if isinstance(caches, (tuple, list)) and len(caches) > 1 else None
        )
        if not _is_txda_state_cache(cache):
            return original(self, *args, **kwargs)

        def report(operation, rows):
            if not getattr(self, "_fl_txda_state_index_reported", False):
                logger.info(
                    "TXDA GDN state indexing active: operation=%s device=%s "
                    "cache_shape=%s cache_strides=%s rows=%d",
                    operation,
                    cache.device,
                    tuple(cache.shape),
                    cache.stride(),
                    rows,
                )
                self._fl_txda_state_index_reported = True

        with TxdaStateCacheIndexMode(cache, on_copy=report):
            return original(self, *args, **kwargs)

    forward_core._fl_txda_state_index = True
    QwenGatedDeltaNetAttention._forward_core = forward_core
    return True
