# Copyright (c) 2026 BAAI. All rights reserved.
"""Compatibility fixes for saving vLLM sharded-state checkpoints."""

import collections
import logging

import torch

logger = logging.getLogger(__name__)


def _filter_subtensors_stride_safe(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Deduplicate aliases without flattening non-contiguous tensors.

    vLLM 0.24's implementation calls ``tensor.view(-1)`` while inspecting
    storage bounds. DeepSeek-V4 contains non-contiguous parameter aliases, so
    saving a processed (Marlin) model fails before the first shard is written.
    """
    storage_groups: dict[tuple[torch.device, int], list[tuple[str, torch.Tensor]]] = (
        collections.defaultdict(list)
    )
    for key, tensor in tensors.items():
        if tensor.numel():
            storage_groups[(tensor.device, tensor.untyped_storage().data_ptr())].append(
                (key, tensor)
            )

    def storage_bounds(tensor: torch.Tensor) -> tuple[int, int]:
        start = tensor.data_ptr()
        max_element_offset = sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride())
            if size
        )
        return start, start + (max_element_offset + 1) * tensor.element_size()

    result: dict[str, torch.Tensor] = {}
    for group in storage_groups.values():
        for key, tensor in group:
            start, end = storage_bounds(tensor)
            for other_key, other in group:
                if not other.is_contiguous():
                    continue
                other_start, other_end = storage_bounds(other)
                if start < other_start or other_end < end:
                    continue
                if (
                    other_start < start
                    or end < other_end
                    or not tensor.is_contiguous()
                ):
                    break
                if other_key < key:
                    break
            else:
                # safetensors only accepts contiguous tensors. This copy is
                # needed only when no contiguous alias covers the same storage.
                result[key] = tensor if tensor.is_contiguous() else tensor.contiguous()
    return result


def patch_vllm_sharded_state_filter() -> bool:
    from vllm.model_executor.model_loader import ShardedStateLoader

    current = ShardedStateLoader._filter_subtensors
    if getattr(current, "_vllm_fl_stride_safe", False):
        return False
    _filter_subtensors_stride_safe._vllm_fl_stride_safe = True
    _filter_subtensors_stride_safe._vllm_fl_original = current
    ShardedStateLoader._filter_subtensors = staticmethod(
        _filter_subtensors_stride_safe
    )
    logger.info("Patched vLLM sharded-state filtering for strided tensors")
    return True


__all__ = ["patch_vllm_sharded_state_filter"]
