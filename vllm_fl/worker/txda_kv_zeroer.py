# Copyright (c) 2026 BAAI. All rights reserved.
"""Zero upstream V2 KV segments through tensor views on TXDA."""

from collections.abc import Iterable
from itertools import product
from numbers import Integral
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger("vllm.fl.txda.kv_zeroer")


class TxdaKVBlockZeroer:
    """Keep vLLM 0.32's segment geometry without indirect device pointers."""

    def __init__(
        self,
        device: torch.device,
        attn_groups_iter: Iterable,
        kernel_block_sizes: list[int],
        static_forward_context: dict[str, Any],
        num_blocks: int,
        runner_only_attn_layers: set[str] | None = None,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("KV zeroing requires a positive number of blocks")
        self.device = device
        self.num_blocks = num_blocks
        self._segments: list[torch.Tensor] = []
        excluded = runner_only_attn_layers or set()
        # Addresses identify aliases only; tensor views retain the real storage.
        seen: dict[int, tuple[int, int, int]] = {}

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if not isinstance(spec, AttentionSpec):
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
            if kernel_bs <= 0 or spec.block_size % kernel_bs:
                raise ValueError("Manager block size must divide into kernel blocks")
            for layer_name in group.layer_names:
                if layer_name in excluded:
                    continue
                kv = static_forward_context[layer_name].kv_cache
                if not isinstance(kv, torch.Tensor):
                    continue
                if group.kernel_block_stride:
                    kv = group.map_kv_cache_to_manager_blocks(kv, num_blocks)
                if kv.device.type != device.type:
                    continue
                if kv.shape[0] == 0 or kv.shape[0] % num_blocks:
                    raise ValueError(
                        f"{layer_name}: kernel blocks must be a positive multiple "
                        "of manager blocks"
                    )
                ratio = kv.shape[0] // num_blocks
                element_size = kv.element_size()
                block_stride_bytes = kv.stride(0) * element_size
                if block_stride_bytes % 4:
                    raise ValueError("KV block stride must be aligned to four bytes")
                outer_dims = [
                    d
                    for d in range(1, kv.ndim)
                    if kv.stride(d) * element_size > block_stride_bytes
                ]
                inner_dims = [d for d in range(1, kv.ndim) if d not in outer_dims]
                page_bytes = element_size + sum(
                    (kv.shape[d] - 1) * kv.stride(d) * element_size for d in inner_dims
                )
                if page_bytes % 4:
                    raise ValueError("KV page span must be aligned to four bytes")
                logical_stride_bytes = block_stride_bytes * ratio
                base_address = kv.data_ptr()
                for outer in product(*(range(kv.shape[d]) for d in outer_dims)):
                    outer_offset = sum(
                        index * kv.stride(d) for index, d in zip(outer, outer_dims)
                    )
                    for virtual_index in range(ratio):
                        offset = outer_offset + virtual_index * kv.stride(0)
                        address = base_address + offset * element_size
                        if address % 4:
                            raise ValueError("KV segment must be aligned to four bytes")
                        previous = seen.get(address)
                        if previous is not None:
                            index, old_stride, old_span = previous
                            if old_stride != logical_stride_bytes:
                                raise ValueError(
                                    "Aliased KV segments have different strides"
                                )
                            if old_span >= page_bytes:
                                continue
                        else:
                            index = len(self._segments)
                        segment = kv.as_strided(
                            (num_blocks, page_bytes // element_size),
                            (kv.stride(0) * ratio, 1),
                            storage_offset=kv.storage_offset() + offset,
                        )
                        if previous is None:
                            self._segments.append(segment)
                        else:
                            self._segments[index] = segment
                        seen[address] = (index, logical_stride_bytes, page_bytes)

        logger.info(
            "TXDA KV zeroer initialized: device=%s segments=%d num_blocks=%d",
            device,
            len(self._segments),
            num_blocks,
        )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero selected logical blocks after validating the whole ID list."""
        validated = []
        for block_id in block_ids:
            if isinstance(block_id, bool) or not isinstance(block_id, Integral):
                raise TypeError("KV block IDs must be CPU integers")
            block_id = int(block_id)
            if not 0 <= block_id < self.num_blocks:
                raise IndexError(f"KV block ID {block_id} is outside the cache")
            validated.append(block_id)
        for block_id in dict.fromkeys(validated):
            for segment in self._segments:
                segment[block_id : block_id + 1].zero_()

    def warmup(self, num_kv_blocks: int) -> None:
        if num_kv_blocks > 0:
            self.zero_block_ids([0])


def patch_txda_v2_kv_zeroer(runner_module) -> bool:
    """Bind the zeroer in the real V2 module, leaving the upstream class intact."""
    from vllm.platforms import current_platform

    if current_platform.device_type != "txda":
        return False
    runner_module.KVBlockZeroer = TxdaKVBlockZeroer
    return True
