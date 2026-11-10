# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Triton-free KV block zeroing for the Kunlunxin XPU backend."""

import logging

import torch

logger = logging.getLogger(__name__)


def _collect_kv_segments(
    device,
    attn_groups,
    kernel_block_sizes,
    static_forward_context,
    num_blocks,
    runner_only_attn_layers,
):
    """Collect the block-first attention views bound by the vLLM 0.32 runner."""
    from vllm.v1.kv_cache_interface import AttentionSpec

    segments = []
    seen_views = set()
    for group in attn_groups:
        if not isinstance(group.kv_cache_spec, AttentionSpec):
            continue
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            continue
        for layer_name in group.layer_names:
            if layer_name in runner_only_attn_layers:
                continue
            kv = static_forward_context[layer_name].kv_cache
            if not isinstance(kv, torch.Tensor) or kv.device.type != device.type:
                continue
            if group.kernel_block_stride:
                kv = group.map_kv_cache_to_manager_blocks(kv, num_blocks)
            assert kv.shape[0] % num_blocks == 0
            view = (kv.data_ptr(), kv.dtype, kv.shape, kv.stride())
            # Packed overlays can have the same address and different spans.
            if view not in seen_views:
                seen_views.add(view)
                segments.append((kv, kv.shape[0] // num_blocks))
    return segments


def _zero_block_ids_pytorch(self, block_ids) -> None:
    if not block_ids or not self._kunlunxin_kv_segments:
        return
    ids = sorted({int(block_id) for block_id in block_ids})
    for kv, ratio in self._kunlunxin_kv_segments:
        # Span zeroing avoids both raw-address Triton and host-to-device
        # copies of index tensors, which fail on P800.
        run_start = None
        run_end = 0
        for block_id in ids:
            start = block_id * ratio
            if start < 0 or start + ratio > kv.shape[0]:
                continue
            if run_start is not None and start != run_end:
                kv.narrow(0, run_start, run_end - run_start).zero_()
                run_start = None
            if run_start is None:
                run_start = start
            run_end = start + ratio
        if run_start is not None:
            kv.narrow(0, run_start, run_end - run_start).zero_()


def apply_kv_block_zero_compat() -> None:
    """Install a copy-free zeroer with the upstream vLLM 0.32 constructor."""
    from vllm.v1.worker.utils import KVBlockZeroer

    if KVBlockZeroer.zero_block_ids is _zero_block_ids_pytorch:
        return

    def init_kunlunxin_zeroer(
        self,
        device,
        attn_groups_iter,
        kernel_block_sizes,
        static_forward_context,
        num_blocks,
        runner_only_attn_layers=None,
    ):
        self.device = device
        self._kunlunxin_kv_segments = _collect_kv_segments(
            device,
            attn_groups_iter,
            kernel_block_sizes,
            static_forward_context,
            num_blocks,
            runner_only_attn_layers or set(),
        )

    KVBlockZeroer.__init__ = init_kunlunxin_zeroer
    KVBlockZeroer.zero_block_ids = _zero_block_ids_pytorch
    logger.info("Installed the Kunlunxin KV block zeroer")
