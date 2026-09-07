# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent packed metadata buffers for multi-KV-group input batches.

``MultiGroupBlockTable.commit_block_table`` performs one CPU->device copy per
KV-cache group.  The groups are logically independent, but their block tables
have the same request dimension.  This helper aliases the group buffers to a
single, fixed-address arena, so a step can copy all active rows with one DMA.

The helper deliberately does not implement dirty-row tracking.  Scheduler
operations update the CPU ``.np`` views through several paths (add, clear,
move, and swap), and attention metadata may also write padded device rows
directly.  Re-copying the complete active prefix is the conservative contract:
it cannot retain a stale row when a request changes ownership or group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class _SavedGroupBuffers:
    block_cpu: torch.Tensor
    block_gpu: torch.Tensor
    block_np: Any
    slot_cpu: torch.Tensor
    slot_gpu: torch.Tensor
    slot_np: Any


class PackedBlockTableArena:
    """Alias all KV-group metadata buffers to one persistent arena.

    The public attributes are intentionally small and duck-typed so this class
    can be used with vLLM's ``MultiGroupBlockTable`` and with lightweight test
    doubles.  ``block_table`` remains the vLLM object; callers continue to use
    its normal group indexing and metadata builders.
    """

    def __init__(
        self,
        block_table: Any,
        *,
        device: torch.device | None = None,
        pin_memory: bool = False,
    ) -> None:
        groups = list(block_table.block_tables)
        if not groups:
            raise ValueError("cannot pack an empty MultiGroupBlockTable")

        first_block = groups[0].block_table
        if device is None:
            device = first_block.gpu.device
        self.device = torch.device(device)
        self.block_table = block_table
        self.groups = groups
        self.group_count = len(groups)
        self.num_rows = int(first_block.cpu.shape[0])

        block_widths: list[int] = []
        block_dtype = first_block.cpu.dtype
        for group in groups:
            buf = group.block_table
            if buf.cpu.ndim != 2 or buf.gpu.ndim != 2:
                raise ValueError("block-table buffers must be rank-2")
            if int(buf.cpu.shape[0]) != self.num_rows:
                raise ValueError("KV groups disagree on max_num_reqs")
            if buf.cpu.dtype != block_dtype or buf.gpu.dtype != block_dtype:
                raise ValueError("KV groups disagree on block-table dtype")
            if buf.gpu.device.type != self.device.type or (
                self.device.index is not None
                and buf.gpu.device.index != self.device.index
            ):
                raise ValueError("block-table device does not match requested device")
            block_widths.append(int(buf.cpu.shape[1]))

        self.block_widths = tuple(block_widths)
        self.block_offsets = tuple(
            [0]
            + [sum(block_widths[:idx]) for idx in range(1, self.group_count)]
        )
        self.total_block_width = sum(block_widths)
        if self.total_block_width <= 0:
            raise ValueError("block-table arena has no columns")

        max_slot_tokens = 0
        slot_dtype = groups[0].slot_mapping.cpu.dtype
        for group in groups:
            slot_buf = group.slot_mapping
            if slot_buf.cpu.ndim != 1 or slot_buf.gpu.ndim != 1:
                raise ValueError("slot-mapping buffers must be rank-1")
            if slot_buf.cpu.dtype != slot_dtype or slot_buf.gpu.dtype != slot_dtype:
                raise ValueError("KV groups disagree on slot-mapping dtype")
            if slot_buf.gpu.device.type != self.device.type or (
                self.device.index is not None
                and slot_buf.gpu.device.index != self.device.index
            ):
                raise ValueError("slot-mapping device does not match requested device")
            max_slot_tokens = max(max_slot_tokens, int(slot_buf.cpu.numel()))
        if max_slot_tokens <= 0:
            raise ValueError("slot-mapping arena has no elements")

        # Keep the original buffer objects alive: close() restores their exact
        # tensor and numpy objects, including any non-standard CpuGpuBuffer
        # implementation supplied by a vendor backend.
        self._saved: list[_SavedGroupBuffers] = []
        self.block_table_cpu = torch.empty(
            (self.num_rows, self.total_block_width),
            dtype=block_dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.block_table_gpu = torch.empty(
            (self.num_rows, self.total_block_width),
            dtype=block_dtype,
            device=self.device,
        )
        self.slot_mapping_cpu = torch.empty(
            (self.group_count, max_slot_tokens),
            dtype=slot_dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.slot_mapping_gpu = torch.empty(
            (self.group_count, max_slot_tokens),
            dtype=slot_dtype,
            device=self.device,
        )

        # Preserve any state produced before the arena was installed.  In the
        # normal runner this is all zeroes, but preserving it makes rebinds and
        # test doubles deterministic.
        for group_idx, group in enumerate(groups):
            block_buf = group.block_table
            slot_buf = group.slot_mapping
            self._saved.append(
                _SavedGroupBuffers(
                    block_cpu=block_buf.cpu,
                    block_gpu=block_buf.gpu,
                    block_np=block_buf.np,
                    slot_cpu=slot_buf.cpu,
                    slot_gpu=slot_buf.gpu,
                    slot_np=slot_buf.np,
                )
            )
            block_start = self.block_offsets[group_idx]
            block_end = block_start + self.block_widths[group_idx]
            self.block_table_cpu[:, block_start:block_end].copy_(block_buf.cpu)
            self.block_table_gpu[:, block_start:block_end].copy_(block_buf.gpu)
            slot_count = int(slot_buf.cpu.numel())
            self.slot_mapping_cpu[group_idx, :slot_count].copy_(slot_buf.cpu)
            self.slot_mapping_gpu[group_idx, :slot_count].copy_(slot_buf.gpu)

        # Metadata consumed by the 2-D producer.  They are persistent device
        # tensors and therefore safe as graph inputs; no host values are read
        # from inside a CUDA graph replay.
        self.block_offsets_gpu = torch.tensor(
            self.block_offsets, dtype=torch.int32, device=self.device
        )
        self.block_widths_gpu = torch.tensor(
            self.block_widths, dtype=torch.int32, device=self.device
        )
        self.block_sizes_gpu = torch.tensor(
            [int(group.block_size) for group in groups],
            dtype=torch.int32,
            device=self.device,
        )
        self.total_cp_world_sizes_gpu = torch.tensor(
            [
                int(group.pcp_world_size) * int(group.dcp_world_size)
                for group in groups
            ],
            dtype=torch.int32,
            device=self.device,
        )
        self.total_cp_ranks_gpu = torch.tensor(
            [
                int(group.pcp_rank) * int(group.dcp_world_size)
                + int(group.dcp_rank)
                for group in groups
            ],
            dtype=torch.int32,
            device=self.device,
        )
        self.cp_interleave_sizes_gpu = torch.tensor(
            [int(group.cp_kv_cache_interleave_size) for group in groups],
            dtype=torch.int32,
            device=self.device,
        )
        self.slot_mapping_group_stride = max_slot_tokens

        # Install fixed-shape views.  ``.np`` is deliberately regenerated from
        # the CPU view instead of retaining the old base array; all scheduler
        # mutations now update the packed source that commit() copies.
        try:
            for group_idx, group in enumerate(groups):
                block_start = self.block_offsets[group_idx]
                block_end = block_start + self.block_widths[group_idx]
                block_buf = group.block_table
                slot_buf = group.slot_mapping
                block_cpu = self.block_table_cpu[:, block_start:block_end]
                block_gpu = self.block_table_gpu[:, block_start:block_end]
                slot_count = int(slot_buf.cpu.numel())
                slot_cpu = self.slot_mapping_cpu[group_idx, :slot_count]
                slot_gpu = self.slot_mapping_gpu[group_idx, :slot_count]
                block_buf.cpu = block_cpu
                block_buf.gpu = block_gpu
                block_buf.np = block_cpu.numpy()
                slot_buf.cpu = slot_cpu
                slot_buf.gpu = slot_gpu
                slot_buf.np = slot_cpu.numpy()
        except Exception:
            # Keep construction transactional.  A vendor buffer may reject a
            # view assignment after one earlier group has already accepted it;
            # falling back must not leave a half-aliased table behind.
            for group, saved in zip(groups, self._saved):
                group.block_table.cpu = saved.block_cpu
                group.block_table.gpu = saved.block_gpu
                group.block_table.np = saved.block_np
                group.slot_mapping.cpu = saved.slot_cpu
                group.slot_mapping.gpu = saved.slot_gpu
                group.slot_mapping.np = saved.slot_np
            raise

        # common_slot_mapping.py discovers this object without importing it,
        # avoiding a model-runner <-> producer import cycle.
        block_table._packed_block_table_arena = self
        self.commit_calls = 0
        self.last_num_reqs: int | None = None
        self.closed = False

    @property
    def packed_block_table_stride(self) -> int:
        return int(self.block_table_gpu.stride(0))

    def commit(self, num_reqs: int) -> None:
        """Copy the active request prefix in one non-blocking DMA."""
        if self.closed:
            raise RuntimeError("packed block-table arena is closed")
        num_reqs = int(num_reqs)
        if num_reqs < 0 or num_reqs > self.num_rows:
            raise ValueError(f"num_reqs={num_reqs} outside [0, {self.num_rows}]")
        self.block_table_gpu[:num_reqs].copy_(
            self.block_table_cpu[:num_reqs], non_blocking=True
        )
        self.commit_calls += 1
        self.last_num_reqs = num_reqs

    def close(self) -> None:
        """Restore the original vLLM buffers and detach from the table."""
        if self.closed:
            return
        # close() is a lifecycle boundary, not a per-step operation.  Drain
        # outstanding H2D/producer work before copying the device arena back
        # to the original buffers so a rebind cannot observe a partial row.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        for group_idx, (group, saved) in enumerate(zip(self.groups, self._saved)):
            block_buf = group.block_table
            slot_buf = group.slot_mapping
            block_start = self.block_offsets[group_idx]
            block_end = block_start + self.block_widths[group_idx]
            slot_count = int(saved.slot_cpu.numel())

            # Rebinds and fallback paths may close an arena while the logical
            # table is still in use.  Preserve the latest state in both the
            # CPU scheduler source and the device metadata buffer before
            # restoring the original objects.  Normal shutdown calls this
            # after graph cleanup/synchronization; rebinds use the same
            # stream-ordered copies before the old table is discarded.
            saved.block_cpu.copy_(
                self.block_table_cpu[:, block_start:block_end]
            )
            saved.block_gpu.copy_(
                self.block_table_gpu[:, block_start:block_end]
            )
            saved.slot_cpu.copy_(self.slot_mapping_cpu[group_idx, :slot_count])
            saved.slot_gpu.copy_(self.slot_mapping_gpu[group_idx, :slot_count])
            block_buf.cpu = saved.block_cpu
            block_buf.gpu = saved.block_gpu
            block_buf.np = saved.block_np
            slot_buf.cpu = saved.slot_cpu
            slot_buf.gpu = saved.slot_gpu
            slot_buf.np = saved.slot_np
        if getattr(self.block_table, "_packed_block_table_arena", None) is self:
            delattr(self.block_table, "_packed_block_table_arena")
        self.closed = True

    def __enter__(self) -> "PackedBlockTableArena":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
