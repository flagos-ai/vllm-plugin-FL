# Copyright (c) 2026 BAAI. All rights reserved.
"""Reference eager preprocessing for TXDA and upstream vLLM 0.32 V2.

The vendor compiler can silently miscompile the pointer-to-pointer staged
write and block-table kernels. Keep the upstream buffers and runner, but use
native tensor copies and CPU integer arithmetic for these small control paths.
This fallback deliberately synchronizes device metadata and supports CP=1 only.
"""

from collections.abc import Sequence

import torch


def _apply_staged_write(self) -> None:
    """Apply typed CPU staging lists without dereferencing device pointers."""
    if not self._staged_write_indices:
        return

    contents = torch.tensor(self._staged_write_contents, dtype=self.dtype, device="cpu")
    previous_end = 0
    for index, start, end in zip(
        self._staged_write_indices,
        self._staged_write_starts,
        self._staged_write_cu_lens,
        strict=True,
    ):
        if not 0 <= index < self.gpu.shape[0]:
            raise IndexError(f"Staged write row {index} is outside the tensor")
        # A one-row view also supports the 1D tensors used for request lengths.
        # view() fails rather than silently copying a non-contiguous target.
        row = self.gpu[index : index + 1].view(-1)
        length = end - previous_end
        if length < 0 or end > len(contents) or not 0 <= start <= row.numel() - length:
            raise IndexError("Staged write exceeds its contents or target row")
        row[start : start + length].copy_(contents[previous_end:end])
        previous_end = end

    # Leave the metadata available for retry if any tensor copy above fails.
    self.clear_staged_writes()


def _apply_fused_staged_writes(
    self,
    tensors: Sequence,
    output_ptrs: torch.Tensor,
    output_strides: torch.Tensor,
) -> None:
    """Use the real destination tensors, retaining the upstream pointer API."""
    # The pointer arrays describe tensors already available here. Reading or
    # reinterpreting those addresses would reintroduce the broken device path.
    for tensor in tensors:
        tensor.apply_write()


def _gather_block_tables(
    self,
    idx_mapping: torch.Tensor,
    num_reqs_padded: int,
    out: tuple[torch.Tensor, ...] | None = None,
    out_ptrs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    if self.num_kv_cache_groups == 0:
        return ()
    if out is None:
        out = tuple(self.input_block_tables)
    else:
        assert out_ptrs is not None
        assert len(out) == self.num_kv_cache_groups

    # tolist() transfers vectors without invoking the unsupported TXDA scalar
    # .item() path. num_blocks.np is the upstream CPU source of truth.
    indices = idx_mapping.tolist()
    if not len(indices) <= num_reqs_padded <= min(bt.shape[0] for bt in out):
        raise ValueError("Padded request count exceeds block-table capacity")
    for group, destination in enumerate(out):
        destination.zero_()
        source = self.block_tables[group].gpu
        for batch_index, request_index in enumerate(indices):
            if request_index < 0:
                continue
            count = int(self.num_blocks.np[group, request_index])
            destination[batch_index, :count].copy_(source[request_index, :count])
    return tuple(bt[:num_reqs_padded] for bt in out)


def _compute_slot_mappings(
    self,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_tokens_padded: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    destination = self.slot_mappings if out is None else out
    if self.num_kv_cache_groups == 0:
        return destination[:, :num_tokens_padded]
    if self.cp_size != 1:
        raise NotImplementedError("TXDA V2 reference slot mappings require CP=1")
    if not 0 <= num_tokens_padded <= destination.shape[1]:
        raise ValueError("Padded token count exceeds slot-mapping capacity")

    indices = idx_mapping.tolist()
    starts = query_start_loc.tolist()
    token_positions = positions.tolist()
    # Fill the whole persistent buffer, including the tail beyond the returned
    # view. Upstream pads from actual_num_tokens through output capacity.
    slots = torch.full(
        destination.shape, PAD_SLOT_ID, dtype=destination.dtype, device="cpu"
    )
    for group, block_size in enumerate(self.block_sizes):
        if not self._slot_mapping_enabled[group]:
            continue
        source = self.block_tables[group].gpu
        for batch_index, request_index in enumerate(indices):
            if request_index < 0:
                continue
            # Transfer a complete row once; never read GPU scalar block IDs.
            block_ids = source[request_index].cpu().tolist()
            for token_index in range(starts[batch_index], starts[batch_index + 1]):
                position = token_positions[token_index]
                block_index, offset = divmod(position, block_size)
                slots[group, token_index] = block_ids[block_index] * block_size + offset
    destination.copy_(slots)
    return destination[:, :num_tokens_padded]


def patch_txda_v2_preprocess() -> bool:
    """Install idempotent TXDA-only eager fallbacks; leave other devices alone."""
    from vllm.platforms import current_platform

    if current_platform.device_type != "txda":
        return False

    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.buffer_utils import FusedStagedWriter, StagedWriteTensor

    StagedWriteTensor.apply_write = _apply_staged_write
    FusedStagedWriter.apply = _apply_fused_staged_writes
    BlockTables.gather_block_tables = _gather_block_tables
    BlockTables.compute_slot_mappings = _compute_slot_mappings
    return True
