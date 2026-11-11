# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
#
# Hotfix: FlagTree XPU backend uses _load_ptr (a pointer-to-pointer
# dereference implemented via gm2lm) inside _gather_block_tables_kernel and
# _compute_slot_mappings_kernel.  gm2lm of a pointer loaded from an arbitrary
# address cannot be lowered:
#
#     'triton_xpu.gm2lm' op using value defined outside the region
#
# Replace BlockTables.gather_block_tables and BlockTables.compute_slot_mappings
# with pure-torch implementations that index into self.input_block_tables
# and self.block_tables[i].gpu directly.

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_vllm_fl_kunlunxin_block_tables_torch"


def apply_block_tables_compat() -> None:
    """Replace BlockTables.gather_block_tables and compute_slot_mappings."""
    try:
        from vllm.v1.worker.gpu import block_table as bt_mod
    except ImportError:
        return

    BlockTables = getattr(bt_mod, "BlockTables", None)
    if BlockTables is None:
        return

    existing = getattr(BlockTables, "gather_block_tables", None)
    if existing is None:
        return
    if getattr(existing, _PATCH_FLAG, False):
        return

    import torch

    def _gather_block_tables_torch(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
        out=None,
        out_ptrs=None,
    ):
        """Pure-torch replacement for _gather_block_tables_kernel.

        For each batch position i in [0, num_reqs_padded):
          - if i < num_reqs: copy block_tables[g].gpu[req_idx, :] to
            input_block_tables[g][i, :]
          - else: zero the row (padding)
        """
        if self.num_kv_cache_groups == 0:
            return ()
        if out is None:
            out = tuple(self.input_block_tables)
        num_reqs = idx_mapping.shape[0]
        for g, (src, dst) in enumerate(zip(self.block_tables, out)):
            src_gpu = src.gpu  # [max_num_reqs, max_num_blocks]
            for i in range(num_reqs_padded):
                if i < num_reqs:
                    req_idx = idx_mapping[i].item()
                    nb = self.num_blocks.gpu[g, req_idx].item()
                    max_nb = src_gpu.shape[1]
                    dst[i, :nb] = src_gpu[req_idx, :nb]
                    if nb < max_nb:
                        dst[i, nb:] = 0
                else:
                    dst[i, :] = 0
        return tuple(t[:num_reqs_padded] for t in out)

    def _compute_slot_mappings_torch(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
        out=None,
    ):
        """Pure-torch replacement for _compute_slot_mappings_kernel.

        For each request and each of its tokens, compute:
            slot_id = block_number * block_size + block_offset
        where block_number comes from input_block_tables[g][batch_idx, block_idx].
        Uses PAD_SLOT_ID (-1) for dummy requests and disabled mapping groups.
        """
        from vllm.v1.worker.gpu.block_table import PAD_SLOT_ID
        import torch

        if self.num_kv_cache_groups == 0:
            sm = self.slot_mappings if out is None else out
            return sm[:, :num_tokens_padded]

        num_reqs = idx_mapping.shape[0]
        slot_mappings = self.slot_mappings if out is None else out
        # Pad tail to PAD_SLOT_ID
        actual_num_tokens = query_start_loc[num_reqs].item()
        slot_mappings[:, actual_num_tokens:num_tokens_padded] = PAD_SLOT_ID

        for batch_idx in range(num_reqs):
            req_idx = idx_mapping[batch_idx].item()
            start = query_start_loc[batch_idx].item()
            end = query_start_loc[batch_idx + 1].item()
            if start == end:
                continue

            pos = positions[start:end]  # [query_len]

            for g in range(self.num_kv_cache_groups):
                block_size = self.block_sizes[g]
                mapping_enabled = self._slot_mapping_enabled[g]

                if not mapping_enabled or req_idx < 0:
                    slot_mappings[g, start:end] = PAD_SLOT_ID
                    continue

                bt_row = self.input_block_tables[g][batch_idx]  # [max_num_blocks]
                block_indices = pos // block_size
                block_offsets = pos % block_size
                block_numbers = bt_row[block_indices]
                slot_ids = block_numbers * block_size + block_offsets
                slot_mappings[g, start:end] = slot_ids

        return slot_mappings[:, :num_tokens_padded]

    setattr(_gather_block_tables_torch, _PATCH_FLAG, True)
    setattr(_compute_slot_mappings_torch, _PATCH_FLAG, True)
    BlockTables.gather_block_tables = _gather_block_tables_torch
    BlockTables.compute_slot_mappings = _compute_slot_mappings_torch
    logger.info(
        "kunlunxin block_tables_compat: replaced gather_block_tables and "
        "compute_slot_mappings with pure-torch (FlagTree cannot lower "
        "_load_ptr / gm2lm pointer dereference in these kernels)"
    )
