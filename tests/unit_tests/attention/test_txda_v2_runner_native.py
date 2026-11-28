# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU contracts for the native V2-runner replacements used on TXDA.

The TXDA Triton backend miscompiles kernels that cast a runtime-loaded
integer to a pointer, which breaks the V2 block-table, slot-mapping, staged
write and Mamba "align" kernels. ``vendor/txda/patches`` replaces them with
plain tensor indexing; these tests pin the semantics against small CPU
tensors so they need neither a device nor vLLM's kernels.
"""

import os

import pytest
import torch

# Defense-in-depth alongside the conftest gating: skip the whole module when a
# CI job pins a non-TXDA platform (``tests/run.py`` exports FL_TEST_PLATFORM
# from --platform), even if the conftest hook is ever bypassed. An unset value
# means a direct pytest run on a TXDA host, so the tests still run.
_platform = os.environ.get("FL_TEST_PLATFORM")
if _platform and _platform != "txda":
    pytest.skip("TXDA-only contract test", allow_module_level=True)

from vllm_fl.dispatch.backends.vendor.txda.patches.block_table_native import (
    apply_staged_write,
    compute_slot_mappings,
    gather_block_tables,
)
from vllm_fl.dispatch.backends.vendor.txda.patches.mamba_align import (
    aligned_state_indices,
    copy_mamba_state_blocks,
)

PAD = -1


def test_gather_uses_batch_rows_for_destination_and_slots_for_source():
    """Slots handed out out of batch order must not be mixed up.

    This is the 16-request bug: indexing both sides by the slot made requests
    read each other's KV once the slot order stopped matching the batch order.
    """
    # Two requests whose slots are the reverse of their batch order.
    slots = torch.tensor([1, 0], dtype=torch.int64)
    block_tables = [torch.tensor([[10, 11, 12], [20, 21, 22]])]
    num_blocks = torch.tensor([[3, 3]])
    out = [torch.zeros(2, 3, dtype=torch.int64)]

    result = gather_block_tables(
        1, out, block_tables, num_blocks, slots, num_reqs_padded=2
    )

    assert result[0].tolist() == [[20, 21, 22], [10, 11, 12]]


def test_gather_copies_only_each_requests_own_blocks():
    """Ids past a request's block count are stale and must not be copied."""
    slots = torch.tensor([0], dtype=torch.int64)
    block_tables = [torch.tensor([[7, 8, 9]])]
    num_blocks = torch.tensor([[1]])
    out = [torch.full((1, 3), 99, dtype=torch.int64)]

    result = gather_block_tables(
        1, out, block_tables, num_blocks, slots, num_reqs_padded=1
    )

    assert result[0].tolist() == [[7, 0, 0]]


def test_gather_zeroes_padding_rows_and_skips_negative_slots():
    slots = torch.tensor([2, -1], dtype=torch.int64)
    block_tables = [torch.tensor([[0, 0], [3, 4], [5, 6]])]
    num_blocks = torch.tensor([[2, 0, 2]])
    out = [torch.full((3, 2), 42, dtype=torch.int64)]

    result = gather_block_tables(
        1, out, block_tables, num_blocks, slots, num_reqs_padded=3
    )

    # Row 0 holds slot 2's blocks; the dummy row and the padded row are zero.
    assert result[0].tolist() == [[5, 6], [0, 0], [0, 0]]


def test_slot_mappings_follow_positions_and_block_tables():
    # One request, 5 tokens, block size 4 -> slots 0..3 then block 1 slot 0.
    slots = torch.tensor([0], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 5], dtype=torch.int32)
    positions = torch.tensor([0, 1, 2, 3, 4, 0, 0], dtype=torch.int64)
    block_tables = [torch.tensor([[100, 200]])]
    slot_mappings = torch.zeros(1, 8, dtype=torch.int64)

    result = compute_slot_mappings(
        1,
        slot_mappings,
        [4],
        [True],
        block_tables,
        slots,
        query_start_loc,
        positions,
        8,
        PAD,
    )

    # Slot ids are ``physical_block * block_size + offset``.
    assert result[0].tolist() == [400, 401, 402, 403, 800, PAD, PAD, PAD]


def test_slot_mappings_pad_dummy_requests_and_respect_disabled_groups():
    slots = torch.tensor([0, -1], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    positions = torch.tensor([0, 1, 0], dtype=torch.int64)
    block_tables = [torch.tensor([[5]]), torch.tensor([[9]])]
    slot_mappings = torch.zeros(2, 4, dtype=torch.int64)

    result = compute_slot_mappings(
        2,
        slot_mappings,
        [4, 4],
        [True, False],
        block_tables,
        slots,
        query_start_loc,
        positions,
        4,
        PAD,
    )

    assert result[0].tolist() == [20, 21, PAD, PAD]
    # A group with slot mapping disabled contributes no slots at all.
    assert result[1].tolist() == [PAD, PAD, PAD, PAD]


def test_apply_staged_write_handles_one_and_two_dimensional_targets():
    block_table = torch.zeros(2, 4, dtype=torch.int32)
    # Two writes to the flattened block table, plus a scalar-position write.
    apply_staged_write(
        block_table,
        indices=[0, 0, 1],
        starts=[0, 2, 0],
        contents=[11, 12, 13, 21, 22, 23],
        cu_lens=[2, 3, 6],
    )
    assert block_table.tolist() == [[11, 12, 13, 0], [21, 22, 23, 0]]

    flat = torch.zeros(4, dtype=torch.int32)
    apply_staged_write(
        flat, indices=[1, 3], starts=[0, 0], contents=[7, 8], cu_lens=[1, 2]
    )
    assert flat.tolist() == [0, 7, 0, 8]


def test_aligned_state_indices_gathers_checkpoint_columns():
    # block_size 4, seq_len 10 -> first checkpoint column (10-1)//4 = 2.
    out = torch.zeros(1, 2, 3, dtype=torch.int64)
    table = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])

    result = aligned_state_indices(out, [table], torch.tensor([10, 4]), 2, 4)

    # Columns 2,3,4 for the long request; (4-1)//4 = 0 -> 0,1,2 for the short.
    assert result.tolist() == [[[2, 3, 4], [0, 1, 2]]]


def test_copy_mamba_state_blocks_copies_whole_blocks():
    temporal = torch.zeros(3, 2)
    temporal[0] = torch.tensor([1.0, 2.0])
    conv = torch.zeros(3, 4, 2)
    conv[0] = 3.0
    tables = [torch.tensor([[0, 1, 2]])]

    copy_mamba_state_blocks(
        [conv, temporal],
        [0, 0],
        tables,
        rows=torch.tensor([0]),
        src_col=torch.tensor([0]),
        dst_col=torch.tensor([2]),
        bias=torch.tensor([0]),
    )

    assert torch.equal(temporal[2], torch.tensor([1.0, 2.0]))
    assert torch.equal(conv[2], torch.full((4, 2), 3.0))


def test_copy_mamba_state_blocks_refuses_speculative_token_bias():
    """Without spec decode a copy never shifts mid-block; assert that."""
    with pytest.raises(RuntimeError, match="num_accepted_tokens=1"):
        copy_mamba_state_blocks(
            [torch.zeros(2, 4)],
            [0],
            [torch.tensor([[0, 1]])],
            rows=torch.tensor([0]),
            src_col=torch.tensor([0]),
            dst_col=torch.tensor([1]),
            bias=torch.tensor([1]),
        )
