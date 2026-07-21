# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for the Ascend slot mapping Triton kernel launcher.

Adapted from vllm-ascend PR #12096 (tests/ut/worker/a2/test_block_table.py).
The launch-grid assertions run on CPU by mocking the Triton kernel; numerical
correctness of the kernel itself requires an NPU and is covered by e2e tests.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


def _make_fake_block_table(max_num_batched_tokens: int,
                           block_size: int = 128,
                           blocks_per_kv_block: int = 1):
    return SimpleNamespace(
        block_size=block_size,
        blocks_per_kv_block=blocks_per_kv_block,
        max_num_batched_tokens=max_num_batched_tokens,
        cp_kv_cache_interleave_size=1,
        block_table=SimpleNamespace(
            gpu=torch.zeros((4, 8), dtype=torch.int32)),
        slot_mapping=SimpleNamespace(
            gpu=torch.zeros((max_num_batched_tokens, ), dtype=torch.int64)),
    )


def _run_launch_case(num_reqs: int, num_tokens: int,
                     max_num_batched_tokens: int, expected_pad_blocks: int,
                     block_size: int):
    from vllm_fl.dispatch.backends.vendor.ascend.impl import (
        compute_slot_mapping as csm)

    block_table = _make_fake_block_table(max_num_batched_tokens,
                                         block_size=block_size)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    positions = torch.zeros(num_tokens, dtype=torch.int64)

    kernel_mock = MagicMock()
    launcher_mock = MagicMock()
    kernel_mock.__getitem__.return_value = launcher_mock
    with patch.object(csm, "_compute_slot_mapping_kernel", kernel_mock):
        csm._launch_kernel(block_table, num_reqs, num_tokens, query_start_loc,
                           positions)

    kernel_mock.__getitem__.assert_called_once_with(
        (num_reqs + expected_pad_blocks, ))
    launcher_mock.assert_called_once()
    args, kwargs = launcher_mock.call_args
    assert args[0] == num_tokens
    assert args[1] == max_num_batched_tokens
    assert args[2] == num_reqs
    assert kwargs["BLOCK_SIZE"] == 1024
    assert kwargs["BLOCK_TABLE_WINDOW_SIZE"] == 16
    assert kwargs["KV_CACHE_BLOCK_SIZE"] == block_size
    assert kwargs["BLOCKS_PER_KV_BLOCK"] == 1


@pytest.mark.parametrize("block_size", [128])
def test_small_num_tokens_splits_padding_across_programs(block_size):
    """Small active token count should split padding across programs."""
    _run_launch_case(num_reqs=2,
                     num_tokens=3,
                     max_num_batched_tokens=4096,
                     expected_pad_blocks=4,
                     block_size=block_size)


@pytest.mark.parametrize("block_size", [128])
def test_full_batch_needs_no_padding_programs(block_size):
    """No padding programs are needed when active tokens fill the graph size."""
    _run_launch_case(num_reqs=2,
                     num_tokens=512,
                     max_num_batched_tokens=512,
                     expected_pad_blocks=0,
                     block_size=block_size)


def test_hybrid_blocks_pass_physical_block_size():
    """With hybrid blocks the kernel receives the physical (kv-cache) block size."""
    from vllm_fl.dispatch.backends.vendor.ascend.impl import (
        compute_slot_mapping as csm)

    # kernel block size 64, 2 kernel blocks per 128-token physical block
    block_table = _make_fake_block_table(512,
                                         block_size=64,
                                         blocks_per_kv_block=2)
    kernel_mock = MagicMock()
    kernel_mock.__getitem__.return_value = MagicMock()
    with patch.object(csm, "_compute_slot_mapping_kernel", kernel_mock):
        csm._launch_kernel(block_table, 1, 10, torch.zeros(2,
                                                           dtype=torch.int32),
                           torch.zeros(10, dtype=torch.int64))
    _, kwargs = kernel_mock.__getitem__.return_value.call_args
    assert kwargs["KV_CACHE_BLOCK_SIZE"] == 128
    assert kwargs["BLOCKS_PER_KV_BLOCK"] == 2
