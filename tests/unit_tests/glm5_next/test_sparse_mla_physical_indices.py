# SPDX-License-Identifier: Apache-2.0
"""Regression tests for FlagGems sparse-MLA logical-to-physical mapping."""

import torch

from vllm_fl.dispatch.backends.flaggems.impl.mla_sparse import (
    _convert_request_to_physical_indices,
)


def test_invalid_middle_block_is_compacted_out() -> None:
    request_ids = torch.tensor([0], dtype=torch.int32)
    block_table = torch.tensor([[10, -1, 12]], dtype=torch.int32)
    token_indices = torch.tensor([[0, 64, 128, 1]], dtype=torch.int32)

    physical, valid_lengths = _convert_request_to_physical_indices(
        request_ids, block_table, token_indices, block_size=64
    )

    assert physical.tolist() == [[640, 768, 641, -1]]
    assert valid_lengths.tolist() == [3]


def test_negative_and_out_of_range_tokens_stay_after_valid_prefix() -> None:
    request_ids = torch.tensor([0, 1], dtype=torch.int32)
    block_table = torch.tensor([[2, 3], [5, -1]], dtype=torch.int32)
    token_indices = torch.tensor([[-1, 65, 0, 128], [64, 1, -1, 0]], dtype=torch.int32)

    physical, valid_lengths = _convert_request_to_physical_indices(
        request_ids, block_table, token_indices, block_size=64
    )

    assert physical.tolist() == [[193, 128, -1, -1], [321, 320, -1, -1]]
    assert valid_lengths.tolist() == [2, 2]
