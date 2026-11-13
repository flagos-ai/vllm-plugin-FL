# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.mamba_align_compat import (
    MambaGroupStates,
    _copy_states,
)


@pytest.mark.parametrize("dim_first", [False, True], ids=["sd", "ds"])
def test_mamba_align_copies_shifted_conv_and_temporal_states(dim_first):
    # Each cache has padded block strides. Restrict updates to actual state
    # views and preserve both the destination's untouched tail and its padding.
    conv_pages = torch.arange(8 * 4 * 5).view(8, 4, 5)
    conv = conv_pages[..., :3]
    if dim_first:
        conv = conv.transpose(-1, -2)
    temporal_pages = torch.arange(8 * 9).view(8, 9)
    temporal = temporal_pages[:, :7]
    table = torch.tensor([[0, 1, 2], [3, 4, 5], [5, 6, 7]])
    groups = [MambaGroupStates(table, [conv], [temporal])]

    expected_conv_pages = conv_pages.clone()
    expected_conv_pages[4, :3, :3] = conv_pages[3, 1:, :3]
    # Rows 1 and 2 intentionally exercise a same-physical-block overlap
    # and a permuted request index, respectively.
    expected_conv_pages[6, :3, :3] = conv_pages[6, 1:, :3]
    expected_temporal_pages = temporal_pages.clone()
    expected_temporal_pages[6, :7] = temporal_pages[7, :7]

    _copy_states(
        groups,
        torch.tensor([2, 0, 1]),
        torch.tensor([1, 99, 0]),
        torch.tensor([1, 99, 1]),
        torch.tensor([1, 99, 1]),
        torch.tensor([True, False, True]),
        dim_first=dim_first,
    )

    assert torch.equal(conv_pages, expected_conv_pages)
    assert torch.equal(temporal_pages, expected_temporal_pages)


def test_mamba_align_looks_up_each_group_and_copies_distinct_blocks():
    first = torch.arange(12).view(6, 2)
    second = first.clone() * 10
    expected_first = first.clone()
    expected_first[2] = first[1]
    expected_second = second.clone()
    expected_second[4] = second[5]

    _copy_states(
        [
            MambaGroupStates(torch.tensor([[1, 2]]), [], [first]),
            MambaGroupStates(torch.tensor([[5, 4]]), [], [second]),
        ],
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([0]),
        torch.tensor([True]),
        dim_first=False,
    )

    assert torch.equal(first, expected_first)
    assert torch.equal(second, expected_second)
