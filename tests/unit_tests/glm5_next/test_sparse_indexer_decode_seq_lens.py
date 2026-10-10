# SPDX-License-Identifier: Apache-2.0
"""Ordinary decode and graph padding must not read adjacent prefill positions."""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.kernels.glm5_next.portable import (
    append_tail_to_topk,
    expand_pools_to_tokens,
)
from vllm_fl.kernels.glm5_next.sparse_attn_indexer_kpool import (
    _decode_topk_seq_lens,
    _decode_write_layout,
)


@pytest.mark.parametrize("padded", [False, True])
def test_decode_positions_and_tail_are_request_local(padded):
    lengths = torch.tensor([1, 0, 1] if padded else [1, 1, 1], dtype=torch.int32)
    positions = torch.tensor([30, 9, 555, 556] if padded else [30, 100, 9, 555])
    count = 2 if padded else 3
    metadata = SimpleNamespace(decode_lens=lengths, requires_padding=padded)
    uniform, group_lengths, width = _decode_write_layout(metadata, 3, count)
    assert uniform is not padded
    assert group_lengths is lengths
    assert width == 1
    seq_lens = _decode_topk_seq_lens(positions, lengths, count, 3, width, padded)
    assert seq_lens.tolist() == ([31, 0, 10] if padded else [31, 101, 10])
    pool_ids = torch.arange(4).expand(3, 4)
    expanded = expand_pools_to_tokens(
        pool_ids, torch.ones_like(pool_ids, dtype=torch.bool), 16, 4
    )
    out = append_tail_to_topk(expanded, seq_lens, seq_lens // 4, 4)
    assert out[:, 16:].tolist() == [
        [28, 29, 30],
        [-1, -1, -1] if padded else [100, -1, -1],
        [8, 9, -1],
    ]


def test_decode_writer_rejects_multi_token_verify():
    metadata = SimpleNamespace(decode_lens=torch.tensor([1, 3]), requires_padding=True)
    with pytest.raises(ValueError, match="one decode token"):
        _decode_write_layout(metadata, 2, 4)
