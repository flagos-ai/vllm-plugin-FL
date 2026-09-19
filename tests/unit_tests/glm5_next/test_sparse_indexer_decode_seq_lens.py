# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the decode top-k row->token mapping (no GPU required).

On non-uniform decode batches (``requires_padding`` -- mixed plain-decode and
spec-verify requests, or variable MTP verify lens; taken on Hopper where the
varlen/flatten logits path is unavailable), the pool-topk rows follow the
PADDED ``[batch_size, next_n]`` grid: row ``(b, t)`` is flat decode token
``offset_b + t``. The former inline ``dec_seq = positions[:n] + 1``
(``n = batch_size * next_n``) indexes the flat per-token layout with padded
coordinates, so rows after the first non-uniform request read another
request's positions and rows past the decode region read prefill tokens --
``expand_pools_and_append_tail`` then anchors the tail at a foreign length.

These tests pin the defect with the exact production arithmetic and verify
the layout-aware replacement (``_decode_topk_seq_lens``), including the tail
expansion consequences via the pure-torch expand/append pair that the fused
kernel is documented to replicate.
"""

import torch

from vllm_fl.kernels.glm5_next.portable import (
    append_tail_to_topk,
    expand_pools_to_tokens,
)
from vllm_fl.kernels.glm5_next.sparse_attn_indexer_kpool import (
    _decode_topk_seq_lens,
    _decode_write_layout,
)

KPOOL = 4
TOPK_TOKENS = 16
SELECT_K = TOPK_TOKENS // KPOOL


class LegacyDecodeMetadata:
    """Exact field surface exposed by stock vLLM 0.24 decode metadata."""

    def __init__(self, decode_lens: torch.Tensor, requires_padding: bool):
        self.decode_lens = decode_lens
        self.requires_padding = requires_padding


def test_decode_write_layout_accepts_v024_metadata_without_extension_fields():
    """Stock 0.24 metadata must not require GLM5-Next-only attributes."""
    decode_lens = torch.tensor([1, 1], dtype=torch.int32)
    metadata = LegacyDecodeMetadata(decode_lens, requires_padding=False)

    use_uniform, group_lens, lmax = _decode_write_layout(metadata, 2, 2)

    assert use_uniform
    assert group_lens is decode_lens
    assert lmax == 1


def test_decode_write_layout_keeps_v024_nonuniform_fallback_host_static():
    decode_lens = torch.tensor([1, 3], dtype=torch.int32)
    metadata = LegacyDecodeMetadata(decode_lens, requires_padding=True)

    use_uniform, group_lens, lmax = _decode_write_layout(metadata, 2, 4)

    assert not use_uniform
    assert group_lens is decode_lens
    assert lmax == 4


def make_non_uniform_batch():
    """3 requests: plain decode (1 token), MTP verify (4), adaptive verify (3).

    Flat decode positions (production layout: decode tokens first, then any
    prefill tokens of the same batch):
      req0: [30]           (context len 30)
      req1: [100..103]     (verify at context len 100)
      req2: [7, 8, 9]      (verify at context len 7)
    Followed by 4 prefill tokens at positions 555..558 so the flat tensor is
    at least ``n = 3 * 4`` long, as it is in a real mixed batch.
    """
    decode_lens = torch.tensor([1, 4, 3], dtype=torch.int64)
    per_req_positions = [[30], [100, 101, 102, 103], [7, 8, 9]]
    flat_decode = [p for req in per_req_positions for p in req]
    positions = torch.tensor(flat_decode + [555, 556, 557, 558], dtype=torch.int64)
    return decode_lens, per_req_positions, positions


def expected_row_seq_lens(per_req_positions, batch_size, next_n):
    """Ground truth: row (b, t) -> pos + 1 for real rows, 0 for pad rows."""
    out = torch.zeros(batch_size * next_n, dtype=torch.int32)
    for b, req in enumerate(per_req_positions):
        for t, pos in enumerate(req):
            out[b * next_n + t] = pos + 1
    return out


def test_legacy_flat_layout_misaligns_and_bleeds():
    """The bug: ``positions[:n] + 1`` with padded-row coordinates reads other
    requests' positions and prefill positions."""
    decode_lens, per_req, positions = make_non_uniform_batch()
    batch_size = decode_lens.shape[0]
    next_n = int(decode_lens.max())
    n = batch_size * next_n
    assert n == 12

    legacy = positions[:n].to(torch.int32) + 1
    expected = expected_row_seq_lens(per_req, batch_size, next_n)

    # Row (1, 0): req1's first verify token (pos 100, seq 101) reads flat
    # index 4 -> req1's LAST verify token (pos 103). Its true tail token is
    # dropped and the tail anchors 3 tokens late.
    assert int(legacy[4]) == 104 and int(expected[4]) == 101

    # Rows (2, *): req2 starts at flat offset 5, but padded coordinates point
    # at flat indices 8..11 -- the batch's PREFILL positions.
    for t in range(3):
        assert int(legacy[2 * next_n + t]) >= 556, (
            "legacy row (2, t) should read prefill positions"
        )
    assert not torch.equal(legacy, expected)


def test_helper_uniform_layout_matches_flat_slice():
    """Uniform batches keep the flat shortcut (zero behavior/perf change)."""
    per_req = [[200, 201, 202, 203], [50, 51, 52, 53]]
    positions = torch.tensor([p for r in per_req for p in r], dtype=torch.int64)
    decode_lens = torch.tensor([4, 4], dtype=torch.int64)
    out = _decode_topk_seq_lens(positions, decode_lens, 8, 2, 4, requires_padding=False)
    assert torch.equal(out, positions[:8].to(torch.int32) + 1)


def test_helper_padded_layout_per_row():
    """Non-uniform batches map every padded row to its own token's position;
    pad rows get 0 (empty tail)."""
    decode_lens, per_req, positions = make_non_uniform_batch()
    out = _decode_topk_seq_lens(positions, decode_lens, 8, 3, 4, requires_padding=True)
    expected = expected_row_seq_lens(per_req, 3, 4)
    assert torch.equal(out, expected)
    # Pad rows (0, 1..3) and (2, 3) collapse to 0 -> no tail appended.
    assert out[1] == 0 and out[2] == 0 and out[3] == 0 and out[11] == 0


def expand_tail_region(dec_seq):
    """Run the production pure-torch expand + append pair (the fused kernel
    replicates it exactly on the identity path) and return the tail columns
    [TOPK_TOKENS, TOPK_TOKENS + KPOOL - 1)."""
    rows = dec_seq.shape[0]
    pool_ids = torch.arange(SELECT_K, dtype=torch.int64).expand(rows, SELECT_K)
    valid = torch.ones_like(pool_ids, dtype=torch.bool)
    expanded = expand_pools_to_tokens(pool_ids, valid, TOPK_TOKENS, KPOOL)
    seq_lens = dec_seq.to(torch.int32)
    pool_lens = (seq_lens // KPOOL).to(torch.int32)
    out = append_tail_to_topk(expanded, seq_lens, pool_lens, KPOOL)
    return out[:, TOPK_TOKENS:]


def test_tail_expansion_legacy_vs_fixed():
    """End-to-end tail consequence: the fixed mapping appends exactly the
    request's trailing incomplete pool; the legacy flat slice drops real tail
    tokens and emits indices far past the request's own sequence."""
    decode_lens, per_req, positions = make_non_uniform_batch()
    n = 12
    fixed = _decode_topk_seq_lens(
        positions, decode_lens, 8, 3, 4, requires_padding=True
    )
    legacy = positions[:n].to(torch.int32) + 1

    fixed_tail = expand_tail_region(fixed)
    legacy_tail = expand_tail_region(legacy)

    # Fixed: row (1, 0) (seq 101) keeps its single tail token 100; row (2, 2)
    # (seq 10) keeps its full trailing pool [8, 9].
    assert fixed_tail[4, 0].item() == 100
    assert fixed_tail[10, :2].tolist() == [8, 9]

    # Legacy: row (1, 0) loses its tail token (anchored at 104, count 0)...
    assert legacy_tail[4, 0].item() == -1
    # ...and row (2, 2) reads prefill position 557 -> tail indices 556/557,
    # way past req2's 10-token sequence -> out-of-bounds block-table reads.
    assert legacy_tail[10, :2].tolist() == [556, 557]

    assert not torch.equal(legacy_tail, fixed_tail)
