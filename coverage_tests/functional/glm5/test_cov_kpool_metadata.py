# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-GPU tests for vllm_fl/models/glm5/kpool_metadata.py (M1).

Every mapping is pure integer math over a page table. Each test builds a
small table on the pinned device, runs the mapping, and compares against a
scalar/loop reference computed independently in the test — no shared
implementation is reused for the expected value. The reference is written
from the documented contract: pool ``p`` owns tokens ``[p*pool_size,
(p+1)*pool_size)`` and lives in the block that holds its *first* token page;
token-page row ``r`` maps to physical page ``page_table[r]``.
"""

import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.glm5.kpool_metadata import (
    append_tail_to_topk,
    build_pooled_page_table,
    compute_pooled_write_locs,
    expand_pools_to_tokens,
    history_group_budget_for_topk,
)

pytestmark = [pytest.mark.gpu]


def _reference_pooled_write_locs(page_table, pool_ids, pool_size):
    """Scalar reference: block-of-first-token-page + in-block pool offset."""
    out = []
    for pool in pool_ids.tolist():
        pool_page_group = pool // 64
        token_page_row = pool_page_group * pool_size
        packed_page = int(page_table[token_page_row].item())
        out.append(packed_page * 64 + pool % 64)
    return out


class TestComputePooledWriteLocs:
    """Pool id -> physical slot mapping follows the documented layout."""

    @pytest.mark.parametrize("pool_size", [2, 4, 8])
    def test_matches_scalar_reference_and_rest_of_table(self, device, pool_size):
        """Slots land exactly at block(first token page)*64 + pool-in-block."""
        torch.manual_seed(11)
        # 12 token pages: pool ids must satisfy (pool//64)*pool_size < 12,
        # i.e. every addressable pool's first token page exists in the table.
        pages = torch.randint(0, 1000, (12,), device=device, dtype=torch.int64)
        max_pool = (12 // pool_size) * 64
        pool_ids = torch.tensor(
            [0, 1, max_pool // 2, max_pool - 2, max_pool - 1], device=device
        )
        if max_pool > 64:
            # ids spanning the 0th and 1st page groups exercise the
            # pool_page_group multiply; only valid when that group is in range.
            pool_ids = torch.cat([pool_ids, torch.tensor([64, 65], device=device)])
        got = compute_pooled_write_locs(pages, pool_ids, pool_size)
        expected = _reference_pooled_write_locs(pages, pool_ids, pool_size)
        assert got.dtype == torch.int64
        assert got.tolist() == expected

    def test_uses_first_token_page_of_each_pool_group(self, device):
        """A pool's block comes from its FIRST token page, not later ones.

        Pool id ``p`` belongs to page group ``p // 64``; that group's first
        token page is row ``(p // 64) * pool_size``, so a pool in group 1
        reads ``pages[pool_size]`` while group 0 pools read ``pages[0]``.
        """
        pages = torch.tensor([7, 8, 9, 10, 11, 12, 13, 14], device=device)
        for pool_size in (2, 4):
            for pool, expected_row in ((0, 0), (1, 0), (63, 0), (64, pool_size)):
                got = compute_pooled_write_locs(
                    pages, torch.tensor([pool], device=device), pool_size
                )
                expected = int(pages[expected_row].item()) * 64 + pool % 64
                assert got.tolist() == [expected], (pool_size, pool)

    def test_accepts_int32_pool_ids_and_table(self, device):
        """int32 inputs are upcast internally; the mapping is unchanged."""
        pages32 = torch.tensor([5, 6, 7, 8], device=device, dtype=torch.int32)
        got = compute_pooled_write_locs(
            pages32, torch.tensor([0, 3], device=device, dtype=torch.int32), 4
        )
        # pool 3 -> page group 0 -> token page row 0 -> block 5; slot 5*64+3.
        assert got.tolist() == [5 * 64 + 0, 5 * 64 + 3]


class TestBuildPooledPageTable:
    """Pool-granular page tables take every pool_size-th token column."""

    @pytest.mark.parametrize("pool_size", [2, 4, 8])
    def test_takes_every_pool_sized_column(self, device, pool_size):
        """Result equals the strided column selection, freshly laid out."""
        torch.manual_seed(13)
        table = torch.randint(0, 500, (5, 64), device=device, dtype=torch.int64)
        got = build_pooled_page_table(table, pool_size)
        assert got.shape == (5, 64 // pool_size)
        for row in range(5):
            expected = [int(table[row, c].item()) for c in range(0, 64, pool_size)]
            assert got[row].tolist() == expected

    def test_result_is_fresh_row_major_even_from_transposed_view(self, device):
        """A transposed (non-contiguous) input still yields stride(-1)==1."""
        base = torch.randint(0, 500, (64, 3), device=device, dtype=torch.int64)
        table = base.t()  # shape (3, 64), stride (-1, 64) — non row-major
        got = build_pooled_page_table(table, 4)
        assert got.is_contiguous()
        assert got.stride(-1) == 1
        assert got.shape == (3, 16)
        for row in range(3):
            expected = [int(table[row, c].item()) for c in range(0, 64, 4)]
            assert got[row].tolist() == expected

    def test_copies_not_views_the_source(self, device):
        """Mutating the result must not write through to the source table."""
        table = torch.arange(128, device=device, dtype=torch.int64).reshape(2, 64)
        got = build_pooled_page_table(table, 4)
        snapshot = got.clone()
        got += 1
        assert not torch.equal(got, snapshot) or snapshot.max() == 0
        assert torch.equal(
            table, torch.arange(128, device=device, dtype=torch.int64).reshape(2, 64)
        )


class TestHistoryGroupBudget:
    """topk tokens need exactly topk/pool_size selected pools."""

    @pytest.mark.parametrize(
        ("topk", "pool_size", "expected"),
        [(64, 4, 16), (2048, 4, 512), (128, 2, 64), (96, 8, 12), (32, 32, 1)],
    )
    def test_budget_divides_exactly(self, topk, pool_size, expected):
        """A divisible pool budget produces an integer number of cache blocks."""
        assert history_group_budget_for_topk(topk, pool_size) == expected


class TestExpandPoolsToTokens:
    """Selected pools expand to a strict-width token topk tensor."""

    def _reference(self, group_ids, group_valid, pool_size, page_table=None):
        out = []
        for row_ids, row_valid in zip(group_ids.tolist(), group_valid.tolist()):
            row = []
            for pool, valid in zip(row_ids, row_valid):
                for offset in range(pool_size):
                    token = pool * pool_size + offset
                    if page_table is not None:
                        token = int(page_table[min(max(token, 0), 63)].item())
                    if valid:
                        row.append(token)
                    else:
                        row.append(-1)
            out.append(row)
        return out

    def test_plain_expansion_marks_invalid_minus_one(self, device):
        """token = pool*pool_size + offset; invalid pools expand to -1."""
        torch.manual_seed(17)
        group_ids = torch.tensor([[0, 5, 2], [7, 1, 3]], device=device)
        group_valid = torch.tensor(
            [[True, False, True], [False, True, True]], device=device
        )
        got = expand_pools_to_tokens(group_ids, group_valid, 12, 4)
        assert got.dtype == torch.int32
        assert got.shape == (2, 12)
        assert got.tolist() == self._reference(group_ids, group_valid, 4)

    def test_page_table_expansion_gathers_physical_pages(self, device):
        """With a page table, tokens gather the physical page of their slot."""
        torch.manual_seed(19)
        table = torch.randint(0, 900, (2, 64), device=device, dtype=torch.int64)
        group_ids = torch.tensor([[0, 2], [1, 3]], device=device)
        group_valid = torch.ones(2, 2, dtype=torch.bool, device=device)
        got = expand_pools_to_tokens(group_ids, group_valid, 8, 4, page_table=table)
        # Row 1 uses row 1's table slice in the reference too.
        ref = []
        for row in range(2):
            ref.append(
                [
                    int(table[row, min(max(p * 4 + o, 0), 63)].item())
                    for p in group_ids[row].tolist()
                    for o in range(4)
                ]
            )
        assert got.tolist() == ref

    def test_offsets_expansion_adds_row_offsets(self, device):
        """topk_offsets shift each row's tokens by its own offset."""
        group_ids = torch.tensor([[0, 3], [2, 1]], device=device)
        group_valid = torch.ones(2, 2, dtype=torch.bool, device=device)
        offsets = torch.tensor([[100], [0]], device=device)
        got = expand_pools_to_tokens(group_ids, group_valid, 8, 4, topk_offsets=offsets)
        row0 = [p * 4 + o + 100 for p in (0, 3) for o in range(4)]
        row1 = [p * 4 + o for p in (2, 1) for o in range(4)]
        assert got.tolist() == [row0, row1]

    def test_offsets_accepts_one_dimensional_form(self, device):
        """A 1-D offsets tensor also shifts each row by its own value."""
        group_ids = torch.tensor([[0, 1], [0, 1]], device=device)
        group_valid = torch.ones(2, 2, dtype=torch.bool, device=device)
        offsets = torch.tensor([10, 20], device=device)
        got = expand_pools_to_tokens(group_ids, group_valid, 8, 4, topk_offsets=offsets)
        base = [p * 4 + o for p in (0, 1) for o in range(4)]
        assert got.tolist() == [[v + 10 for v in base], [v + 20 for v in base]]


class TestAppendTailToTopk:
    """Trailing partial-pool tokens are appended after the expanded history."""

    def _reference(self, topk, seq_lens, pool_lens, pool_size, page_table=None):
        out = []
        for row, (seq_len, pool_len) in enumerate(
            zip(seq_lens.tolist(), pool_lens.tolist())
        ):
            history = topk[row].tolist()
            tail_start = pool_len * pool_size
            tail = [
                tail_start + off
                for off in range(min(seq_len - tail_start, pool_size - 1))
            ]
            if page_table is not None:
                last = page_table.shape[1] - 1
                tail = [int(page_table[row, min(max(t, 0), last)].item()) for t in tail]
            width = len(history) + pool_size - 1
            row_out = list(history)
            while len(row_out) < width:
                row_out.append(-1)
            for i, t in enumerate(tail):
                row_out[len(history) + i] = t
            out.append(row_out)
        return out

    def test_appends_each_rows_own_tail_after_history(self, device):
        """Row r's tail tokens follow its history; padding stays -1."""
        topk = torch.tensor(
            [[0, 4, 8, 12], [16, 20, 24, 28]], dtype=torch.int32, device=device
        )
        seq_lens = torch.tensor([13, 15], device=device)
        pool_lens = torch.tensor([3, 3], device=device)
        got = append_tail_to_topk(topk, seq_lens, pool_lens, 4)
        assert got.dtype == torch.int32
        assert got.shape == (2, 7)
        assert got.tolist() == self._reference(topk, seq_lens, pool_lens, 4)

    def test_per_row_history_is_not_shared_across_rows(self, device):
        """The history gather must read each row's own topk (row-0 bug)."""
        topk = torch.tensor(
            [[101, 102, 103, 104], [201, 202, 203, 204]],
            dtype=torch.int32,
            device=device,
        )
        seq_lens = torch.tensor([9, 9], device=device)
        pool_lens = torch.tensor([2, 2], device=device)
        got = append_tail_to_topk(topk, seq_lens, pool_lens, 4)
        expected = self._reference(topk, seq_lens, pool_lens, 4)
        assert got.tolist() == expected
        assert got[1, :4].tolist() == [201, 202, 203, 204]

    def test_full_pool_sequence_has_empty_tail(self, device):
        """seq_len == pool_len*pool_size leaves all tail slots -1."""
        topk = torch.tensor([[5, 6]], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([8], device=device)
        pool_lens = torch.tensor([2], device=device)
        got = append_tail_to_topk(topk, seq_lens, pool_lens, 4)
        # out width = n_cols + pool_size - 1 = 5; every tail slot stays -1.
        assert got.tolist() == [[5, 6, -1, -1, -1]]

    def test_page_table_maps_tail_tokens(self, device):
        """With a page table, tail values gather physical pages."""
        torch.manual_seed(23)
        table = torch.randint(0, 700, (1, 16), device=device, dtype=torch.int64)
        topk = torch.tensor([[3, 7]], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([10], device=device)
        pool_lens = torch.tensor([2], device=device)
        got = append_tail_to_topk(topk, seq_lens, pool_lens, 4, page_table=table)
        assert got.tolist() == self._reference(topk, seq_lens, pool_lens, 4, table)

    def test_offsets_shift_tail_tokens(self, device):
        """topk_offsets add to raw tail positions only."""
        topk = torch.tensor([[1, 2]], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([11], device=device)
        pool_lens = torch.tensor([2], device=device)
        offsets = torch.tensor([100], device=device)
        got = append_tail_to_topk(topk, seq_lens, pool_lens, 4, topk_offsets=offsets)
        # tail raw positions 8, 9, 10 -> +100 each; history untouched.
        assert got.tolist() == [[1, 2, 108, 109, 110]]

    def test_pool_size_one_returns_input_unchanged(self, device):
        """tail_pool == 0 short-circuits: the tensor object is returned."""
        topk = torch.tensor([[4, 5, 6]], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([3], device=device)
        pool_lens = torch.tensor([3], device=device)
        got = append_tail_to_topk(topk, seq_lens, pool_lens, 1)
        assert got is topk
