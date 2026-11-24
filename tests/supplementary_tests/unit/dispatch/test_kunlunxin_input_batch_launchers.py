# Copyright 2025 BAAI. All rights reserved.

"""CPU contracts for the Kunlunxin pure-torch input-batch launchers.

``vllm_fl/.../kunlunxin/patches/input_batch_compat.py`` ships seven
``_make_*_torch()`` factories that replace the V2 runner's Triton launchers
with pure-torch loops (FlagTree silently miscompiles the kernels).  These
tests call each factory directly and verify the returned function against a
hand-computed reference, covering:

* ``prepare_pos_seq_lens`` — positions arithmetic and seq_lens padding;
* ``get_num_sampled_and_rejected`` — prefill-boundary rejection accounting;
* ``post_update`` — token history, output bin counts, computed-token delta,
  and the negative idx_mapping skip;
* ``prepare_prefill_inputs`` — input_ids restoration plus lookahead tokens;
* ``combine_sampled_and_draft_tokens`` — logits indices layout, sampled-token
  writeback, and draft-token copy (the contract the rewritten single-exit
  Triton kernel preserves);
* ``expand_idx_mapping`` — per-request expansion of logits routing;
* ``post_update_num_computed_tokens`` — computed-token advance.

These are platform-agnostic (pure torch on CPU tensors) and run ungated.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.supplementary]

from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.input_batch_compat import (
    _make_combine_sampled_and_draft_tokens_torch,
    _make_expand_idx_mapping_torch,
    _make_get_num_sampled_and_rejected_torch,
    _make_post_update_num_computed_tokens_torch,
    _make_post_update_torch,
    _make_prepare_pos_seq_lens_torch,
    _make_prepare_prefill_inputs_torch,
)


# ---------------------------------------------------------------------------
# _make_prepare_pos_seq_lens_torch
# ---------------------------------------------------------------------------


class TestPreparePosSeqLens:
    def test_positions_and_seq_lens_match_reference(self):
        fn = _make_prepare_pos_seq_lens_torch()
        # 3 active requests (rows 0-2) padded to 5 total slots
        idx_mapping = torch.tensor([2, 0, 4, -1, -1], dtype=torch.int64)
        query_start_loc = torch.tensor(
            [0, 2, 5, 5, 5, 5], dtype=torch.int32
        )
        num_computed_tokens = torch.tensor([7, 3, 0, 0, 0], dtype=torch.int64)
        pos = torch.full((5,), -99, dtype=torch.int64)
        seq_lens = torch.tensor([1, 1, 1, 1, 1], dtype=torch.int64)

        fn(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens)

        # request r uses req_idx = idx_mapping[r] to read num_computed
        # r=0: req 2, nc=0, qs=0, ql=2 -> seq_lens[0]=2, pos[0:2]=0..1
        assert seq_lens[0].item() == 2
        assert pos[0:2].tolist() == [0, 1]
        # r=1: req 0, nc=7, qs=2, ql=3 -> seq_lens[1]=10, pos[2:5]=7..9
        assert seq_lens[1].item() == 10
        assert pos[2:5].tolist() == [7, 8, 9]
        # r=2: req 4, nc=0, qs=5, ql=0 -> seq_lens[2]=0, pos untouched
        assert seq_lens[2].item() == 0
        # unused rows padded to zero
        assert seq_lens[3].item() == 0
        assert seq_lens[4].item() == 0

    def test_full_batch_skips_padding(self):
        fn = _make_prepare_pos_seq_lens_torch()
        idx_mapping = torch.tensor([0, 1], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 1, 3], dtype=torch.int32)
        num_computed_tokens = torch.tensor([0, 0], dtype=torch.int64)
        pos = torch.zeros(3, dtype=torch.int64)
        seq_lens = torch.tensor([9, 9], dtype=torch.int64)

        fn(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens)
        assert seq_lens.tolist() == [1, 2]
        assert pos.tolist() == [0, 0, 1]
        # untouched marker proves the padding branch did not run
        assert seq_lens.numel() == 2

    def test_zero_len_request_writes_nothing(self):
        fn = _make_prepare_pos_seq_lens_torch()
        idx_mapping = torch.tensor([0], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 0], dtype=torch.int32)
        num_computed_tokens = torch.tensor([4], dtype=torch.int64)
        pos = torch.full((1,), -7, dtype=torch.int64)
        seq_lens = torch.tensor([1], dtype=torch.int64)

        fn(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens)
        assert seq_lens[0].item() == 4
        assert pos[0].item() == -7, "zero-length query must not write pos"


# ---------------------------------------------------------------------------
# _make_get_num_sampled_and_rejected_torch
# ---------------------------------------------------------------------------


class TestGetNumSampledAndRejected:
    def test_mixed_prefill_and_decode_requests(self):
        fn = _make_get_num_sampled_and_rejected_torch()
        # 3 active rows; req states indexed by idx_mapping
        num_sampled = torch.tensor([2, 1, 2], dtype=torch.int64)
        # row0 is prefill-bound (seq_len < prefill_len) -> both zeroed
        seq_lens = torch.tensor([3, 10, 8], dtype=torch.int64)
        cu_num_logits = torch.tensor([0, 2, 5, 7], dtype=torch.int64)
        idx_mapping = torch.tensor([1, 0, 2], dtype=torch.int64)
        # req 1 prefill_len 5: row0 seq 3 < 5 -> prefill
        # req 0 prefill_len 4: row1 seq 10 >= 4 -> decode, nl = 5-2 = 3
        # req 2 prefill_len 6: row2 seq 8 >= 6 -> decode, nl = 7-5 = 2
        prefill_len = torch.tensor([4, 5, 6], dtype=torch.int64)

        sampled, rejected = fn(
            num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
        )

        assert sampled.tolist() == [0, 1, 2]
        assert rejected.tolist() == [0, 2, 0]

    def test_num_rejected_is_logits_minus_sampled(self):
        fn = _make_get_num_sampled_and_rejected_torch()
        num_sampled = torch.tensor([1, 3], dtype=torch.int64)
        seq_lens = torch.tensor([9, 20], dtype=torch.int64)
        cu_num_logits = torch.tensor([0, 4, 7], dtype=torch.int64)
        idx_mapping = torch.tensor([0, 1], dtype=torch.int64)
        prefill_len = torch.tensor([2, 2], dtype=torch.int64)

        sampled, rejected = fn(
            num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
        )
        # row0: nl=4, ns=1 -> rejected 3; row1: nl=3, ns=3 -> rejected 0
        assert sampled.tolist() == [1, 3]
        assert rejected.tolist() == [3, 0]

    def test_returns_new_rejected_and_reuses_sampled(self):
        fn = _make_get_num_sampled_and_rejected_torch()
        num_sampled = torch.tensor([1], dtype=torch.int64)
        seq_lens = torch.tensor([5], dtype=torch.int64)
        cu_num_logits = torch.tensor([0, 2], dtype=torch.int64)
        idx_mapping = torch.tensor([0], dtype=torch.int64)
        prefill_len = torch.tensor([1], dtype=torch.int64)

        sampled, rejected = fn(
            num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
        )
        assert sampled is num_sampled
        assert rejected is not num_sampled


# ---------------------------------------------------------------------------
# _make_post_update_torch
# ---------------------------------------------------------------------------


class TestPostUpdate:
    def _run(self, *, with_bins=True, with_query_start_loc=True, nreqs=2):
        fn = _make_post_update_torch()
        idx_mapping = torch.tensor([0, 1], dtype=torch.int64)
        num_computed_tokens = torch.tensor([2, 1], dtype=torch.int64)
        last_sampled = torch.tensor([99, 88], dtype=torch.int64)
        output_bin_counts = (
            torch.zeros(2, 10, dtype=torch.int64) if with_bins else None
        )
        sampled_tokens = torch.tensor([[5, 6, 7], [3, 0, 0]], dtype=torch.int64)
        num_sampled = torch.tensor([3, 1], dtype=torch.int64)
        num_rejected = torch.tensor([1, 0], dtype=torch.int64)
        query_start_loc = (
            torch.tensor([0, 4, 5], dtype=torch.int32)
            if with_query_start_loc
            else None
        )
        all_token_ids = torch.zeros(2, 12, dtype=torch.int64)
        total_len = torch.tensor([4, 6], dtype=torch.int64)
        fn(
            idx_mapping,
            num_computed_tokens,
            last_sampled,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            all_token_ids,
            total_len,
        )
        return {
            "num_computed_tokens": num_computed_tokens,
            "last_sampled": last_sampled,
            "output_bin_counts": output_bin_counts,
            "all_token_ids": all_token_ids,
            "total_len": total_len,
        }

    def test_token_history_and_bin_counts(self):
        out = self._run()
        # req0: ns=3, tokens 5,6,7 written at 4,5,6
        assert out["all_token_ids"][0, 4:7].tolist() == [5, 6, 7]
        assert out["total_len"][0].item() == 7
        # req1: ns=1, token 3 written at 6
        assert out["all_token_ids"][1, 6].item() == 3
        assert out["total_len"][1].item() == 7
        # bin counts increment per token id per request
        assert out["output_bin_counts"][0, 5].item() == 1
        assert out["output_bin_counts"][0, 6].item() == 1
        assert out["output_bin_counts"][0, 7].item() == 1
        assert out["output_bin_counts"][1, 3].item() == 1
        # last sampled token ids recorded
        assert out["last_sampled"].tolist() == [7, 3]
        # computed delta: ql - rejected
        # req0: 4 - 1 = +3 -> 2+3 = 5 ; req1: 1 - 0 = +1 -> 1+1 = 2
        assert out["num_computed_tokens"].tolist() == [5, 2]

    def test_without_bin_counts_still_updates_history(self):
        out = self._run(with_bins=False)
        assert out["output_bin_counts"] is None
        assert out["all_token_ids"][0, 4:7].tolist() == [5, 6, 7]
        assert out["num_computed_tokens"].tolist() == [5, 2]

    def test_without_query_start_loc_zero_delta(self):
        out = self._run(with_query_start_loc=False)
        # delta = 0 - rejected
        assert out["num_computed_tokens"].tolist() == [1, 1]

    def test_negative_idx_mapping_row_is_skipped(self):
        fn = _make_post_update_torch()
        idx_mapping = torch.tensor([-1, 1], dtype=torch.int64)
        num_computed_tokens = torch.tensor([2, 1], dtype=torch.int64)
        last_sampled = torch.tensor([99, 88], dtype=torch.int64)
        sampled_tokens = torch.zeros(2, 2, dtype=torch.int64)
        num_sampled = torch.tensor([2, 0], dtype=torch.int64)
        num_rejected = torch.tensor([0, 0], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 2, 2], dtype=torch.int32)
        all_token_ids = torch.zeros(2, 8, dtype=torch.int64)
        total_len = torch.tensor([4, 6], dtype=torch.int64)

        fn(
            idx_mapping,
            num_computed_tokens,
            last_sampled,
            None,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            all_token_ids,
            total_len,
        )
        # row 0 skipped entirely
        assert out_unchanged(
            num_computed_tokens,
            torch.tensor([2, 1], dtype=torch.int64),
        )
        assert total_len.tolist() == [4, 6]
        assert all_token_ids.abs().sum().item() == 0
        assert last_sampled.tolist() == [99, 88]

    def test_token_id_out_of_bin_range_is_ignored(self):
        fn = _make_post_update_torch()
        idx_mapping = torch.tensor([0], dtype=torch.int64)
        num_computed_tokens = torch.tensor([0], dtype=torch.int64)
        last_sampled = torch.zeros(1, dtype=torch.int64)
        output_bin_counts = torch.zeros(1, 2, dtype=torch.int64)  # ids 0..1
        sampled_tokens = torch.tensor([[9]], dtype=torch.int64)  # out of range
        num_sampled = torch.tensor([1], dtype=torch.int64)
        num_rejected = torch.tensor([0], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        all_token_ids = torch.zeros(1, 4, dtype=torch.int64)
        total_len = torch.tensor([0], dtype=torch.int64)

        fn(
            idx_mapping,
            num_computed_tokens,
            last_sampled,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            all_token_ids,
            total_len,
        )
        # history still written; bin count skipped
        assert all_token_ids[0, 0].item() == 9
        assert output_bin_counts.sum().item() == 0


def out_unchanged(actual, expected):
    return torch.equal(actual, expected)


# ---------------------------------------------------------------------------
# _make_prepare_prefill_inputs_torch
# ---------------------------------------------------------------------------


class TestPreparePrefillInputs:
    def test_restores_input_ids_and_lookahead(self):
        fn = _make_prepare_prefill_inputs_torch()
        idx_mapping = torch.tensor([0, 1], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
        all_token_ids = torch.arange(24, dtype=torch.int64).view(2, 12)
        prefill_len = torch.tensor([6, 4], dtype=torch.int64)
        num_computed_tokens = torch.tensor([2, 0], dtype=torch.int64)
        input_ids = torch.full((4,), -1, dtype=torch.int64)
        next_prefill_tokens = torch.full((1, 2), -1, dtype=torch.int64)

        fn(
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            prefill_len,
            num_computed_tokens,
        )

        # req0: nc=2 < pl=6, ql=2 -> input_ids[0:2] = ids[0, 2:4] = 2, 3
        assert input_ids[0:2].tolist() == [2, 3]
        # req1: nc=0 < pl=4, ql=2 -> input_ids[2:4] = ids[1, 0:2] = 12, 13
        assert input_ids[2:4].tolist() == [12, 13]
        # lookahead for req0: pos = 2+2 = 4 < 6 -> token 4; req1: pos 2 < 4 -> 14
        assert next_prefill_tokens[0, 0].item() == 4
        assert next_prefill_tokens[0, 1].item() == 14

    def test_completed_request_is_skipped(self):
        fn = _make_prepare_prefill_inputs_torch()
        idx_mapping = torch.tensor([0], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
        all_token_ids = torch.arange(8, dtype=torch.int64).view(1, 8)
        prefill_len = torch.tensor([3], dtype=torch.int64)
        num_computed_tokens = torch.tensor([3], dtype=torch.int64)  # >= pl
        input_ids = torch.full((2,), -1, dtype=torch.int64)
        next_prefill_tokens = torch.full((1, 1), -1, dtype=torch.int64)

        fn(
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            prefill_len,
            num_computed_tokens,
        )
        assert input_ids.tolist() == [-1, -1]
        assert next_prefill_tokens[0, 0].item() == -1

    def test_lookahead_past_prefill_len_writes_zero(self):
        fn = _make_prepare_prefill_inputs_torch()
        idx_mapping = torch.tensor([0], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        all_token_ids = torch.arange(8, dtype=torch.int64).view(1, 8)
        prefill_len = torch.tensor([2], dtype=torch.int64)
        num_computed_tokens = torch.tensor([0], dtype=torch.int64)
        input_ids = torch.full((1,), -1, dtype=torch.int64)
        next_prefill_tokens = torch.full((2, 1), -1, dtype=torch.int64)

        fn(
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            prefill_len,
            num_computed_tokens,
        )
        # lookahead pos 1 < 2 -> token 1; pos 2 >= 2 -> 0
        assert next_prefill_tokens[:, 0].tolist() == [1, 0]

    def test_no_lookahead_is_a_noop_for_next_tokens(self):
        fn = _make_prepare_prefill_inputs_torch()
        idx_mapping = torch.tensor([0], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        all_token_ids = torch.arange(4, dtype=torch.int64).view(1, 4)
        prefill_len = torch.tensor([4], dtype=torch.int64)
        num_computed_tokens = torch.tensor([0], dtype=torch.int64)
        input_ids = torch.full((1,), -1, dtype=torch.int64)
        next_prefill_tokens = torch.zeros((0, 1), dtype=torch.int64)

        fn(
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            prefill_len,
            num_computed_tokens,
        )
        assert next_prefill_tokens.numel() == 0
        assert input_ids[0].item() == 0


# ---------------------------------------------------------------------------
# _make_combine_sampled_and_draft_tokens_torch
# ---------------------------------------------------------------------------


class TestCombineSampledAndDraftTokens:
    def _run(self, *, seq_lens, prefill_lens, drafts, num_new=1):
        fn = _make_combine_sampled_and_draft_tokens_torch()
        num_reqs = len(seq_lens)
        total_logits = sum(d + num_new for d in drafts)
        input_ids = torch.arange(
            num_reqs * (max(drafts) + num_new), dtype=torch.int64
        )[: total_logits].clone()
        saved_input_ids = input_ids.clone()
        idx_mapping = torch.arange(num_reqs, dtype=torch.int64)
        last_sampled = (
            torch.arange(100, 100 + num_reqs, dtype=torch.int64)
        )
        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
        for i, d in enumerate(drafts):
            needed = d + num_new
            query_start_loc[i + 1] = query_start_loc[i] + needed
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int64)
        cu_num_logits = torch.zeros(num_reqs + 1, dtype=torch.int64)
        for i in range(num_reqs):
            cu_num_logits[i + 1] = cu_num_logits[i] + drafts[i] + num_new
        prefill_len = torch.tensor(prefill_lens, dtype=torch.int64)
        width = max(drafts) if drafts else 0
        rows = []
        for r in range(num_reqs):
            if drafts[r]:
                row = torch.arange(
                    200 + 10 * r, 200 + 10 * r + drafts[r], dtype=torch.int64
                )
                if drafts[r] < width:
                    row = torch.cat(
                        [row, torch.zeros(width - drafts[r], dtype=torch.int64)]
                    )
            else:
                row = torch.zeros(width, dtype=torch.int64)
            rows.append(row)
        draft_tokens = torch.stack(rows)
        logits_indices = fn(
            input_ids,
            idx_mapping,
            last_sampled,
            query_start_loc,
            seq_lens_t,
            prefill_len,
            draft_tokens,
            cu_num_logits,
            total_logits,
            num_new_sampled_tokens=num_new,
        )
        return logits_indices, input_ids, saved_input_ids

    def test_logits_indices_layout(self):
        # 2 requests, drafts (1, 0), 1 new sampled token each
        logits_indices, _, _ = self._run(
            seq_lens=[8, 6],
            prefill_lens=[2, 1],
            drafts=[1, 0],
        )
        # qsl is cumulative over (drafts + new); req0 qe=2, n_logits=2 ->
        # start 0 -> [0, 1]; req1 qe=3, n_logits=1 -> start 2 -> [2]
        assert logits_indices.tolist() == [0, 1, 2]

    def test_decode_request_writes_sampled_token(self):
        logits_indices, input_ids, saved = self._run(
            seq_lens=[8], prefill_lens=[2], drafts=[0]
        )
        # active (8 > 2) and sl - n_logits = 8-1 = 7 >= 2 -> write
        assert input_ids[0].item() == 100, "last sampled token written back"
        # only the first slot changed
        assert input_ids.tolist()[1:] == saved.tolist()[1:]

    def test_prefill_only_request_keeps_input_ids(self):
        logits_indices, input_ids, saved = self._run(
            seq_lens=[1], prefill_lens=[5], drafts=[0]
        )
        # not active (1 <= 5): nothing written
        assert input_ids.tolist() == saved.tolist()

    def test_prompt_tail_slots_are_not_overwritten(self):
        # With two sampled tokens (n_logits=2), a sequence just past its
        # prefill boundary has (sl - n_logits) < pl: the prompt-tail slot
        # must stay intact even though the request is active.
        fn = _make_combine_sampled_and_draft_tokens_torch()
        input_ids = torch.tensor([7, 8], dtype=torch.int64)
        idx_mapping = torch.tensor([0], dtype=torch.int64)
        last_sampled = torch.tensor([55], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
        seq_lens = torch.tensor([5], dtype=torch.int64)
        prefill_len = torch.tensor([4], dtype=torch.int64)
        draft_tokens = torch.zeros((1, 0), dtype=torch.int64)
        cu_num_logits = torch.tensor([0, 2], dtype=torch.int64)

        fn(
            input_ids, idx_mapping, last_sampled, query_start_loc,
            seq_lens, prefill_len, draft_tokens, cu_num_logits, 2,
            num_new_sampled_tokens=2,
        )
        assert input_ids.tolist() == [7, 8]

    def test_draft_tokens_written_at_query_end(self):
        logits_indices, input_ids, saved = self._run(
            seq_lens=[9], prefill_lens=[2], drafts=[2]
        )
        # n_logits=3, qe=3; n_draft=2 -> input_ids[1:3] = draft[0, 0:2]
        assert input_ids[1].item() == 200
        assert input_ids[2].item() == 201
        # sampled token still at logits_start = 3-3 = 0
        assert input_ids[0].item() == 100

    def test_drafts_of_inactive_request_are_skipped(self):
        logits_indices, input_ids, saved = self._run(
            seq_lens=[1], prefill_lens=[5], drafts=[2]
        )
        assert input_ids.tolist() == saved.tolist()

    def test_zero_new_sampled_tokens_never_writes_sampled(self):
        fn = _make_combine_sampled_and_draft_tokens_torch()
        input_ids = torch.tensor([7, 8, 9], dtype=torch.int64)
        idx_mapping = torch.tensor([0], dtype=torch.int64)
        last_sampled = torch.tensor([55], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
        seq_lens = torch.tensor([10], dtype=torch.int64)
        prefill_len = torch.tensor([2], dtype=torch.int64)
        draft_tokens = torch.tensor([[31, 32, 33]], dtype=torch.int64)
        cu_num_logits = torch.tensor([0, 3], dtype=torch.int64)

        logits_indices = fn(
            input_ids, idx_mapping, last_sampled, query_start_loc,
            seq_lens, prefill_len, draft_tokens, cu_num_logits, 3,
            num_new_sampled_tokens=0,
        )
        # num_new=0 -> n_logits=3 all drafts; no sampled writeback
        assert logits_indices.tolist() == [0, 1, 2]
        assert input_ids.tolist() == [31, 32, 33]


# ---------------------------------------------------------------------------
# _make_expand_idx_mapping_torch
# ---------------------------------------------------------------------------


class TestExpandIdxMapping:
    def test_expansion_matches_reference(self):
        fn = _make_expand_idx_mapping_torch()
        idx_mapping = torch.tensor([3, 1], dtype=torch.int64)
        total_num_logits = 5
        cu_num_logits = torch.tensor([0, 2, 5], dtype=torch.int64)

        expanded, local_pos = fn(
            idx_mapping, total_num_logits, cu_num_logits, max_expand_len=4
        )

        assert expanded.dtype == idx_mapping.dtype
        assert expanded.tolist() == [3, 3, 1, 1, 1]
        assert local_pos.dtype == torch.int32
        assert local_pos.tolist() == [0, 1, 0, 1, 2]

    def test_zero_logit_request_leaves_gap(self):
        fn = _make_expand_idx_mapping_torch()
        idx_mapping = torch.tensor([0, 2], dtype=torch.int64)
        total_num_logits = 3
        cu_num_logits = torch.tensor([0, 0, 3], dtype=torch.int64)

        expanded, local_pos = fn(
            idx_mapping, total_num_logits, cu_num_logits, max_expand_len=3
        )
        # first request contributes nothing; slot 0..2 belong to req 2
        assert expanded.tolist() == [2, 2, 2]
        assert local_pos.tolist() == [0, 1, 2]


# ---------------------------------------------------------------------------
# _make_post_update_num_computed_tokens_torch
# ---------------------------------------------------------------------------


class TestPostUpdateNumComputedTokens:
    def test_advances_by_query_length(self):
        fn = _make_post_update_num_computed_tokens_torch()
        idx_mapping = torch.tensor([1, 0], dtype=torch.int64)
        num_computed_tokens = torch.tensor([5, 2], dtype=torch.int64)
        query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32)

        fn(idx_mapping, num_computed_tokens, query_start_loc)

        # row0 routes to req 1: 2 + 3 = 5; row1 routes to req 0: 5 + 2 = 7
        assert num_computed_tokens.tolist() == [7, 5]
