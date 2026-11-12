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

"""Single-GPU tests for vllm_fl/models/glm5/indexer_topk.py (M1).

``_select`` writes top-k relative indices (relative to each row's start)
into a caller-owned buffer. Every test checks the buffer against an
independent scalar reference: per row, sort the valid window's logits by
value, take the top-k, and compare as a sorted value multiset while also
checking uniqueness, window bounds and -1 padding of unused slots.
"""

import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.glm5 import indexer_topk

pytestmark = [pytest.mark.gpu]


def _reference_row(values, start, end, k):
    """Top-k of values[start:end] as a sorted-by-value list of floats."""
    lo = max(start, 0)
    hi = min(max(end, 0), len(values))
    valid = [float(values[i]) for i in range(lo, hi)]
    return sorted(valid, reverse=True)[:k]


def _check_row(output_row, values, start, end, k):
    selected = output_row[output_row >= 0].long().tolist()
    assert selected == sorted(selected)
    valid_count = len(range(max(start, 0), min(max(end, 0), len(values))))
    assert len(selected) == min(k, valid_count)
    assert len(set(selected)) == len(selected)
    assert all(0 <= index + start < end for index in selected)
    expected = _reference_row(values, start, end, k)
    actual = sorted((float(values[index + start]) for index in selected), reverse=True)
    assert actual == expected
    assert (output_row[len(selected) :] == -1).all()


class TestPrefillSelect:
    """topk_prefill fills a caller buffer with relative top-k indices."""

    def test_matches_scalar_reference_across_tiles(self, device):
        """70 rows exercise the 32-row internal tiling twice."""
        torch.manual_seed(7)
        rows, columns, k = 70, 24, 6
        logits = torch.randn(rows, columns, device=device)
        starts = torch.arange(rows, device=device) % 5
        ends = starts + torch.tensor([3, 7, 12, 20, 24] * 14, device=device)[:rows]
        out = torch.full((rows, k), -7, dtype=torch.int32, device=device)
        indexer_topk.topk_prefill(logits, starts, ends, out, rows, *logits.stride(), k)
        values = logits.cpu()
        for row in range(rows):
            _check_row(
                out[row].cpu(),
                values[row].tolist(),
                int(starts[row]),
                int(ends[row]),
                k,
            )

    def test_writes_only_the_used_submatrix_of_the_buffer(self, device):
        """Rows/cols past num_rows x top_k stay untouched."""
        torch.manual_seed(11)
        logits = torch.randn(6, 16, device=device)
        storage = torch.full((9, 9), -9, dtype=torch.int32, device=device)
        out = storage[:6, :5]
        starts = torch.zeros(6, dtype=torch.long, device=device)
        ends = torch.full((6,), 16, device=device)
        indexer_topk.topk_prefill(logits, starts, ends, out, 6, *logits.stride(), 5)
        assert (storage[6:] == -9).all()
        assert (storage[:6, 5:] == -9).all()

    def test_stride_mismatch_rejected(self, device):
        """vllm_fl/models/glm5/indexer_topk.py:9 pins the view contract."""
        logits = torch.randn(4, 16, device=device)
        strided = torch.randn(4, 32, device=device)[:, ::2]
        starts = torch.zeros(4, dtype=torch.long, device=device)
        ends = torch.full((4,), 16, device=device)
        buffer = torch.empty(4, 5, dtype=torch.int32, device=device)
        for strides in (strided.stride(), (17, 2)):
            with pytest.raises(ValueError, match="strides do not match"):
                indexer_topk.topk_prefill(logits, starts, ends, buffer, 4, *strides, 5)

    def test_small_buffer_rejected(self, device):
        """vllm_fl/models/glm5/indexer_topk.py:11 pins buffer sizing."""
        logits = torch.randn(4, 16, device=device)
        starts = torch.zeros(4, dtype=torch.long, device=device)
        ends = torch.full((4,), 16, device=device)
        with pytest.raises(ValueError, match="buffer is too small"):
            indexer_topk.topk_prefill(
                logits,
                starts,
                ends,
                torch.empty(3, 5, dtype=torch.int32, device=device),
                4,
                *logits.stride(),
                5,
            )
        with pytest.raises(ValueError, match="buffer is too small"):
            indexer_topk.topk_prefill(
                logits,
                starts,
                ends,
                torch.empty(4, 4, dtype=torch.int32, device=device),
                4,
                *logits.stride(),
                5,
            )

    def test_top_k_zero_fills_minus_one_and_returns(self, device):
        """vllm_fl/models/glm5/indexer_topk.py:16 short-circuits count=0."""
        logits = torch.randn(4, 16, device=device)
        out = torch.full((4, 5), -9, dtype=torch.int32, device=device)
        starts = torch.zeros(4, dtype=torch.long, device=device)
        ends = torch.full((4,), 16, device=device)
        indexer_topk.topk_prefill(logits, starts, ends, out, 4, *logits.stride(), 0)
        assert (out == -1).all()

    def test_empty_columns_fill_minus_one(self, device):
        """count = min(top_k, 0) == 0 with a column-less logits view."""
        logits = torch.empty(4, 0, device=device)
        out = torch.full((4, 5), -9, dtype=torch.int32, device=device)
        starts = torch.zeros(4, dtype=torch.long, device=device)
        ends = torch.zeros(4, device=device)
        indexer_topk.topk_prefill(logits, starts, ends, out, 4, *logits.stride(), 5)
        assert (out == -1).all()

    def test_all_negative_inf_rows_stay_unique_and_in_window(self, device):
        """-inf ties must not duplicate indices or leave the window."""
        logits = torch.full((4, 12), -torch.inf, device=device)
        starts = torch.zeros(4, dtype=torch.long, device=device)
        ends = torch.full((4,), 12, device=device)
        out = torch.empty(4, 6, dtype=torch.int32, device=device)
        indexer_topk.topk_prefill(logits, starts, ends, out, 4, *logits.stride(), 6)
        for row in range(4):
            selected = out[row].tolist()
            assert (out[row] >= 0).all()
            assert len(set(selected)) == 6
            assert all(0 <= value < 12 for value in selected)


class TestDecodeSelect:
    """topk_decode derives per-offset ends from request lengths."""

    @pytest.mark.parametrize("next_n", [1, 3])
    def test_request_lengths_include_each_mtp_offset(self, device, next_n):
        """Check the top-k window separately for each MTP offset."""
        torch.manual_seed(13)
        lengths = torch.tensor([[0], [5], [17]], dtype=torch.int32, device=device)
        rows = lengths.numel() * next_n
        logits = torch.randn(rows, 32, device=device)
        k = 9
        out = torch.empty(rows, k, dtype=torch.int32, device=device)
        indexer_topk.topk_decode(
            logits, next_n, lengths, out, rows, *logits.stride(), k
        )
        values = logits.cpu()
        for row in range(rows):
            seq_len = int(lengths.flatten()[row // next_n])
            end = seq_len - next_n + row % next_n + 1
            _check_row(out[row].cpu(), values[row].tolist(), 0, end, k)

    def test_decode_respects_max_seq_len_argument_position(self, device):
        """max_seq_len is accepted (and ignored) as a keyword argument."""
        torch.manual_seed(3)
        logits = torch.randn(2, 8, device=device)
        out = torch.empty(2, 4, dtype=torch.int32, device=device)
        lengths = torch.tensor([[8], [8]], dtype=torch.int32, device=device)
        indexer_topk.topk_decode(
            logits,
            1,
            lengths,
            out,
            2,
            *logits.stride(),
            4,
            max_seq_len=64,
        )
        for row in range(2):
            selected = out[row][out[row] >= 0].long().tolist()
            assert len(selected) == 4
            expected = sorted((float(logits[row, i]) for i in range(8)), reverse=True)[
                :4
            ]
            actual = sorted((float(logits[row, i]) for i in selected), reverse=True)
            assert actual == expected
