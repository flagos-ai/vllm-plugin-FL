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

"""QSA split-workspace ownership, chunked selection and library forwarding.

The chunk size is forced small so the row-chunking loop is observed directly;
every kernel call receives an explicit row slice whose expected contents come
from the input tensors.  Workspace tests assert allocation shapes, dtypes and
cache identity without requiring an accelerator.
"""

import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.gpu.ops import qsa as ops


class TestTritonSupportProbe:
    @pytest.mark.parametrize("has_triton", [False, True])
    def test_non_host_devices_follow_the_triton_flag(self, monkeypatch, has_triton):
        monkeypatch.setattr(ops, "HAS_TRITON", has_triton)
        assert (
            ops.qsa_forward_metadata_triton_supported(torch.device("cuda"))
            is has_triton
        )
        assert (
            ops.qsa_forward_metadata_triton_supported(torch.device("cpu")) is False
        )


class TestPrepareSplitWorkspace:
    def _inputs(self):
        q = torch.zeros(2, 3, 4, dtype=torch.float32)
        k_cache = torch.zeros(4, 2, 6, 4)
        return q, k_cache

    def test_single_split_needs_no_workspace(self, monkeypatch):
        monkeypatch.setattr(ops, "qsa_sparse_split_count", lambda *args: 1)
        q, k_cache = self._inputs()
        cache = {}
        assert ops.qsa_prepare_split_workspace(q, k_cache, 5, cache) == (1, None)
        assert cache == {}

    def test_multi_split_workspace_is_cached_by_bucket(self, monkeypatch):
        monkeypatch.setattr(ops, "qsa_sparse_split_count", lambda *args: 3)
        q, k_cache = self._inputs()
        cache = {}
        splits, workspace = ops.qsa_prepare_split_workspace(q, k_cache, 5, cache)
        assert splits == 3
        output, stats_a, stats_b = workspace
        assert output.shape == (2, 3, 3, 4)
        assert stats_a.shape == (2, 3, 3)
        assert stats_b.shape == (2, 3, 3)
        assert output.dtype is torch.float32
        assert stats_a.dtype is torch.float32
        key = (2, 3, 6, 4, 5, 3)
        assert cache[key] is workspace
        again = ops.qsa_prepare_split_workspace(q, k_cache, 5, cache)
        assert again[1] is workspace

    def test_missing_cache_still_allocates_a_fresh_workspace(self, monkeypatch):
        monkeypatch.setattr(ops, "qsa_sparse_split_count", lambda *args: 2)
        q, k_cache = self._inputs()
        first = ops.qsa_prepare_split_workspace(q, k_cache, 7)
        second = ops.qsa_prepare_split_workspace(q, k_cache, 7)
        assert first[0] == second[0] == 2
        assert first[1][0] is not second[1][0]
        assert first[1][0].shape == (2, 3, 2, 4)


class TestSparsePagedAttentionForwarding:
    def _inputs(self):
        q = torch.zeros(2, 3, 4)
        k_cache = torch.zeros(4, 2, 6, 4)
        v_cache = torch.zeros(4, 2, 6, 4)
        indices = torch.zeros(2, 5, dtype=torch.int32)
        block_table = torch.zeros(1, 4, dtype=torch.int32)
        token_to_req = torch.zeros(2, dtype=torch.int32)
        return q, k_cache, v_cache, indices, block_table, token_to_req

    def _install(self, monkeypatch, splits):
        calls = []
        monkeypatch.setattr(ops, "qsa_sparse_split_count", lambda *args: splits)
        monkeypatch.setattr(
            ops,
            "_qsa_sparse_paged_attention",
            lambda *args, **kwargs: calls.append((args, kwargs)) or "attended",
        )
        return calls

    def test_single_split_forwards_without_workspace(self, monkeypatch):
        calls = self._install(monkeypatch, splits=1)
        inputs = self._inputs()
        assert ops.qsa_sparse_paged_attention(*inputs, 0.5) == "attended"
        _, kwargs = calls[0]
        assert kwargs["softmax_scale"] == 0.5
        assert kwargs["split_workspace"] is None
        assert kwargs["out"] is None
        assert kwargs["gate"] is None

    def test_multi_split_allocates_a_workspace(self, monkeypatch):
        calls = self._install(monkeypatch, splits=3)
        inputs = self._inputs()
        ops.qsa_sparse_paged_attention(
            *inputs,
            softmax_scale=0.25,
            out="out",
            gate="gate",
        )
        _, kwargs = calls[0]
        output, stats_a, stats_b = kwargs["split_workspace"]
        assert output.shape == (2, 3, 3, 4)
        assert stats_a.shape == (2, 3, 3)
        assert stats_b.shape == (2, 3, 3)
        assert kwargs["softmax_scale"] == 0.25
        assert kwargs["out"] == "out"
        assert kwargs["gate"] == "gate"

    def test_explicit_workspace_is_forwarded_unchanged(self, monkeypatch):
        calls = self._install(monkeypatch, splits=3)
        inputs = self._inputs()
        workspace = ("out", "a", "b")
        ops.qsa_sparse_paged_attention(*inputs, split_workspace=workspace)
        assert calls[0][1]["split_workspace"] is workspace


class TestChunkedSelection:
    def _inputs(self):
        q = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
        k_cache = torch.zeros(2, 2, 3, 4)
        page_table = torch.zeros(2, 4, dtype=torch.int32)
        token_to_req = torch.tensor([0, 0, 1], dtype=torch.int32)
        query_positions = torch.tensor([0, 1, 2], dtype=torch.int64)
        sequence_lengths = torch.tensor([4, 4], dtype=torch.int32)
        return (
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
        )

    def test_rows_are_processed_in_budget_sized_chunks(self, monkeypatch):
        monkeypatch.setattr(ops, "_LOGITS_WORKSPACE_BYTES", 64)
        calls = []
        expands = []

        def fake_mqa(q, k_cache, page_table, requests, positions, lengths, ratio):
            calls.append((q, k_cache, page_table, requests, positions, lengths))
            return torch.zeros(q.shape[0], 8), torch.full((q.shape[0],), 4)

        def fake_topk(logits, visible, block_topk):
            return torch.full(
                (logits.shape[0], block_topk), 2, dtype=torch.int32
            )

        def fake_expand(blocks, positions, lengths, requests, ratio, topk, out):
            expands.append((blocks, positions, requests, out))
            out.fill_(int(blocks[0, 0]))
            return out

        monkeypatch.setattr(ops, "qsa_mqa_paged", fake_mqa)
        monkeypatch.setattr(ops, "_qsa_deterministic_block_topk", fake_topk)
        monkeypatch.setattr(ops, "expand_qsa_block_indices", fake_expand)
        inputs = self._inputs()
        out = ops.qsa_select_paged_tokens(*inputs, token_topk=8, compress_ratio=2)
        # columns = page_table width 4 * k_cache k-dim 2 == 8; 64 // (8 * 32)
        # is zero, so the loop clamps to one row per chunk.
        assert len(calls) == 3
        assert [call[0].shape[0] for call in calls] == [1, 1, 1]
        assert [call[4].tolist() for call in calls] == [[0], [1], [2]]
        assert out.shape == (3, 9)
        assert out.dtype is torch.int32
        assert out.tolist() == torch.full((3, 9), 2).tolist()
        assert [entry[3].shape for entry in expands] == [
            (1, 9),
            (1, 9),
            (1, 9),
        ]

    def test_whole_batch_is_one_chunk_when_the_budget_allows(self, monkeypatch):
        calls = []
        captured = {}

        def fake_mqa(q, *args):
            calls.append(q)
            return torch.zeros(q.shape[0], 8), torch.full((q.shape[0],), 4)

        def fake_expand(blocks, positions, lengths, requests, ratio, topk, out):
            captured["blocks"] = blocks
            captured["out"] = out
            return out

        monkeypatch.setattr(ops, "qsa_mqa_paged", fake_mqa)
        monkeypatch.setattr(
            ops,
            "_qsa_deterministic_block_topk",
            lambda logits, visible, k: torch.zeros(
                logits.shape[0], k, dtype=torch.int32
            ),
        )
        monkeypatch.setattr(ops, "expand_qsa_block_indices", fake_expand)
        inputs = self._inputs()
        out = ops.qsa_select_paged_tokens(*inputs, token_topk=8, compress_ratio=2)
        assert len(calls) == 1
        assert calls[0].shape == (3, 2, 4)
        assert captured["blocks"].shape == (3, 4)  # topk 8 // ratio 2
        assert captured["out"].data_ptr() == out.data_ptr()

    def test_output_width_is_validated(self):
        inputs = self._inputs()
        with pytest.raises(ValueError, match="invalid shape"):
            ops.qsa_select_paged_tokens(
                *inputs,
                token_topk=8,
                compress_ratio=2,
                out=torch.zeros(3, 10, dtype=torch.int32),
            )


class TestSelectAllDelegation:
    def test_all_visible_blocks_are_expanded_in_canonical_order(self, monkeypatch):
        captured = {}

        def fake_expand(blocks, positions, lengths, requests, ratio, topk, out):
            captured.update(
                blocks=blocks,
                positions=positions,
                lengths=lengths,
                requests=requests,
                ratio=ratio,
                topk=topk,
                out=out,
            )
            return "selected"

        monkeypatch.setattr(ops, "expand_qsa_block_indices", fake_expand)
        positions = torch.tensor([0, 1, 2], dtype=torch.int64)
        lengths = torch.tensor([3], dtype=torch.int32)
        requests = torch.zeros(3, dtype=torch.int32)
        assert (
            ops.qsa_select_all_paged_tokens(
                positions, lengths, requests, 8, 2, 4, out="buffer"
            )
            == "selected"
        )
        assert captured["blocks"].tolist() == [[0, 1, 2, 3]] * 3
        assert captured["blocks"].dtype is torch.int32
        assert captured["positions"] is positions
        assert captured["lengths"] is lengths
        assert captured["requests"] is requests
        assert (captured["ratio"], captured["topk"]) == (2, 8)
        assert captured["out"] == "buffer"

    def test_empty_rows_produce_an_empty_block_grid(self, monkeypatch):
        captured = {}

        def fake_expand(blocks, *args):
            captured["shape"] = blocks.shape
            return "selected"

        monkeypatch.setattr(ops, "expand_qsa_block_indices", fake_expand)
        assert (
            ops.qsa_select_all_paged_tokens(
                torch.empty(0, dtype=torch.int64),
                torch.tensor([0], dtype=torch.int32),
                torch.empty(0, dtype=torch.int32),
                8,
                2,
                4,
            )
            == "selected"
        )
        assert captured["shape"] == (0, 4)
