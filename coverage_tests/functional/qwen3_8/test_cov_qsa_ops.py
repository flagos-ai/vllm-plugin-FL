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

"""QSA metadata validation, deterministic TopK and index expansion.

The graph-safe builder rejects malformed ABIs before any Triton launch, so
those checks are exercised on CPU with the capability probe injected. The
deterministic TopK is pure Torch and is compared against a
``(-score, index)`` selection reference. The accelerator test drives the
expansion kernel through ``qsa_select_all_paged_tokens`` and compares with a
Python implementation of the documented complete-block + causal-tail layout.
"""

import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.gpu.ops import qsa as ops


class TestTritonCapability:
    def test_host_tensors_are_not_triton_devices(self):
        assert ops._is_triton_device(torch.zeros(1)) is False
        assert ops._is_triton_device(torch.zeros(1, device="meta")) is False
        assert ops.qsa_forward_metadata_triton_supported(torch.device("cpu")) is False
        assert ops.qsa_forward_metadata_triton_supported(torch.device("meta")) is False


def _metadata_tensors(rows=2, device="cpu"):
    return dict(
        token_to_req=torch.zeros(rows, dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([1, 1], dtype=torch.int32, device=device),
        block_table=torch.zeros(2, 2, dtype=torch.int32, device=device),
        common_slot_mapping=torch.zeros(rows, dtype=torch.int64, device=device),
        logical_positions=torch.zeros(rows, dtype=torch.int64, device=device),
        qsa_slot_mapping=torch.zeros(rows, dtype=torch.int64, device=device),
    )


class TestBuildForwardMetadata:
    """Early exits and every ABI validation of the fused metadata builder."""

    def test_ratio_one_never_enters_the_kernel(self):
        tensors = _metadata_tensors()
        before = tensors["logical_positions"].clone()
        assert (
            ops.build_qsa_forward_metadata(
                **tensors,
                storage_block_size=4,
                compress_ratio=1,
            )
            is False
        )
        assert torch.equal(tensors["logical_positions"], before)

    def test_host_tensor_without_triton_returns_false(self):
        assert (
            ops.build_qsa_forward_metadata(
                **_metadata_tensors(), storage_block_size=4, compress_ratio=4
            )
            is False
        )

    @pytest.fixture()
    def force_triton(self, monkeypatch):
        monkeypatch.setattr(ops, "_is_triton_device", lambda tensor: True)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda t: t.update(token_to_req=torch.zeros(2, dtype=torch.int32).view(2, 1)), "incompatible shapes"),
            (lambda t: t.update(common_slot_mapping=torch.zeros(1, dtype=torch.int64)), "incompatible shapes"),
            (lambda t: t.update(query_start_loc=torch.tensor([0, 1], dtype=torch.int32)), "terminal offset"),
            (lambda t: t.update(logical_positions=torch.zeros(2, dtype=torch.int32)), "must use int64"),
            (lambda t: t.update(qsa_slot_mapping=torch.zeros(2, dtype=torch.int32)), "must use int64"),
            (lambda t: t.update(token_to_req=torch.zeros(2, dtype=torch.int64)), "must use int32"),
            (lambda t: t.update(seq_lens=torch.zeros(2, dtype=torch.int64, device="meta")), "same device"),
        ],
    )
    def test_rejects_incompatible_metadata(self, force_triton, mutate, message):
        tensors = _metadata_tensors()
        mutate(tensors)
        with pytest.raises(ValueError, match=message):
            ops.build_qsa_forward_metadata(
                **tensors, storage_block_size=4, compress_ratio=4
            )

    def test_non_positive_geometry_is_rejected(self, force_triton):
        with pytest.raises(ValueError, match="must be positive"):
            ops.build_qsa_forward_metadata(
                **_metadata_tensors(), storage_block_size=0, compress_ratio=4
            )
        with pytest.raises(ValueError, match="must be positive"):
            ops.build_qsa_forward_metadata(
                **_metadata_tensors(), storage_block_size=4, compress_ratio=0
            )

    def test_zero_rows_is_a_successful_no_op(self, force_triton):
        tensors = _metadata_tensors(rows=0)
        assert (
            ops.build_qsa_forward_metadata(
                **tensors, storage_block_size=4, compress_ratio=4
            )
            is True
        )

    def test_empty_requests_or_table_returns_false(self, force_triton):
        empty_requests = _metadata_tensors()
        empty_requests["seq_lens"] = torch.zeros(0, dtype=torch.int32)
        empty_requests["query_start_loc"] = torch.tensor([0], dtype=torch.int32)
        assert (
            ops.build_qsa_forward_metadata(
                **empty_requests, storage_block_size=4, compress_ratio=4
            )
            is False
        )
        empty_table = _metadata_tensors()
        empty_table["block_table"] = torch.zeros(2, 0, dtype=torch.int32)
        assert (
            ops.build_qsa_forward_metadata(
                **empty_table, storage_block_size=4, compress_ratio=4
            )
            is False
        )


def _topk_reference(logits, visible, block_topk):
    rows, columns = logits.shape
    width = min(block_topk, columns)
    result = []
    for row in range(rows):
        order = sorted(
            range(columns), key=lambda column: (-float(logits[row, column]), column)
        )
        chosen = sorted(column for column in order[:width] if column < visible[row])
        result.append(chosen + [-1] * (block_topk - len(chosen)))
    return result


class TestDeterministicBlockTopk:
    """Score-descending membership with logical-index ascending order."""

    def test_ties_and_invisible_blocks_are_resolved_deterministically(self):
        logits = torch.tensor(
            [[0.5, 2.0, 1.0, 1.5], [3.0, 2.0, 1.0, 0.0]], dtype=torch.float32
        )
        visible = torch.tensor([3, 4], dtype=torch.int32)
        got = ops._qsa_deterministic_block_topk(logits, visible, 2)
        assert got.dtype == torch.int32
        assert got.tolist() == _topk_reference(logits, visible, 2)
        assert got.tolist() == [[1, -1], [0, 1]]

    def test_topk_wider_than_the_table_keeps_the_strict_width(self):
        logits = torch.tensor([[0.0, 2.0, 1.0]])
        visible = torch.tensor([2])
        got = ops._qsa_deterministic_block_topk(logits, visible, 5)
        assert got.shape == (1, 5)
        assert got.tolist() == [[0, 1, -1, -1, -1]]

    def test_invalid_shapes_and_budget_are_rejected(self):
        logits = torch.zeros(2, 4)
        with pytest.raises(ValueError, match="incompatible shapes"):
            ops._qsa_deterministic_block_topk(logits, torch.zeros(3), 2)
        with pytest.raises(ValueError, match="incompatible shapes"):
            ops._qsa_deterministic_block_topk(torch.zeros(4), torch.zeros(2), 2)
        with pytest.raises(ValueError, match="must be positive"):
            ops._qsa_deterministic_block_topk(logits, torch.zeros(2), 0)


class TestSplitWorkspace:
    def test_workspace_shape_and_dtype_are_derived_from_q(self):
        q = torch.zeros(2, 3, 4, dtype=torch.float32)
        output, stats_a, stats_b = ops._qsa_get_split_workspace(q, 2)
        assert output.shape == (2, 3, 2, 4)
        assert stats_a.shape == (2, 3, 2)
        assert stats_b.shape == (2, 3, 2)
        assert output.dtype is torch.float32
        storage = {
            output.untyped_storage().data_ptr(),
            stats_a.untyped_storage().data_ptr(),
            stats_b.untyped_storage().data_ptr(),
        }
        assert len(storage) == 3


class TestSelectPagedTokens:
    def test_empty_rows_return_the_expected_width(self):
        q = torch.zeros(0, 2, 4)
        out = ops.qsa_select_paged_tokens(
            q,
            torch.zeros(1, 2, 1, 4),
            torch.zeros(1, 2, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int64),
            torch.tensor([0], dtype=torch.int32),
            token_topk=4,
            compress_ratio=2,
        )
        assert out.shape == (0, 5)
        assert out.dtype == torch.int32

    def test_explicit_output_shape_is_enforced(self):
        with pytest.raises(ValueError, match="invalid shape"):
            ops.qsa_select_paged_tokens(
                torch.zeros(0, 2, 4),
                torch.zeros(1, 2, 1, 4),
                torch.zeros(1, 2, dtype=torch.int32),
                torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.int64),
                torch.tensor([0], dtype=torch.int32),
                token_topk=4,
                compress_ratio=2,
                out=torch.zeros(0, 4, dtype=torch.int32),
            )


class TestSelectAllValidation:
    def _call(self, **overrides):
        values = dict(
            query_positions=torch.zeros(2, dtype=torch.int64),
            sequence_lengths=torch.tensor([4], dtype=torch.int32),
            token_to_req=torch.zeros(2, dtype=torch.int32),
            token_topk=8,
            compress_ratio=2,
            max_model_len=8,
        )
        values.update(overrides)
        return ops.qsa_select_all_paged_tokens(**values)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"max_model_len": 0}, "within budget"),
            ({"max_model_len": 9}, "within budget"),
            ({"compress_ratio": 0}, "divisible by compression"),
            ({"compress_ratio": 3}, "divisible by compression"),
        ],
    )
    def test_rejects_contexts_outside_the_budget(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            self._call(**overrides)

    def test_valid_host_call_requires_an_accelerator_kernel(self):
        with pytest.raises(RuntimeError, match="accelerator Triton"):
            self._call()


class TestExpandValidation:
    def test_host_tensor_is_rejected_before_validation(self):
        with pytest.raises(RuntimeError, match="accelerator Triton"):
            ops.expand_qsa_block_indices(
                torch.zeros(1, 2, dtype=torch.int32),
                torch.zeros(1, dtype=torch.int64),
                torch.tensor([4], dtype=torch.int32),
                torch.zeros(1, dtype=torch.int32),
                compress_ratio=2,
                token_topk=4,
            )

    @pytest.fixture()
    def force_triton(self, monkeypatch):
        monkeypatch.setattr(ops, "_is_triton_device", lambda tensor: True)

    def _valid(self, **overrides):
        values = dict(
            block_indices=torch.zeros(2, 2, dtype=torch.int32),
            query_positions=torch.tensor([0, 1], dtype=torch.int64),
            sequence_lengths=torch.tensor([4], dtype=torch.int32),
            token_to_req=torch.tensor([0, 0], dtype=torch.int32),
            compress_ratio=2,
            token_topk=4,
        )
        values.update(overrides)
        return values

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"token_topk": 5}, "divisible by compression"),
            ({"block_indices": torch.zeros(3, 2, dtype=torch.int32)}, "invalid shape"),
            ({"token_to_req": torch.zeros(3, dtype=torch.int32)}, "must match"),
            ({"sequence_lengths": torch.zeros(0, dtype=torch.int32)}, "nonempty"),
            ({"out": torch.zeros(2, 4, dtype=torch.int32)}, "invalid shape"),
        ],
    )
    def test_rejects_invalid_arguments(self, force_triton, overrides, message):
        with pytest.raises(ValueError, match=message):
            ops.expand_qsa_block_indices(**self._valid(**overrides))

    def test_zero_rows_returns_the_output_unchanged(self, force_triton):
        out = ops.expand_qsa_block_indices(
            **self._valid(
                block_indices=torch.zeros(0, 2, dtype=torch.int32),
                query_positions=torch.empty(0, dtype=torch.int64),
                token_to_req=torch.empty(0, dtype=torch.int32),
            )
        )
        assert out.shape == (0, 5)
        assert out.dtype == torch.int32


def _select_all_reference(query_positions, sequence_lengths, token_to_req, topk, ratio):
    rows = len(query_positions)
    block_topk = topk // ratio
    width = topk + ratio - 1
    blocks = list(range(block_topk))
    result = []
    for row in range(rows):
        query_position = int(query_positions[row])
        request = int(token_to_req[row])
        sequence_length = int(sequence_lengths[request])
        complete = min(
            min((query_position + 1) // ratio, sequence_length // ratio), block_topk
        )
        expanded = complete * ratio
        tail_start = ((query_position + 1) // ratio) * ratio
        tail_count = (query_position + 1) - tail_start
        values = []
        for column in range(width):
            if column < expanded:
                rank = column // ratio
                offset = column % ratio
                token = blocks[min(rank, block_topk - 1)] * ratio + offset
                valid = 0 <= token < sequence_length
            else:
                tail_offset = column - expanded
                is_tail = (
                    column >= expanded
                    and tail_offset < tail_count
                    and tail_offset < ratio - 1
                )
                token = tail_start + tail_offset
                valid = is_tail and 0 <= token < sequence_length
            values.append(token if valid else -1)
        result.append(values)
    return result


@pytest.mark.gpu
class TestSelectAllAccelerator:
    def test_expansion_matches_complete_block_and_tail_reference(self, device):
        topk, ratio = 8, 2
        query_positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
        sequence_lengths = torch.tensor([4], dtype=torch.int32, device=device)
        token_to_req = torch.zeros(4, dtype=torch.int32, device=device)
        got = ops.qsa_select_all_paged_tokens(
            query_positions,
            sequence_lengths,
            token_to_req,
            topk,
            ratio,
            4,
        )
        expected = _select_all_reference(
            query_positions.cpu().tolist(),
            sequence_lengths.cpu().tolist(),
            token_to_req.cpu().tolist(),
            topk,
            ratio,
        )
        assert got.shape == (4, topk + ratio - 1)
        assert got.dtype == torch.int32
        assert got.cpu().tolist() == expected
        assert got.cpu().tolist()[0][:2] == [0, -1]
        assert got.cpu().tolist()[3][:4] == [0, 1, 2, 3]
