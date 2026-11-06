# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for V2 phase selection and native decode-graph padding."""

import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def helpers():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "AscendAttentionState",
        "classify_ascend_attention_state",
        "get_ascend_v2_attention_state",
        "pad_ascend_v2_decode_metadata",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    future = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    )
    namespace = {"Enum": Enum, "torch": torch}
    exec(
        compile(ast.Module(body=[future, *nodes], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return SimpleNamespace(**namespace)


def common(query_lengths, sequence_lengths, padded_tokens=None):
    starts = torch.tensor([0, *query_lengths], dtype=torch.int32).cumsum(0)
    return SimpleNamespace(
        num_reqs=len(query_lengths),
        num_actual_tokens=sum(query_lengths)
        if padded_tokens is None
        else padded_tokens,
        query_start_loc_cpu=starts,
        seq_lens_cpu_upper_bound=torch.tensor([9999] * len(query_lengths)),
    ), torch.tensor(sequence_lengths, dtype=torch.int32)


@pytest.mark.parametrize("context", [2, 17, 257])
def test_changing_batches_preserve_decode_and_native_null_padding(helpers, context):
    for active, padded in ((4, 4), (3, 4), (2, 2), (1, 1), (3, 4)):
        lengths = [1] * active + [0] * (padded - active)
        seq_lens = [context + i for i in range(active)] + [0] * (padded - active)
        metadata, exact_seq_lens = common(lengths, seq_lens, padded)
        original_starts = metadata.query_start_loc_cpu.clone()
        original_seq_lens = exact_seq_lens.clone()
        original_counts = (metadata.num_reqs, metadata.num_actual_tokens)
        metadata.slot_mapping = torch.tensor(
            [128 + i for i in range(active)] + [-1] * (padded - active)
        )
        original_slots = metadata.slot_mapping.clone()
        state = helpers.get_ascend_v2_attention_state(
            metadata, exact_seq_lens, False, None
        )
        assert state is helpers.AscendAttentionState.DecodeOnly
        starts, native_seq_lens = helpers.pad_ascend_v2_decode_metadata(
            metadata, metadata.query_start_loc_cpu, exact_seq_lens
        )
        assert starts.tolist() == list(range(padded + 1))
        assert native_seq_lens.tolist() == seq_lens[:active] + [1] * (padded - active)
        torch.testing.assert_close(
            metadata.query_start_loc_cpu, original_starts, rtol=0, atol=0
        )
        torch.testing.assert_close(exact_seq_lens, original_seq_lens, rtol=0, atol=0)
        torch.testing.assert_close(
            metadata.slot_mapping, original_slots, rtol=0, atol=0
        )
        assert (metadata.num_reqs, metadata.num_actual_tokens) == original_counts
        if active == padded:
            assert starts.data_ptr() == metadata.query_start_loc_cpu.data_ptr()
            assert native_seq_lens is exact_seq_lens
        else:
            assert starts.data_ptr() != metadata.query_start_loc_cpu.data_ptr()
            assert native_seq_lens.data_ptr() != exact_seq_lens.data_ptr()


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    "queries,seq_lens,expected",
    [
        ([1, 0, 0, 0], [1, 0, 0, 0], "PrefillNoCache"),
        ([4, 0, 0, 0], [4, 0, 0, 0], "PrefillNoCache"),
        ([3, 1, 0, 0], [9, 14, 0, 0], None),
    ],
)
def test_real_prefill_is_not_reclassified_as_graph_decode(
    helpers, chunked, queries, seq_lens, expected
):
    metadata, exact = common(queries, seq_lens)
    actual = helpers.get_ascend_v2_attention_state(metadata, exact, chunked, None)
    if expected is None:
        expected = "ChunkedPrefill" if chunked else "PrefillCacheHit"
    assert actual is getattr(helpers.AscendAttentionState, expected)
    assert actual is not helpers.AscendAttentionState.DecodeOnly


def test_one_token_extension_over_existing_context_retains_decode(helpers):
    metadata, exact = common([1, 0, 0, 0], [16, 0, 0, 0], 4)
    state = helpers.get_ascend_v2_attention_state(metadata, exact, False, None)
    assert state is helpers.AscendAttentionState.DecodeOnly


def test_inconsistent_or_nontrailing_padding_is_rejected(helpers):
    for queries, seq_lens, tokens in (
        ([1, 0, 1, 0], [9, 0, 8, 0], 4),
        ([1, 0], [8, 0], 4),
    ):
        metadata, exact = common(queries, seq_lens, tokens)
        with pytest.raises(ValueError, match="trailing empty requests"):
            helpers.pad_ascend_v2_decode_metadata(
                metadata, metadata.query_start_loc_cpu, exact
            )
