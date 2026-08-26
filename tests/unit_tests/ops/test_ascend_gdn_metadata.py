from types import SimpleNamespace

import pytest
import torch


def _import_gdn_helpers():
    try:
        from vllm_fl.dispatch.backends.vendor.ascend.impl.gdn import (
            _make_single_sequence_chunk_meta,
            _runtime_num_actual_tokens,
            _use_split_multi_prefill,
        )
    except (AttributeError, ImportError) as exc:
        pytest.skip(f"Ascend GDN dependencies are unavailable: {exc}")
    return (
        _make_single_sequence_chunk_meta,
        _use_split_multi_prefill,
        _runtime_num_actual_tokens,
    )


@pytest.mark.parametrize(
    "num_prefills,num_decodes,expected",
    [
        (0, 4, False),
        (1, 0, False),
        (2, 0, True),
        (1, 1, True),
    ],
)
def test_split_multi_prefill_includes_mixed_batches(
    num_prefills, num_decodes, expected
):
    _, use_split, _ = _import_gdn_helpers()
    metadata = SimpleNamespace(
        num_prefills=num_prefills,
        num_decodes=num_decodes,
    )

    assert use_split(metadata) is expected


@pytest.mark.parametrize(
    "seq_len,expected_chunks",
    [(1, 1), (64, 1), (65, 2), (129, 3)],
)
def test_single_sequence_chunk_metadata_boundaries(seq_len, expected_chunks):
    make_meta, _, _ = _import_gdn_helpers()
    metadata = make_meta(
        seq_len=seq_len,
        num_value_heads=8,
        device=torch.device("cpu"),
    )

    assert metadata.chunk_indices_chunk64.dtype == torch.int32
    assert metadata.chunk_indices_chunk64.shape == (expected_chunks, 2)
    assert metadata.chunk_indices_chunk64[:, 0].tolist() == [0] * expected_chunks
    assert metadata.chunk_indices_chunk64[:, 1].tolist() == list(
        range(expected_chunks)
    )
    assert metadata.chunk_offsets_chunk64.tolist() == [0, expected_chunks]
    assert metadata.update_chunk_offsets_chunk64.tolist() == [
        0,
        expected_chunks + 1,
    ]
    assert metadata.final_chunk_indices_chunk64.tolist() == [expected_chunks]


def test_prefill_runtime_token_count_uses_tensor_shape():
    _, _, runtime_num_tokens = _import_gdn_helpers()
    stale_capture_metadata = SimpleNamespace(
        num_prefills=1,
        num_actual_tokens=1,
    )

    assert runtime_num_tokens(
        stale_capture_metadata,
        torch.empty(16384, 3),
    ) == 16384


def test_decode_runtime_token_count_keeps_unpadded_metadata_value():
    _, _, runtime_num_tokens = _import_gdn_helpers()
    decode_metadata = SimpleNamespace(
        num_prefills=0,
        num_actual_tokens=1,
    )

    assert runtime_num_tokens(decode_metadata, torch.empty(2, 3)) == 1
