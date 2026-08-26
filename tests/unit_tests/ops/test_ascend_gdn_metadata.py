from types import SimpleNamespace

import pytest
import torch


def _import_gdn_helpers():
    try:
        from vllm_fl.dispatch.backends.vendor.ascend.impl import gdn
    except (AttributeError, ImportError) as exc:
        pytest.skip(f"Ascend GDN dependencies are unavailable: {exc}")
    return gdn


def _import_gdn_patch_helpers():
    try:
        from vllm_fl.dispatch.backends.vendor.ascend.patches import (
            patch_gdn_attn,
        )
    except (AttributeError, ImportError) as exc:
        pytest.skip(f"Ascend GDN patch dependencies are unavailable: {exc}")
    return patch_gdn_attn


@pytest.mark.parametrize(
    ("query_start_loc_values", "expected_chunk_counts"),
    [
        ((0, 64, 129), (1, 2)),
        ((0, 4, 4, 12), (1, 0, 1)),
        ((0, 1, 33), (1, 1)),
    ],
    ids=(
        "variable_length_prefills",
        "mixed_batch_with_padding",
        "prefix_cache_resume",
    ),
)
def test_cpu_prebuilt_metadata_keeps_whole_batch_boundaries(
    query_start_loc_values,
    expected_chunk_counts,
):
    patch_gdn_attn = _import_gdn_patch_helpers()
    builder = SimpleNamespace(
        _ascend_gdn_chunk_size=64,
        _ascend_gdn_large_block_size=1216,
        _ascend_gdn_cumsum_block_size=512,
    )
    query_start_loc = torch.tensor(query_start_loc_values, dtype=torch.int32)

    metadata = patch_gdn_attn._build_non_spec_chunked_prefill_meta_cpu(
        builder,
        query_start_loc,
    )

    chunk_counts = torch.tensor(expected_chunk_counts, dtype=torch.int32)
    expected_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            chunk_counts.cumsum(dim=0),
        )
    )
    expected_update_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            (chunk_counts + 1).cumsum(dim=0),
        )
    )
    nonempty_chunk_counts = chunk_counts[chunk_counts > 0]
    expected_sequence_ids = torch.repeat_interleave(
        torch.arange(nonempty_chunk_counts.numel(), dtype=torch.int32),
        nonempty_chunk_counts,
    )

    assert torch.equal(metadata.chunk_offsets_chunk64, expected_offsets)
    assert torch.equal(
        metadata.update_chunk_offsets_chunk64,
        expected_update_offsets,
    )
    assert torch.equal(
        metadata.final_chunk_indices_chunk64,
        expected_update_offsets[1:] - 1,
    )
    assert torch.equal(
        metadata.chunk_indices_chunk64[:, 0],
        expected_sequence_ids,
    )


@pytest.mark.parametrize(
    ("query_start_loc_values", "num_prefills", "num_decodes"),
    [
        ((0, 64, 129), 2, 0),
        ((0, 1, 66, 67), 1, 2),
        ((0, 1, 33), 2, 0),
    ],
    ids=(
        "variable_length_prefills",
        "mixed_prefill_decode",
        "prefix_cache_resume",
    ),
)
def test_multi_sequence_prefill_uses_one_batched_chunk_call(
    monkeypatch,
    query_start_loc_values,
    num_prefills,
    num_decodes,
):
    gdn = _import_gdn_helpers()
    sentinels = {
        name: object()
        for name in (
            "query",
            "key",
            "value",
            "g",
            "beta",
            "initial_state",
            "prebuilt_meta",
        )
    }
    query_start_loc = torch.tensor(query_start_loc_values, dtype=torch.int32)
    legacy_prebuilt_meta = object()
    attn_metadata = SimpleNamespace(
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        # The patched builder stores the authoritative metadata here.  Keep a
        # stale legacy value too, so the test proves the runtime does not use it.
        non_spec_prefill_fallback_meta=SimpleNamespace(
            causal_conv1d=object(),
            chunk=sentinels["prebuilt_meta"],
        ),
        non_spec_chunked_prefill_meta=legacy_prebuilt_meta,
    )
    chunk_calls = []
    metadata_calls = []

    get_prebuilt_meta = gdn.get_non_spec_chunked_prefill_meta

    def spy_get_prebuilt_meta(metadata):
        metadata_calls.append(metadata)
        return get_prebuilt_meta(metadata)

    monkeypatch.setattr(gdn, "get_non_spec_chunked_prefill_meta", spy_get_prebuilt_meta)

    def fake_chunk_gated_delta_rule(**kwargs):
        chunk_calls.append(kwargs)
        return "batched-output", "batched-final-state"

    monkeypatch.setattr(gdn, "chunk_gated_delta_rule", fake_chunk_gated_delta_rule)

    result = gdn._run_batched_chunk_prefill(
        query=sentinels["query"],
        key=sentinels["key"],
        value=sentinels["value"],
        g=sentinels["g"],
        beta=sentinels["beta"],
        initial_state=sentinels["initial_state"],
        query_start_loc=query_start_loc,
        attn_metadata=attn_metadata,
    )

    assert result == ("batched-output", "batched-final-state")
    assert metadata_calls == [attn_metadata]
    assert len(chunk_calls) == 1

    call = chunk_calls[0]
    assert set(call) == {
        "q",
        "k",
        "v",
        "g",
        "beta",
        "initial_state",
        "output_final_state",
        "cu_seqlens",
        "prebuilt_meta",
        "head_first",
        "use_qk_l2norm_in_kernel",
    }
    assert call["q"] is sentinels["query"]
    assert call["k"] is sentinels["key"]
    assert call["v"] is sentinels["value"]
    assert call["g"] is sentinels["g"]
    assert call["beta"] is sentinels["beta"]
    assert call["initial_state"] is sentinels["initial_state"]
    assert call["cu_seqlens"] is query_start_loc
    assert call["prebuilt_meta"] is sentinels["prebuilt_meta"]
    assert call["prebuilt_meta"] is not legacy_prebuilt_meta
    assert call["output_final_state"] is True
    assert call["head_first"] is False
    assert call["use_qk_l2norm_in_kernel"] is True


def test_batched_prefill_rejects_missing_fallback_metadata_before_kernel(
    monkeypatch,
):
    gdn = _import_gdn_helpers()
    chunk_calls = []
    monkeypatch.setattr(
        gdn,
        "chunk_gated_delta_rule",
        lambda **kwargs: chunk_calls.append(kwargs),
    )

    with pytest.raises(
        RuntimeError,
        match="non_spec_prefill_fallback_meta\\.chunk",
    ):
        gdn._run_batched_chunk_prefill(
            query=object(),
            key=object(),
            value=object(),
            g=object(),
            beta=object(),
            initial_state=object(),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            attn_metadata=SimpleNamespace(
                non_spec_prefill_fallback_meta=None,
            ),
        )

    assert chunk_calls == []


def test_eager_prefill_runtime_token_count_discards_padding():
    runtime_num_tokens = _import_gdn_helpers()._runtime_num_actual_tokens
    eager_metadata = SimpleNamespace(
        num_prefills=1,
        num_actual_tokens=274,
    )

    assert runtime_num_tokens(
        eager_metadata,
        torch.empty(512, 3),
    ) == 274


def test_traced_prefill_runtime_token_count_uses_tensor_shape(monkeypatch):
    runtime_num_tokens = _import_gdn_helpers()._runtime_num_actual_tokens
    stale_capture_metadata = SimpleNamespace(
        num_prefills=1,
        num_actual_tokens=1,
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    assert runtime_num_tokens(
        stale_capture_metadata,
        torch.empty(16384, 3),
    ) == 16384


def test_decode_runtime_token_count_keeps_unpadded_metadata_value():
    runtime_num_tokens = _import_gdn_helpers()._runtime_num_actual_tokens
    decode_metadata = SimpleNamespace(
        num_prefills=0,
        num_actual_tokens=1,
    )

    assert runtime_num_tokens(decode_metadata, torch.empty(2, 3)) == 1
