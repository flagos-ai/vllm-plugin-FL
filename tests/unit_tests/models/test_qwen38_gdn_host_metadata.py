# SPDX-License-Identifier: Apache-2.0
"""Host metadata must preserve the existing GDN kernel selection and state slots."""

import importlib.util
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from _qwen38_cpu import ROOT, declarations


@pytest.fixture
def gdn_host(monkeypatch):
    native_path = (
        Path(importlib.util.find_spec("vllm").origin).parent
        / "v1/attention/backends/gdn_attn.py"
    )
    native = declarations(
        native_path, "_gdn_host_native", names={"GDNAttentionMetadata"}
    )

    class Parent:
        def build(self, *args):
            return self.native_metadata

    context = SimpleNamespace(available=True, attn_metadata=None, calls=[])
    mod = declarations(
        ROOT / "vllm_fl/models/qwen3_8_flash_next/gdn.py",
        "_gdn_host_local",
        {
            "GDNAttentionMetadata": native.GDNAttentionMetadata,
            "GDNAttentionMetadataBuilder": Parent,
            "fields": fields,
            "NULL_BLOCK_ID": 0,
            "resolve_op": lambda name: lambda **kw: context.calls.append(kw),
            "get_forward_context": lambda: context,
            "is_forward_context_available": lambda: context.available,
        },
        names={
            "chunk_skew_bucket_from_query_starts",
            "Qwen38GDNAttentionMetadata",
            "Qwen38GDNAttentionMetadataBuilder",
            "_FGVPrefill",
        },
    )
    utils = ModuleType("vllm.model_executor.layers.fla.ops.utils")
    utils.FLA_CHUNK_SIZE = 64
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    return mod, native.GDNAttentionMetadata, context


@pytest.mark.parametrize(
    "counts,expected",
    [
        ([100, 100], 0),
        ([115, 85], 0),
        ([116, 84], 1),
        ([150, 50], 1),
        ([151, 49], 2),
        ([250, 50, 0], 2),
        ([251, 49, 0], 3),
        ([0, 0], 0),
        ([129], 0),
    ],
)
def test_host_bucket_preserves_legacy_boundaries(gdn_host, counts, expected):
    mod, _, _ = gdn_host
    lengths = [1, 1] + [64 * n for n in counts]
    starts = [0] + torch.tensor(lengths).cumsum(0).tolist()
    assert (
        mod.chunk_skew_bucket_from_query_starts(starts, 2, len(counts), 64) == expected
    )


def test_host_bucket_matches_device_statistics_for_irregular_lengths(gdn_host):
    mod, _, _ = gdn_host
    rng = torch.Generator().manual_seed(248320)
    for n in [1, 2, 3, 7, 64]:
        for _ in range(20):
            lengths = torch.randint(0, 8193, (n,), generator=rng)
            counts = (lengths + 63) // 64
            max_nt, sum_nt = int(counts.max()), int(counts.sum())
            ratio = max_nt / (sum_nt / n) if sum_nt else 1
            expected = (
                0
                if n <= 1 or ratio <= 1.15
                else 1
                if ratio <= 1.5
                else 2
                if ratio <= 2.5
                else 3
            )
            starts = [0, 1, 2] + (lengths.cumsum(0) + 2).tolist()
            assert mod.chunk_skew_bucket_from_query_starts(starts, 2, n, 64) == expected


@pytest.mark.parametrize(
    "starts,d,p,c",
    [([0, 1], -1, 1, 64), ([0, 1], 0, 2, 64), ([0, 2, 1], 0, 2, 64), ([0, 1], 0, 1, 0)],
)
def test_invalid_host_geometry_rejected(gdn_host, starts, d, p, c):
    with pytest.raises(ValueError):
        gdn_host[0].chunk_skew_bucket_from_query_starts(starts, d, p, c)


def metadata_for(native):
    values = {f.name: None for f in fields(native)}
    values.update(
        num_prefills=2,
        num_decodes=2,
        num_spec_decodes=0,
        chunk_indices=torch.tensor([[0, 0], [1, 0], [1, 1]], dtype=torch.int32),
        chunk_offsets=torch.tensor([0, 1, 3], dtype=torch.int32),
        non_spec_query_start_loc=torch.tensor([0, 1, 2, 65, 193], dtype=torch.int32),
    )
    return native(**values)


def test_builder_and_group_rebinding_preserve_host_hint_and_independent_slots(gdn_host):
    mod, native, _ = gdn_host
    builder = mod.Qwen38GDNAttentionMetadataBuilder.__new__(
        mod.Qwen38GDNAttentionMetadataBuilder
    )
    builder.use_spec_decode = False
    builder.use_host_chunk_metadata = True
    builder.supports_update_block_table = True
    builder.use_full_cuda_graph = False
    builder.native_metadata = metadata_for(native)
    common = SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 1, 2, 65, 193]))
    m = builder.build(0, common)
    assert m.chunk_skew_bucket == 1
    assert m.chunk_indices is builder.native_metadata.chunk_indices
    assert m.chunk_offsets is builder.native_metadata.chunk_offsets
    a = builder.update_block_table(m, torch.tensor([[1], [2], [3], [4]]), None)
    b = builder.update_block_table(m, torch.tensor([[5], [6], [7], [8]]), None)
    assert type(a) is type(m) is type(b)
    assert a.chunk_skew_bucket == b.chunk_skew_bucket == 1
    assert a.prefill_state_indices.tolist() == [3, 4]
    assert b.prefill_state_indices.tolist() == [7, 8]
    assert a.chunk_offsets is b.chunk_offsets is m.chunk_offsets
    builder.native_metadata = replace(builder.native_metadata, chunk_indices=None)
    with pytest.raises(ValueError, match="precomputed"):
        builder.build(0, common)


@pytest.mark.parametrize(
    "bucket,available", [(0, True), (3, True), (None, True), (0, False)]
)
def test_prefill_optional_hint_and_no_context_fallback(gdn_host, bucket, available):
    mod, native, context = gdn_host
    m = metadata_for(native)
    values = {f.name: getattr(m, f.name) for f in fields(m)}
    context.attn_metadata = {
        "layer": mod.Qwen38GDNAttentionMetadata(**values, chunk_skew_bucket=bucket)
    }
    context.available = available
    layer = mod._FGVPrefill("layer")
    x = torch.empty(1)
    layer(
        x,
        x,
        x,
        x,
        x,
        None,
        True,
        chunk_indices=m.chunk_indices,
        chunk_offsets=m.chunk_offsets,
    )
    sent = context.calls[-1]
    assert sent["chunk_indices"] is m.chunk_indices
    assert sent["chunk_offsets"] is m.chunk_offsets
    if bucket is not None and available:
        assert sent["chunk_skew_bucket"] == bucket
    else:
        assert "chunk_skew_bucket" not in sent
