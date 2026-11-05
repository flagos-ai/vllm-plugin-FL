# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 integration and contract regressions."""

import importlib.util
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from _qwen38_cpu import ROOT, declarations

# Layout contracts.

# SPDX-License-Identifier: Apache-2.0


@pytest.mark.parametrize(
    "case,spec,prefills,decodes,expected_calls",
    [
        ("profile", False, 0, 0, 0),
        ("prefill", False, 2, 0, 0),
        ("spec_only", True, 0, 0, 0),
        ("mixed_prefill", True, 2, 0, 1),
        ("mixed_decode", True, 0, 1, 1),
    ],
)
def test_layout_provider_runs_only_before_mixed_native_gather(
    monkeypatch, case, spec, prefills, decodes, expected_calls
):
    class NativeMetadata:
        def __init__(self):
            self.spec_sequence_masks = object() if spec else None
            self.num_prefills = prefills
            self.num_decodes = decodes

    class NativeAttention(torch.nn.Module):
        def __init__(self, config, vllm_config, prefix, gqa_interleaved_layout):
            super().__init__()
            self.prefix = prefix

        def _forward_core(self, mixed_qkv, b, a, core_attn_out):
            self.received = (mixed_qkv, b, a, core_attn_out)
            return core_attn_out

    prefix = "model.layers.0.linear_attn"
    metadata = None if case == "profile" else {prefix: NativeMetadata()}
    context_module = ModuleType("vllm.forward_context")
    context_module.get_forward_context = lambda: SimpleNamespace(attn_metadata=metadata)
    context_module.is_forward_context_available = lambda: True
    native_module = ModuleType(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )
    native_module.GDNAttentionMetadata = NativeMetadata
    native_module.QwenGatedDeltaNetAttention = NativeAttention
    native_module.causal_conv1d_update = None
    native_module.is_conv_state_dim_first = None
    backend_module = ModuleType("vllm.v1.attention.backends.gdn_attn")
    backend_module.NULL_BLOCK_ID = 0
    backend_module.GDNAttentionBackend = type("NativeBackend", (), {})
    backend_module.GDNAttentionMetadataBuilder = type("NativeBuilder", (), {})
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)
    layout_calls, resolved = [], []

    def layout(value):
        layout_calls.append(value)
        return value.contiguous()

    def resolve(name):
        resolved.append(name)
        return layout if name == "qwen38_gdn_mixed_qkv" else lambda **kwargs: None

    dispatch_module = ModuleType("vllm_fl.dispatch")
    dispatch_module.resolve_op = resolve
    monkeypatch.setitem(sys.modules, context_module.__name__, context_module)
    monkeypatch.setitem(sys.modules, native_module.__name__, native_module)
    monkeypatch.setitem(sys.modules, dispatch_module.__name__, dispatch_module)
    module_name = "_qwen38_gdn_layout_test"
    path = ROOT / "vllm_fl/models/qwen3_8_flash_next/gdn.py"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    module_spec.loader.exec_module(module)
    layer = module.Qwen38GatedDeltaNetAttention(None, None, prefix=prefix)

    qkv = torch.arange(42 * 2048, dtype=torch.float32).reshape(42, 2048)[:, :1280]
    b, a, output = torch.empty(42, 6), torch.empty(42, 6), torch.empty(42, 6, 128)
    assert layer._forward_core(qkv, b, a, output) is output
    assert resolved == ["gdn_prefill", "gdn_packed_decode", "qwen38_gdn_mixed_qkv"]
    assert len(layout_calls) == expected_calls
    actual, sent_b, sent_a, sent_output = layer.received
    assert sent_b is b and sent_a is a and sent_output is output
    torch.testing.assert_close(actual, qkv, atol=0, rtol=0)
    if expected_calls:
        assert layout_calls[0] is qkv
        assert actual.is_contiguous()
    else:
        assert actual is qkv
        assert actual.stride() == (2048, 1)


def layout__metadata_classes(monkeypatch):
    """Execute pinned native metadata declarations with CPU framework stubs."""
    from dataclasses import dataclass

    from _qwen38_cpu import GenericBase, declarations

    native_path = (
        Path(importlib.util.find_spec("vllm").origin).parent
        / "v1/attention/backends/gdn_attn.py"
    )
    native = declarations(
        native_path,
        "_qwen38_native_gdn_metadata_cpu",
        {
            "dataclass": dataclass,
            "AttentionMetadataBuilder": GenericBase,
            "AttentionCGSupport": SimpleNamespace(UNIFORM_BATCH=2),
            "mamba_get_block_table_tensor": lambda table, *args: table,
            "async_tensor_h2d": lambda value, device: value.to(device),
            "split_decodes_and_prefills": lambda m, **kwargs: (
                int((torch.diff(m.query_start_loc_cpu) == 1).sum()),
                0,
                int((torch.diff(m.query_start_loc_cpu) == 1).sum()),
                0,
            ),
            "NULL_BLOCK_ID": 0,
        },
        names={"GDNAttentionMetadata", "GDNAttentionMetadataBuilder"},
    )
    local = declarations(
        ROOT / "vllm_fl/models/qwen3_8_flash_next/gdn.py",
        "_qwen38_local_gdn_metadata_cpu",
        {
            "GDNAttentionMetadataBuilder": native.GDNAttentionMetadataBuilder,
            "NULL_BLOCK_ID": 0,
        },
        names={"Qwen38GDNAttentionMetadataBuilder"},
    )
    return native.GDNAttentionMetadataBuilder, local.Qwen38GDNAttentionMetadataBuilder


def layout__metadata_builder(cls, k=3, graph=False):
    builder = cls.__new__(cls)
    builder.num_spec = k
    builder.use_spec_decode = k > 0
    builder.use_host_chunk_metadata = False
    builder.use_full_cuda_graph = graph
    builder.kv_cache_spec = object()
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none")
    )
    builder.decode_cudagraph_max_bs = 32
    builder.spec_state_indices_tensor = torch.empty((8, k + 1), dtype=torch.int32)
    builder.spec_sequence_masks = torch.empty(8, dtype=torch.bool)
    builder.spec_token_indx = torch.empty(32, dtype=torch.int32)
    builder.non_spec_token_indx = torch.empty(32, dtype=torch.int32)
    builder.spec_query_start_loc = torch.empty(9, dtype=torch.int32)
    builder.num_accepted_tokens = torch.empty(8, dtype=torch.int32)
    return builder


def layout__metadata_common(lengths, k=3, padded_tokens=0):
    starts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    requests = len(lengths)
    table = torch.arange(1, requests * (k + 1) + 1, dtype=torch.int32).reshape(
        requests, k + 1
    )
    if lengths[-1] == 0:
        table[-1].zero_()
    return SimpleNamespace(
        num_reqs=requests,
        num_actual_tokens=int(starts[-1]) + padded_tokens,
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        block_table_tensor=table,
        seq_lens=torch.tensor(lengths) + 20,
        compute_num_computed_tokens=lambda: torch.full((requests,), 20),
    )


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize(
    "lengths,counts",
    [
        ([1, 1, 1, 1, 0], [0, 0, 0, 0, -1]),
        ([1, 2, 3, 4, 0], [0, 1, 2, 3, -1]),
    ],
)
def test_gdn_k3_accepted_prefixes_survive_zero_draft_and_padding(
    monkeypatch, graph, lengths, counts
):
    native, local = layout__metadata_classes(monkeypatch)
    common = layout__metadata_common(lengths, padded_tokens=2)
    accepted = torch.tensor([1, 2, 3, 4, 1], dtype=torch.int32)
    counts = torch.tensor(counts, dtype=torch.int32)
    saved_counts = counts.clone()
    metadata = layout__metadata_builder(local, graph=graph).build(
        0, common, accepted, counts
    )
    # a accepted drafts means a+1 verified tokens, including the mandatory
    # target token. Native recurrent state and conv both read accepted-1.
    assert metadata.num_accepted_tokens[:4].tolist() == [1, 2, 3, 4]
    rows = torch.arange(4)
    selected = metadata.spec_state_indices_tensor[rows, accepted[:4] - 1]
    assert selected.tolist() == [1, 6, 11, 16]
    assert (
        metadata.spec_query_start_loc[:5].tolist()
        == common.query_start_loc[:5].tolist()
    )
    assert metadata.num_spec_decode_tokens == sum(lengths)
    assert metadata.num_spec_decodes == 4
    assert metadata.num_decodes == 0 and metadata.num_prefills == 0
    assert torch.equal(counts, saved_counts)
    if graph:
        assert metadata.spec_state_indices_tensor[-1].tolist() == [0] * 4
        assert metadata.num_accepted_tokens[-1] == 1
        assert not metadata.spec_sequence_masks[-1]
    if max(lengths) == 1:
        # Regression control: the unmodified 0.24 builder discards all accepted
        # offsets and selects column zero when the whole batch has no drafts.
        control = layout__metadata_builder(native).build(0, common, accepted, counts)
        assert control.num_accepted_tokens is None
        assert control.non_spec_state_indices_tensor[:4].tolist() == [1, 5, 9, 13]


@pytest.mark.parametrize(
    "k,counts,expected",
    [
        (3, [-1, 0, 3, -1], [-1, 1, 3, -1]),
        (0, [-1, 0, -1], [-1, 0, -1]),
        (3, None, None),
    ],
)
def test_gdn_classification_preserves_prefill_and_disabled_spec(
    monkeypatch, k, counts, expected
):
    native, local = layout__metadata_classes(monkeypatch)
    received = []

    def capture(self, prefix, common, accepted, markers, fast):
        received.append(markers)
        return "unchanged-native-result"

    monkeypatch.setattr(native, "build", capture)
    counts = None if counts is None else torch.tensor(counts, dtype=torch.int32)
    result = layout__metadata_builder(local, k=k).build(0, object(), None, counts)
    assert result == "unchanged-native-result"
    assert (None if received[0] is None else received[0].tolist()) == expected
    if k == 0 or counts is None:
        assert received[0] is counts


@pytest.mark.parametrize("graph", [False, True])
def test_gdn_k1_zero_current_draft_restores_prior_accepted_state(monkeypatch, graph):
    native, local = layout__metadata_classes(monkeypatch)
    common = layout__metadata_common([1, 1, 0], k=1)
    counts = torch.tensor([0, 0, -1], dtype=torch.int32)
    accepted = torch.tensor([1, 2, 1], dtype=torch.int32)
    saved_counts = counts.clone()
    saved_starts = common.query_start_loc.clone()
    metadata = layout__metadata_builder(local, k=1, graph=graph).build(
        0, common, accepted, counts
    )
    assert metadata.num_accepted_tokens[:2].tolist() == [1, 2]
    selected = metadata.spec_state_indices_tensor[torch.arange(2), accepted[:2] - 1]
    assert selected.tolist() == [1, 4]
    assert metadata.num_spec_decodes == 2
    assert metadata.num_spec_decode_tokens == 2
    assert metadata.num_decodes == 0 and metadata.num_prefills == 0
    assert metadata.spec_query_start_loc[:3].tolist() == [0, 1, 2]
    assert torch.equal(counts, saved_counts)
    assert torch.equal(common.query_start_loc, saved_starts)
    if graph:
        assert metadata.spec_state_indices_tensor[-1].tolist() == [0, 0]
        assert metadata.num_accepted_tokens[-1] == 1
        assert not metadata.spec_sequence_masks[-1]
    # The native classification defect precedes graph-buffer rebinding.
    # Keep this negative control eager: the fixed path above separately
    # exercises both CPU metadata graph modes, not a GPU graph launch.
    control = layout__metadata_builder(native, k=1).build(0, common, accepted, counts)
    assert control.num_accepted_tokens is None
    assert control.non_spec_state_indices_tensor[:2].tolist() == [1, 3]


@pytest.mark.parametrize(
    "enabled,vendor,device,spec,mode,backend,expected",
    [
        (True, "metax", "cuda", None, "none", "triton", True),
        (False, "metax", "cuda", None, "none", "triton", False),
        (True, "nvidia", "cuda", None, "none", "triton", False),
        (True, "metax", "cpu", None, "none", "triton", False),
        (True, "metax", "cuda", object(), "none", "triton", False),
        (True, "metax", "cuda", None, "align", "triton", False),
        (True, "metax", "cuda", None, "none", "cutedsl", False),
    ],
)
def test_gdn_metadata_reuse_is_explicit_and_keeps_unvalidated_paths(
    monkeypatch, enabled, vendor, device, spec, mode, backend, expected
):
    native, local = layout__metadata_classes(monkeypatch)

    def init(self, kv_spec, names, config, dev):
        self.speculative_config = config.speculative_config
        self.gdn_prefill_backend = backend

    monkeypatch.setattr(native, "__init__", init)
    monkeypatch.setitem(
        local.__init__.__globals__,
        "current_platform",
        SimpleNamespace(vendor_name=vendor),
    )
    config = SimpleNamespace(
        additional_config={"qwen38_reuse_gdn_metadata": enabled},
        speculative_config=spec,
        cache_config=SimpleNamespace(mamba_cache_mode=mode),
    )
    builder = local(object(), [], config, SimpleNamespace(type=device))
    assert builder.supports_update_block_table is expected


# Metadata contracts.

# SPDX-License-Identifier: Apache-2.0


@pytest.fixture(name="gdn_host")
def metadata_gdn_host(monkeypatch):
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


def metadata_metadata_for(native):
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
    builder.native_metadata = metadata_metadata_for(native)
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
    m = metadata_metadata_for(native)
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
