# SPDX-License-Identifier: Apache-2.0
"""CPU branch contracts for the Qwen-local GDN layout boundary."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


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
    native_module = ModuleType(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )
    native_module.GDNAttentionMetadata = NativeMetadata
    native_module.QwenGatedDeltaNetAttention = NativeAttention
    native_module.causal_conv1d_update = None
    native_module.is_conv_state_dim_first = None
    backend_module = ModuleType("vllm.v1.attention.backends.gdn_attn")
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


def _metadata_classes(monkeypatch):
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
        {"GDNAttentionMetadataBuilder": native.GDNAttentionMetadataBuilder},
        names={"Qwen38GDNAttentionMetadataBuilder"},
    )
    return native.GDNAttentionMetadataBuilder, local.Qwen38GDNAttentionMetadataBuilder


def _metadata_builder(cls, k=3, graph=False):
    builder = cls.__new__(cls)
    builder.num_spec = k
    builder.use_spec_decode = k > 0
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


def _metadata_common(lengths, k=3, padded_tokens=0):
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
    native, local = _metadata_classes(monkeypatch)
    common = _metadata_common(lengths, padded_tokens=2)
    accepted = torch.tensor([1, 2, 3, 4, 1], dtype=torch.int32)
    counts = torch.tensor(counts, dtype=torch.int32)
    saved_counts = counts.clone()
    metadata = _metadata_builder(local, graph=graph).build(0, common, accepted, counts)
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
        control = _metadata_builder(native).build(0, common, accepted, counts)
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
    native, local = _metadata_classes(monkeypatch)
    received = []

    def capture(self, prefix, common, accepted, markers, fast):
        received.append(markers)
        return "unchanged-native-result"

    monkeypatch.setattr(native, "build", capture)
    counts = None if counts is None else torch.tensor(counts, dtype=torch.int32)
    result = _metadata_builder(local, k=k).build(0, object(), None, counts)
    assert result == "unchanged-native-result"
    assert (None if received[0] is None else received[0].tolist()) == expected
    if k == 0 or counts is None:
        assert received[0] is counts


@pytest.mark.parametrize("graph", [False, True])
def test_gdn_k1_zero_current_draft_restores_prior_accepted_state(monkeypatch, graph):
    native, local = _metadata_classes(monkeypatch)
    common = _metadata_common([1, 1, 0], k=1)
    counts = torch.tensor([0, 0, -1], dtype=torch.int32)
    accepted = torch.tensor([1, 2, 1], dtype=torch.int32)
    saved_counts = counts.clone()
    saved_starts = common.query_start_loc.clone()
    metadata = _metadata_builder(local, k=1, graph=graph).build(
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
    control = _metadata_builder(native, k=1).build(0, common, accepted, counts)
    assert control.num_accepted_tokens is None
    assert control.non_spec_state_indices_tensor[:2].tolist() == [1, 3]
