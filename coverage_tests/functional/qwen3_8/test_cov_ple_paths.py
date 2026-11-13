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

"""PLE n-gram hashing and layer execution paths.

The forward reference recomputes every n-gram id with a plain Python loop from
the documented splitmix64 multipliers and prime-sized head tables, so the
vectorized implementation is checked against an independent formulation.
The layer tests inject the convolution kernels and recording projections; the
expected merge order follows the documented decode-then-prefill layout.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.gpu import ple_layer


class _FakeEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_size=1, prefix=""):
        super().__init__()
        self.org_vocab_size = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_size = padding_size
        values = torch.arange(num_embeddings * embedding_dim, dtype=torch.float32)
        self.weight = nn.Parameter(values.reshape(num_embeddings, embedding_dim))
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=0, org_vocab_end_index=num_embeddings
        )

    def forward(self, ids):
        return self.weight[ids]


def _embedding_config(**overrides):
    values = dict(
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=0,
        vocab_size=10,
        split_ngram_parts=4,
        seed=1234,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=8,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _build_embedding(monkeypatch, **overrides):
    monkeypatch.setattr(ple_layer, "VocabParallelEmbedding", _FakeEmbedding)
    return ple_layer.Qwen3_8FlashNextNGramEmbedding(
        _embedding_config(**overrides),
        embedding_dim=4,
        ple_dense_layer_id=0,
        max_total_tokens=16,
        max_num_reqs=3,
        prefix="p",
    )


def _reference_ids(module, input_ids, query_start_loc, ngram_context):
    """Plain-Python formulation of the documented n-gram id construction."""
    ngram_size = module.ngram_size
    heads = module.heads_per_ngram
    eos = module.eos_token_id
    ids = [int(value) for value in input_ids]
    starts = [int(value) for value in query_start_loc]
    num_reqs = len(starts) - 1
    context = [[int(value) for value in row] for row in ngram_context.tolist()]
    multipliers = [int(value) for value in module.layer_multipliers.tolist()]
    sizes = [int(value) for value in module.ngram_heads_vocab_sizes.tolist()]
    offsets = [int(value) for value in module.ngram_heads_offsets.tolist()]
    rows = []
    for position, token in enumerate(ids):
        request = -1
        for index, start in enumerate(starts):
            if start <= position:
                request = index
        valid = 0 <= request < num_reqs
        safe = min(max(request, 0), num_reqs - 1)
        column = position - starts[safe]
        shifted = [token]
        in_segment = valid
        for shift in range(1, ngram_size):
            source_column = column - shift
            if source_column >= 0:
                previous = ids[max(position - shift, 0)]
            else:
                history_column = min(
                    max(ngram_size - 1 + source_column, 0), ngram_size - 2
                )
                previous = context[safe][history_column]
            in_segment = in_segment and previous != eos
            shifted.append(previous if in_segment else eos)
        mixed = shifted[0] * multipliers[0]
        row = []
        for ngram in range(2, ngram_size + 1):
            index = ngram - 1
            mixed ^= shifted[index] * multipliers[index]
            start = (ngram - 2) * heads
            for head in range(start, start + heads):
                value = mixed % sizes[head] + offsets[head]
                row.append(value if valid else eos)
        rows.append(row)
    return rows


def _reference_forward(module, input_ids, query_start_loc, ngram_context):
    rows = _reference_ids(module, input_ids, query_start_loc, ngram_context)
    weight = module.ngram_embedding.weight.detach()
    expected = []
    for row in rows:
        values = []
        for token_id in row:
            values.extend(weight[token_id].tolist())
        expected.append(values)
    return expected


class TestNGramForward:
    def test_two_requests_match_the_independent_reference(self, monkeypatch):
        module = _build_embedding(monkeypatch)
        input_ids = torch.tensor([5, 6, 7, 8, 9])
        query_start_loc = torch.tensor([0, 2, 5])
        context = torch.tensor([[1, 2], [3, 4]])
        result = module(input_ids, query_start_loc, context)
        assert result.shape == (5, 4)
        assert result.tolist() == _reference_forward(
            module, input_ids, query_start_loc, context
        )

    def test_history_window_and_eos_masking_match_the_reference(self, monkeypatch):
        module = _build_embedding(monkeypatch)
        input_ids = torch.tensor([5, 0, 6])
        query_start_loc = torch.tensor([0, 3])
        context = torch.tensor([[7, 8]])
        result = module(input_ids, query_start_loc, context)
        assert result.tolist() == _reference_forward(
            module, input_ids, query_start_loc, context
        )

    def test_padding_rows_past_the_request_boundary_use_eos(self, monkeypatch):
        module = _build_embedding(monkeypatch)
        input_ids = torch.tensor([5, 6, 7])
        query_start_loc = torch.tensor([0, 2])
        context = torch.tensor([[1, 2], [3, 4]])
        result = module(input_ids, query_start_loc, context)
        assert result.tolist() == _reference_forward(
            module, input_ids, query_start_loc, context
        )
        assert _reference_ids(module, input_ids, query_start_loc, context)[2] == [
            0,
            0,
            0,
            0,
        ]

    def test_empty_token_batch_keeps_the_embedding_width(self, monkeypatch):
        module = _build_embedding(monkeypatch)
        result = module(
            torch.empty(0, dtype=torch.long),
            torch.tensor([0, 0]),
            torch.zeros(1, 2, dtype=torch.long),
        )
        assert result.shape == (0, 4)

    def test_no_requests_and_no_tokens_is_an_empty_result(self, monkeypatch):
        module = _build_embedding(monkeypatch)
        result = module(
            torch.empty(0, dtype=torch.long),
            torch.tensor([0]),
            torch.empty(0, 2, dtype=torch.long),
        )
        assert result.shape == (0, 4)
        assert result.dtype is torch.long

    @pytest.mark.parametrize(
        ("input_ids", "query_start_loc", "context", "message"),
        [
            (
                torch.tensor([1, 2, 3]),
                torch.tensor([0, 2]),
                torch.zeros(1, 3),
                "wrong history width",
            ),
            (
                torch.tensor([1, 2]),
                torch.tensor([0, 2]),
                torch.zeros(0, 2),
                "one row per request",
            ),
            (
                torch.tensor([1, 2]),
                torch.tensor([0, 2]),
                torch.zeros(2),
                "one row per request",
            ),
            (
                torch.tensor([1, 2]),
                torch.tensor([0]),
                torch.zeros(0, 2),
                "token rows without a request boundary",
            ),
        ],
    )
    def test_context_shape_guards(
        self, monkeypatch, input_ids, query_start_loc, context, message
    ):
        module = _build_embedding(monkeypatch)
        with pytest.raises(ValueError, match=message):
            module(input_ids, query_start_loc, context)

    def test_token_and_request_workspace_limits(self, monkeypatch):
        module = _build_embedding(monkeypatch)
        with pytest.raises(ValueError, match="at most 16"):
            module(
                torch.ones(17, dtype=torch.long),
                torch.tensor([0, 17]),
                torch.zeros(1, 2),
            )
        with pytest.raises(ValueError, match="at most 3"):
            module(
                torch.ones(5, dtype=torch.long),
                torch.tensor([0, 1, 2, 3, 4, 5]),
                torch.zeros(5, 2),
            )


class _FakeLinear(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kwargs = kwargs


class _FakeNorm(nn.Module):
    def __init__(self, hidden_size, eps, group_size, dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @property
    def eps(self):
        return self.variance_epsilon


def _ple_config(**overrides):
    values = dict(
        hidden_size=4,
        hc_count=2,
        ple_conv_kernel_size=3,
        ngram_size=3,
        ple_embed_dim=6,
        rms_norm_eps=1e-6,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _ple_vllm_config(**overrides):
    values = dict(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_dtype="float32"),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=4),
        num_speculative_tokens=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _build_ple_layer(monkeypatch, **overrides):
    created = SimpleNamespace(
        embedding=None, key=None, value=None, norms=[], static_context={}
    )

    class FakeEmbedding(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args = args
            self.kwargs = kwargs
            created.embedding = self

    class DispatchLinear(_FakeLinear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if args[1] == 8:
                created.key = self
            else:
                created.value = self

    class FakeNorm(_FakeNorm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.norms.append(self)

    context = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context=created.static_context
        )
    )
    monkeypatch.setattr(ple_layer, "Qwen3_8FlashNextNGramEmbedding", FakeEmbedding)
    monkeypatch.setattr(ple_layer, "ReplicatedLinear", DispatchLinear)
    monkeypatch.setattr(ple_layer, "Qwen3_8FlashNextPLEGroupedNorm", FakeNorm)
    monkeypatch.setattr(ple_layer, "get_current_vllm_config", lambda: context)
    layer = ple_layer.Qwen3_8FlashNextPLELayer(
        overrides.pop("config", _ple_config()),
        vllm_config=_ple_vllm_config(),
        prefix="p",
        **overrides,
    )
    return layer, created


class TestPleLayerConstruction:
    def test_layer_geometry_and_registration(self, monkeypatch):
        layer, created = _build_ple_layer(monkeypatch, layer_idx=2)
        assert layer.hidden_size == 4
        assert layer.hc_count == 2
        assert layer.hc_hidden_size == 8
        assert layer.conv_state_len == 6
        assert layer.short_conv_dilation == 3
        assert layer.conv_kernel_size == 3
        assert layer.num_spec_tokens == 2
        assert layer.ple_dense_layer_id == 2
        assert layer.kv_cache[0].numel() == 0
        assert layer.activation == "silu"
        assert created.embedding.args[1] == 6
        assert created.embedding.args[2] == 2
        assert layer.conv1d.groups == 8
        assert layer.conv1d.padding == (6,)
        assert layer.conv1d.dilation == (3,)
        assert torch.count_nonzero(layer.conv1d.weight) == 0
        assert layer.conv1d.weight._no_reinit is True
        assert created.static_context["p"] is layer

    def test_explicit_dense_layer_id_wins(self, monkeypatch):
        layer, _ = _build_ple_layer(monkeypatch, layer_idx=1, ple_dense_layer_id=5)
        assert layer.ple_dense_layer_id == 5

    def test_duplicate_prefix_is_rejected(self, monkeypatch):
        class FakeNgram(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        static = {}
        monkeypatch.setattr(
            ple_layer,
            "get_current_vllm_config",
            lambda: SimpleNamespace(
                compilation_config=SimpleNamespace(static_forward_context=static)
            ),
        )
        monkeypatch.setattr(ple_layer, "Qwen3_8FlashNextNGramEmbedding", FakeNgram)
        monkeypatch.setattr(ple_layer, "ReplicatedLinear", _FakeLinear)
        monkeypatch.setattr(ple_layer, "Qwen3_8FlashNextPLEGroupedNorm", _FakeNorm)
        layer = ple_layer.Qwen3_8FlashNextPLELayer(
            _ple_config(), vllm_config=_ple_vllm_config(), prefix="dup"
        )
        assert static["dup"] is layer
        with pytest.raises(ValueError, match="Duplicate layer name"):
            ple_layer.Qwen3_8FlashNextPLELayer(
                _ple_config(), vllm_config=_ple_vllm_config(), prefix="dup"
            )

    def test_state_contracts_forward_cache_types(self, monkeypatch):
        layer, _ = _build_ple_layer(monkeypatch)
        recorded = {}

        def fake_dtype(model_dtype, cache_dtype):
            recorded["dtype"] = (model_dtype, cache_dtype)
            return (torch.bfloat16,)

        def fake_shape(*, tp_world_size, intermediate_size, conv_kernel):
            recorded["shape"] = (tp_world_size, intermediate_size, conv_kernel)
            return ((8, 6),)

        monkeypatch.setattr(
            ple_layer.MambaStateDtypeCalculator,
            "short_conv_state_dtype",
            staticmethod(fake_dtype),
        )
        monkeypatch.setattr(
            ple_layer.MambaStateShapeCalculator,
            "short_conv_state_shape",
            staticmethod(fake_shape),
        )
        assert layer.get_state_dtype() == (torch.bfloat16,)
        assert layer.get_state_shape() == ((8, 6),)
        assert recorded["dtype"] == (torch.bfloat16, "float32")
        # conv_state_len 6 + num_spec 2 + 1 == 9.
        assert recorded["shape"] == (1, 8, 9)
        assert layer.mamba_type is ple_layer.MambaAttentionBackendEnum.SHORT_CONV
        assert layer.get_attn_backend() is ple_layer.PleShortConvAttentionBackend


def _dispatch_layer(monkeypatch):
    layer, _ = _build_ple_layer(monkeypatch)
    layer.conv1d.weight.data.fill_(2.0)
    return layer


class TestShortConvKernels:
    def test_profiling_fallback_builds_an_empty_state(self, monkeypatch):
        layer = _dispatch_layer(monkeypatch)
        recorded = []

        def kernel(*args, **kwargs):
            recorded.append((args, kwargs))
            args[1].fill_(7)

        monkeypatch.setattr(ple_layer, "ple_short_conv_", kernel)
        output = layer._short_conv_fallback(torch.zeros(2, 4))
        assert output.tolist() == torch.full((2, 4), 7.0).tolist()
        args, kwargs = recorded[0]
        assert args[2].shape == (0, 4, layer.conv_state_len)
        assert kwargs["dilation"] == layer.short_conv_dilation

    def test_decode_batch_passes_initial_state(self, monkeypatch):
        layer = _dispatch_layer(monkeypatch)
        recorded = []
        monkeypatch.setattr(
            ple_layer,
            "ple_short_conv_",
            lambda *args, **kwargs: recorded.append((args, kwargs)),
        )
        layer._short_conv_dilated_decode_batched(
            torch.zeros(2, 4),
            "state",
            "weights",
            torch.tensor([3, 4]),
            torch.tensor([True, False]),
        )
        args, kwargs = recorded[0]
        assert torch.equal(args[4], torch.tensor([3, 4]))
        assert torch.equal(kwargs["has_initial_state"], torch.tensor([True, False]))
        assert kwargs["null_block_id"] == ple_layer.NULL_BLOCK_ID

    def test_spec_batch_slices_request_bounded_metadata(self, monkeypatch):
        layer = _dispatch_layer(monkeypatch)
        recorded = []
        monkeypatch.setattr(
            ple_layer,
            "ple_short_conv_",
            lambda *args, **kwargs: recorded.append((args, kwargs)),
        )
        layer._short_conv_dilated_spec_batched(
            torch.zeros(3, 4),
            "state",
            "weights",
            torch.tensor([10, 11, 12]),
            torch.tensor([0, 2, 3]),
            torch.tensor([2, 1, 1]),
            2,
        )
        args, kwargs = recorded[0]
        assert torch.equal(args[4], torch.tensor([10, 11, 12]))
        assert kwargs["query_start_loc"].tolist() == [0, 2, 3]
        assert kwargs["accepted_tokens"].tolist() == [2, 1, 1]
        assert kwargs["max_query_len"] == 2

    def test_mixed_spec_batch_merges_spec_first_then_non_spec(self, monkeypatch):
        layer = _dispatch_layer(monkeypatch)
        monkeypatch.setattr(
            layer,
            "_short_conv_dilated_spec_batched",
            lambda **kwargs: torch.full((2, 4), 1.0),
        )
        monkeypatch.setattr(
            layer,
            "_short_conv_dilated_decode_batched",
            lambda **kwargs: torch.full((1, 4), 2.0),
        )
        monkeypatch.setattr(
            layer,
            "_short_conv_dilated_prefill_batched",
            lambda **kwargs: torch.full((2, 4), 3.0),
        )
        metadata = SimpleNamespace(
            num_prefills=1,
            num_decodes=1,
            num_decode_tokens=1,
            num_prefill_tokens=2,
            num_actual_tokens=5,
            spec_sequence_masks=torch.ones(1, dtype=torch.bool),
            num_spec_decodes=1,
            spec_token_indx=torch.tensor([0, 1]),
            non_spec_token_indx=torch.tensor([2, 3, 4]),
            spec_state_indices_tensor=torch.tensor([10]),
            spec_query_start_loc=torch.tensor([0, 2]),
            num_accepted_tokens=torch.tensor([2]),
            spec_query_len=2,
            state_indices_tensor=torch.tensor([20, 21]),
            has_initial_states_d=torch.tensor([True]),
            has_initial_states_p=torch.tensor([False]),
            non_spec_query_start_loc=torch.tensor([0, 1, 3]),
            max_prefill_query_len=2,
        )
        output = layer._short_conv_dilated_dispatch(
            torch.zeros(5, 4), metadata, torch.zeros(1, 3, 4), torch.zeros(3, 4)
        )
        expected = torch.tensor([[1.0] * 4, [1.0] * 4, [2.0] * 4, [3.0] * 4, [3.0] * 4])
        assert output.tolist() == expected.tolist()

    def test_pure_spec_batch_returns_the_spec_output(self, monkeypatch):
        layer = _dispatch_layer(monkeypatch)
        monkeypatch.setattr(
            layer,
            "_short_conv_dilated_spec_batched",
            lambda **kwargs: torch.full((2, 4), 4.0),
        )
        metadata = SimpleNamespace(
            num_prefills=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefill_tokens=0,
            num_actual_tokens=2,
            spec_sequence_masks=torch.ones(1, dtype=torch.bool),
            num_spec_decodes=1,
            spec_state_indices_tensor=torch.tensor([10]),
            spec_query_start_loc=torch.tensor([0, 2]),
            num_accepted_tokens=torch.tensor([2]),
            spec_query_len=2,
            state_indices_tensor=torch.empty(0, dtype=torch.int32),
        )
        output = layer._short_conv_dilated_dispatch(
            torch.zeros(2, 4), metadata, torch.zeros(1, 3, 4), torch.zeros(3, 4)
        )
        assert output.tolist() == torch.full((2, 4), 4.0).tolist()

    def test_short_conv_slices_the_cache_to_the_required_state(self, monkeypatch):
        layer = _dispatch_layer(monkeypatch)
        layer.prefix = "p"
        layer.num_spec_tokens = 2
        layer.kv_cache = (torch.arange(9 * 4, dtype=torch.float32).reshape(1, 4, 9),)
        monkeypatch.setattr(ple_layer, "is_conv_state_dim_first", lambda: True)
        recorded = {}

        def dispatch(inputs, metadata, conv_state, conv_weights, output=None):
            recorded["state"] = conv_state
            recorded["weights"] = conv_weights
            return inputs

        monkeypatch.setattr(layer, "_short_conv_dilated_dispatch", dispatch)
        metadata = object.__new__(ple_layer.PleShortConvAttentionMetadata)
        monkeypatch.setattr(
            ple_layer,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={"p": metadata}),
        )
        assert layer._short_conv(torch.zeros(2, 4)) is not None
        # state capacity is conv_state_len 6 + num_spec 2.
        assert recorded["state"].shape == (1, 4, 8)
        assert recorded["weights"].shape == (8, 3)


class TestPleLayerForward:
    def test_gated_value_and_convolution_are_added(self, monkeypatch):
        layer, _ = _build_ple_layer(monkeypatch)

        class FakeEmbedding(nn.Module):
            def forward(self, *args):
                return torch.zeros(2, 6)

        class FakeProjection(nn.Module):
            def __init__(self, value, width):
                super().__init__()
                self.value = value
                self.width = width

            def forward(self, embeddings):
                return (
                    torch.full((embeddings.shape[0], self.width), self.value),
                    None,
                )

        layer.ple_embedding = FakeEmbedding()
        layer.key_proj = FakeProjection(1.0, 8)
        layer.value_proj = FakeProjection(2.0, 4)
        gated = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
        normalized = torch.ones(2, 2, 4)
        calls = []

        def fake_gate_norm(*args):
            calls.append(args)
            return gated, normalized

        monkeypatch.setattr(ple_layer, "ple_gate_norm", fake_gate_norm)
        recorded = []

        def fake_conv(inputs, output, layer_name):
            recorded.append((inputs, layer_name))
            output.fill_(1)

        monkeypatch.setattr(
            torch.ops.vllm,
            "qwen3_8_flash_next_ple_short_conv",
            fake_conv,
            raising=False,
        )
        hidden = torch.zeros(2, 8)
        out = layer.forward(
            hidden,
            torch.tensor([1, 2]),
            torch.tensor([0, 2]),
            torch.zeros(1, 2),
        )
        assert out.tolist() == (gated.flatten(-2) + 1).tolist()
        assert calls[0][0].shape == (2, 2, 4)
        assert calls[0][1].shape == (2, 2, 4)
        assert calls[0][2].shape == (2, 4)
        assert calls[0][6] == 2
        assert recorded[0][0].tolist() == torch.ones(2, 8).tolist()
        assert recorded[0][1] == "p"
