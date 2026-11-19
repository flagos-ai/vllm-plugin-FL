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

"""PLE layer hash generation, metadata dispatch and load bookkeeping.

The n-gram hash tables are deterministic functions of the config seed, so the
tests recompute them with a splitmix64 loop written from the documented
constants. The short-conv dispatch is exercised with recording stand-ins for
the FlagGems kernels, which keeps the Python dispatch/token-merge logic pure
while asserting exactly which kernel receives which metadata.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.gpu import ple_layer

_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


def _splitmix_reference(value):
    value = (value + _GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _M1) & _MASK64
    value = ((value ^ (value >> 27)) * _M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


class TestHashPrimitives:
    def test_splitmix_matches_known_vector_and_reference(self):
        # Standard splitmix64 output of zero.
        """Verifies splitmix matches known vector and reference."""
        assert ple_layer._splitmix64(0) == 0xE220A8397B1DCDAF
        for value in (1, 2, 1234, _GAMMA, (1 << 63)):
            assert ple_layer._splitmix64(value) == _splitmix_reference(value)

    @pytest.mark.parametrize("value", [2, 3, 5, 7, 11, 97, 2**61 - 1])
    def test_primes_are_accepted(self, value):
        """Verifies primes are accepted."""
        assert ple_layer._is_prime_64(value) is True

    @pytest.mark.parametrize("value", [-3, 0, 1, 4, 9, 15, 561, 2**62 - 1])
    def test_composites_and_small_numbers_are_rejected(self, value):
        """Verifies composites and small numbers are rejected."""
        assert ple_layer._is_prime_64(value) is False

    @pytest.mark.parametrize(
        ("start", "count", "expected"),
        [(0, 0, 0), (0, 1, 2), (1, 1, 2), (2, 1, 3), (10, 1, 11), (10, 3, 17)],
    )
    def test_nth_prime_walk(self, start, count, expected):
        """Verifies nth prime walk."""
        assert ple_layer._nth_prime_after(start, count) == expected


class _FakeEmbedding(nn.Module):
    created: list = []

    def __init__(self, num_embeddings, embedding_dim, padding_size=1, prefix=""):
        super().__init__()
        self.org_vocab_size = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_size = padding_size
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=0, org_vocab_end_index=num_embeddings
        )
        _FakeEmbedding.created.append(
            (num_embeddings, embedding_dim, padding_size, prefix)
        )


def _embedding_config(**overrides):
    values = dict(
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=0,
        vocab_size=100,
        split_ngram_parts=2,
        seed=1234,
        ngram_vocab_size_base=97,
        make_ngram_vocab_size_divisible_by=8,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture()
def fake_embedding(monkeypatch):
    _FakeEmbedding.created = []
    monkeypatch.setattr(ple_layer, "VocabParallelEmbedding", _FakeEmbedding)
    return _FakeEmbedding


def _build_embedding(**overrides):
    return ple_layer.Qwen3_8FlashNextNGramEmbedding(
        _embedding_config(**overrides),
        embedding_dim=4,
        ple_dense_layer_id=0,
        max_total_tokens=32,
        max_num_reqs=4,
        prefix="p",
    )


class TestNGramEmbeddingConstruction:
    def test_hash_tables_follow_the_seeded_contract(self, fake_embedding):
        """Verifies hash tables follow the seeded contract."""
        module = _build_embedding()
        assert module.ngram_heads == 4
        assert module.head_dim == 1
        assert module.unigram_vocab_size == 100

        max_multiplier = ((1 << 63) - 1) // 100
        half_bound = max(1, max_multiplier // 2)
        expected_multipliers = [
            2 * (_splitmix_reference(1234 + _GAMMA * (index + 1)) % half_bound) + 1
            for index in range(3)
        ]
        assert module.layer_multipliers.tolist() == expected_multipliers
        assert all(value % 2 == 1 for value in expected_multipliers)

        expected_sizes = [ple_layer._nth_prime_after(96, head + 1) for head in range(4)]
        assert module.ngram_heads_vocab_sizes.tolist() == expected_sizes
        expected_offsets = [0]
        for size in expected_sizes[:-1]:
            expected_offsets.append(expected_offsets[-1] + size)
        assert module.ngram_heads_offsets.tolist() == expected_offsets

        divisor = 8
        total = sum(expected_sizes)
        padded = math.ceil(total / divisor) * divisor
        assert fake_embedding.created == [(padded, 1, divisor, "p.ngram_embedding")]
        assert module.positions_buffer.tolist() == list(range(32))

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"ngram_size": 1}, "ngram_size must be >= 2"),
            ({"heads_per_ngram": 0}, "heads_per_ngram must be > 0"),
            ({}, "divisible by total ngram heads"),
            ({"split_ngram_parts": 0}, "split_ngram_parts must be positive"),
        ],
    )
    def test_rejects_invalid_geometry(self, fake_embedding, overrides, message):
        """Verifies rejects invalid geometry."""
        kwargs = dict(overrides)
        embedding_dim = 4
        if overrides == {}:
            embedding_dim = 2
        with pytest.raises(ValueError, match=message):
            ple_layer.Qwen3_8FlashNextNGramEmbedding(
                _embedding_config(**kwargs),
                embedding_dim=embedding_dim,
                ple_dense_layer_id=0,
                max_total_tokens=32,
                max_num_reqs=4,
                prefix="p",
            )

    def test_required_shards_cover_the_local_tp_range(self, fake_embedding):
        """Verifies required shards cover the local tp range."""
        module = _build_embedding()
        assert module._required_embedding_shards() == {0, 1}


def _shard_owner(org_vocab_size, split, start, end):
    module = object.__new__(ple_layer.Qwen3_8FlashNextNGramEmbedding)
    nn.Module.__init__(module)
    module.split_ngram_parts = split
    module.ngram_embedding = SimpleNamespace(
        org_vocab_size=org_vocab_size,
        embedding_dim=2,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=start, org_vocab_end_index=end
        ),
    )
    return module


class TestRequiredShards:
    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            (0, 10, {0, 1}),
            (3, 7, {0, 1}),
            (6, 8, {1}),
            (5, 5, set()),
            (0, 20, {0, 1}),
        ],
    )
    def test_shard_ids_are_clamped_to_the_original_vocabulary(
        self, start, end, expected
    ):
        """Verifies shard ids are clamped to the original vocabulary."""
        owner = _shard_owner(10, 2, start, end)
        assert owner._required_embedding_shards() == expected


class TestLoadValidation:
    @staticmethod
    def _root():
        root = nn.Module()
        root.add_module("ple", _shard_owner(10, 2, 0, 10))
        return root

    def test_complete_load_is_released_after_success(self):
        """Verifies complete load is released after success."""
        root = self._root()

        @ple_layer.validate_ple_embedding_weights
        def load(model, weights):
            for module in model.modules():
                if isinstance(module, ple_layer.Qwen3_8FlashNextNGramEmbedding):
                    assert module._ple_embedding_load_state == {
                        "shards": set(),
                        "full": False,
                    }
                    module._ple_embedding_load_state["shards"].update({0, 1})
            return "ok"

        assert load(root, None) == "ok"
        assert not hasattr(root.ple, "_ple_embedding_load_state")

    def test_missing_shards_raise_and_state_is_cleaned_up(self):
        """Verifies missing shards raise and state is cleaned up."""
        root = self._root()

        @ple_layer.validate_ple_embedding_weights
        def load(model, weights):
            for module in model.modules():
                if isinstance(module, ple_layer.Qwen3_8FlashNextNGramEmbedding):
                    module._ple_embedding_load_state["shards"].add(0)
            return "ok"

        with pytest.raises(ValueError, match="Missing PLE embedding shards"):
            load(root, None)
        assert not hasattr(root.ple, "_ple_embedding_load_state")


def _layer():
    layer = object.__new__(ple_layer.Qwen3_8FlashNextPLELayer)
    nn.Module.__init__(layer)
    layer.conv_state_len = 9
    layer.conv_kernel_size = 4
    layer.short_conv_dilation = 3
    return layer


def _preader(name, calls, *, fused=False, fill=None):
    def kernel(*args, **kwargs):
        calls.append((name, args, kwargs))
        if fill is not None:
            args[1].fill_(fill)
        return fused

    return kernel


class TestShortConvDispatch:
    def test_pure_prefill_uses_the_fused_prefill_then_fallback(self, monkeypatch):
        """Verifies pure prefill uses the fused prefill then fallback."""
        calls = []
        monkeypatch.setattr(
            ple_layer,
            "ple_prefill_short_conv_",
            _preader("prefill", calls, fused=False),
        )
        monkeypatch.setattr(
            ple_layer, "ple_short_conv_", _preader("conv", calls, fill=7)
        )
        layer = _layer()
        metadata = SimpleNamespace(
            num_prefills=1,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefill_tokens=3,
            num_actual_tokens=3,
            spec_sequence_masks=None,
            state_indices_tensor=torch.tensor([2], dtype=torch.int32),
            has_initial_states_p=torch.tensor([True]),
            non_spec_query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            max_prefill_query_len=3,
        )
        output = layer._short_conv_dilated_dispatch(
            torch.zeros(3, 4),
            metadata,
            torch.zeros(1, 3, 9),
            torch.zeros(3, 4),
        )
        assert output[:, 0].tolist() == [7, 7, 7]
        assert [name for name, _, _ in calls] == ["prefill", "conv"]
        prefill = calls[0][2]
        assert prefill["num_prefills"] == 1
        assert prefill["state_len"] == 9
        assert prefill["kernel_width"] == 4
        assert prefill["dilation"] == 3
        assert prefill["min_tokens"] == ple_layer.PLE_FUSION_MIN_TOKENS
        conv = calls[1][2]
        assert conv["query_start_loc"].tolist() == [0, 3]
        assert conv["has_initial_state"].tolist() == [True]

    def test_pure_prefill_with_output_buffer_returns_none(self, monkeypatch):
        """Verifies pure prefill with output buffer returns none."""
        calls = []
        monkeypatch.setattr(
            ple_layer, "ple_prefill_short_conv_", _preader("prefill", calls)
        )
        monkeypatch.setattr(
            ple_layer, "ple_short_conv_", _preader("conv", calls, fill=5)
        )
        metadata = SimpleNamespace(
            num_prefills=1,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefill_tokens=3,
            num_actual_tokens=3,
            spec_sequence_masks=None,
            state_indices_tensor=torch.tensor([2], dtype=torch.int32),
            has_initial_states_p=torch.tensor([True]),
            non_spec_query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            max_prefill_query_len=3,
        )
        buffer = torch.empty(3, 4)
        result = _layer()._short_conv_dilated_dispatch(
            torch.zeros(3, 4),
            metadata,
            torch.zeros(1, 3, 9),
            torch.zeros(3, 4),
            output=buffer,
        )
        assert result is None
        assert buffer[:, 0].tolist() == [5, 5, 5]

    def test_empty_prefill_returns_an_empty_result(self, monkeypatch):
        """Verifies empty prefill returns an empty result."""
        monkeypatch.setattr(
            ple_layer, "ple_prefill_short_conv_", _preader("prefill", [])
        )
        metadata = SimpleNamespace(
            num_prefills=1,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefill_tokens=0,
            num_actual_tokens=0,
            spec_sequence_masks=None,
            state_indices_tensor=torch.tensor([2], dtype=torch.int32),
            has_initial_states_p=torch.tensor([True]),
            non_spec_query_start_loc=torch.tensor([0, 0], dtype=torch.int32),
            max_prefill_query_len=0,
        )
        result = _layer()._short_conv_dilated_dispatch(
            torch.zeros(0, 4),
            metadata,
            torch.zeros(1, 3, 9),
            torch.zeros(3, 4),
        )
        assert result.shape == (0, 4)

    def test_speculative_only_batch_uses_varlen_conv(self, monkeypatch):
        """Verifies speculative only batch uses varlen conv."""
        calls = []
        monkeypatch.setattr(
            ple_layer, "ple_short_conv_", _preader("conv", calls, fill=9)
        )
        metadata = SimpleNamespace(
            num_prefills=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefill_tokens=0,
            num_actual_tokens=2,
            spec_sequence_masks=torch.ones(1, dtype=torch.bool),
            num_spec_decodes=1,
            spec_state_indices_tensor=torch.tensor([3], dtype=torch.int32),
            spec_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            num_accepted_tokens=torch.tensor([2], dtype=torch.int32),
            spec_query_len=2,
            state_indices_tensor=torch.empty(0, dtype=torch.int32),
        )
        output = _layer()._short_conv_dilated_dispatch(
            torch.zeros(2, 4),
            metadata,
            torch.zeros(1, 3, 9),
            torch.zeros(3, 4),
        )
        assert output[:, 0].tolist() == [9, 9]
        assert len(calls) == 1
        kwargs = calls[0][2]
        assert calls[0][1][4].tolist() == [3]
        assert kwargs["query_start_loc"].tolist() == [0, 2]
        assert kwargs["accepted_tokens"].tolist() == [2]
        assert kwargs["max_query_len"] == 2

    def test_decode_only_batch_passes_initial_state_flags(self, monkeypatch):
        """Verifies decode only batch passes initial state flags."""
        calls = []
        monkeypatch.setattr(
            ple_layer, "ple_short_conv_", _preader("conv", calls, fill=4)
        )
        metadata = SimpleNamespace(
            num_prefills=0,
            num_decodes=2,
            num_decode_tokens=2,
            num_prefill_tokens=0,
            num_actual_tokens=2,
            spec_sequence_masks=None,
            state_indices_tensor=torch.tensor([4, 5], dtype=torch.int32),
            has_initial_states_d=torch.tensor([True, False]),
        )
        output = _layer()._short_conv_dilated_dispatch(
            torch.zeros(2, 4),
            metadata,
            torch.zeros(1, 3, 9),
            torch.zeros(3, 4),
        )
        assert output[:, 0].tolist() == [4, 4]
        kwargs = calls[0][2]
        assert calls[0][1][4].tolist() == [4, 5]
        assert kwargs["has_initial_state"].tolist() == [True, False]
        assert kwargs["null_block_id"] == ple_layer.NULL_BLOCK_ID


class TestShortConvGuards:
    def _configured_layer(self):
        layer = object.__new__(ple_layer.Qwen3_8FlashNextPLELayer)
        nn.Module.__init__(layer)
        layer.prefix = "p"
        layer.conv_state_len = 4
        layer.num_spec_tokens = 0
        layer.conv1d = SimpleNamespace(weight=torch.zeros(4, 1, 3))
        layer.kv_cache = (torch.zeros(1, 3, 2),)
        layer._short_conv_fallback = lambda inputs: "fallback"
        return layer

    def test_missing_attn_metadata_uses_profiling_fallback(self, monkeypatch):
        """Verifies missing attn metadata uses profiling fallback."""
        monkeypatch.setattr(
            ple_layer,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=None),
        )
        assert self._configured_layer()._short_conv(torch.zeros(1, 4)) == "fallback"

    def test_non_dict_metadata_is_rejected(self, monkeypatch):
        """Verifies non dict metadata is rejected."""
        monkeypatch.setattr(
            ple_layer,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=object()),
        )
        with pytest.raises(RuntimeError, match="per-layer attention metadata dict"):
            self._configured_layer()._short_conv(torch.zeros(1, 4))

    def test_missing_layer_key_is_rejected(self, monkeypatch):
        """Verifies missing layer key is rejected."""
        monkeypatch.setattr(
            ple_layer,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={}),
        )
        with pytest.raises(RuntimeError, match="Missing short-conv metadata"):
            self._configured_layer()._short_conv(torch.zeros(1, 4))

    def test_wrong_metadata_type_is_rejected(self, monkeypatch):
        """Verifies wrong metadata type is rejected."""
        monkeypatch.setattr(
            ple_layer,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={"p": object()}),
        )
        with pytest.raises(TypeError, match="Expected PleShortConvAttentionMetadata"):
            self._configured_layer()._short_conv(torch.zeros(1, 4))

    def test_undersized_cache_is_rejected(self, monkeypatch):
        """Verifies undersized cache is rejected."""
        metadata = object.__new__(ple_layer.PleShortConvAttentionMetadata)
        monkeypatch.setattr(
            ple_layer,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={"p": metadata}),
        )
        monkeypatch.setattr(ple_layer, "is_conv_state_dim_first", lambda: True)
        with pytest.raises(RuntimeError, match="smaller than expected"):
            self._configured_layer()._short_conv(torch.zeros(1, 4))

    def test_forward_requires_matching_token_lengths(self):
        """Verifies forward requires matching token lengths."""
        layer = object.__new__(ple_layer.Qwen3_8FlashNextPLELayer)
        nn.Module.__init__(layer)
        with pytest.raises(ValueError, match="same token length"):
            layer.forward(
                torch.zeros(2, 4),
                torch.zeros(3, dtype=torch.long),
                torch.zeros(2, dtype=torch.long),
                torch.zeros(1, 2),
            )


class TestCustomOpBoundary:
    def test_op_copies_layer_result_into_output(self, monkeypatch):
        """Verifies op copies layer result into output."""

        def short_conv(inputs, output=None):
            return torch.full_like(inputs, 3)

        monkeypatch.setattr(
            ple_layer,
            "get_forward_context",
            lambda: SimpleNamespace(
                no_compile_layers={"p": SimpleNamespace(_short_conv=short_conv)}
            ),
        )
        output = torch.zeros(2, 4)
        ple_layer.qwen3_8_flash_next_ple_short_conv(torch.zeros(2, 4), output, "p")
        assert (output == 3).all()

    def test_op_leaves_output_when_layer_returns_none(self, monkeypatch):
        """Verifies op leaves output when layer returns none."""
        monkeypatch.setattr(
            ple_layer,
            "get_forward_context",
            lambda: SimpleNamespace(
                no_compile_layers={
                    "p": SimpleNamespace(_short_conv=lambda inputs, output=None: None)
                }
            ),
        )
        output = torch.full((2, 4), 6.0)
        ple_layer.qwen3_8_flash_next_ple_short_conv(torch.zeros(2, 4), output, "p")
        assert (output == 6).all()

    def test_fake_op_is_inert(self):
        """Verifies fake op is inert."""
        assert (
            ple_layer.qwen3_8_flash_next_ple_short_conv_fake(
                torch.zeros(1), torch.zeros(1), "p"
            )
            is None
        )
