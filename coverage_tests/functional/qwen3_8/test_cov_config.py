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

"""Pure-CPU contracts for ``vllm_fl/models/qwen3_8_flash_next/config.py``.

The config object owns the model's construction invariants: HyperConnection
stream counts, PLE n-gram table shape, 1-based PLE layer ids and the
QSA indexer budget. Every expected value below is computed from those
documented rules, not by re-reading the config object under test.
"""

import pytest

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.config import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextTextConfig,
    Qwen3_8FlashNextVisionConfig,
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
    Qwen4ExpVisionConfig,
)

_BASE = dict(
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=8,
    vocab_size=64,
    eos_token_id=0,
    layer_types=["full_attention"],
    hc_count=4,
    hc_lowrank=8,
    ple_layer_ids=[],
    ple_embed_dim=16,
    ple_conv_kernel_size=3,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=97,
    make_ngram_vocab_size_divisible_by=8,
    output_gate_type="none",
    rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.25},
)

_QSA = dict(
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=8,
    indexer_budget=2048,
    indexer_compress_ratio=4,
)


def _text_config(**overrides):
    values = dict(_BASE)
    values.update(overrides)
    return Qwen3_8FlashNextTextConfig(**values)


def _text_kwargs(**overrides):
    values = dict(_BASE)
    values.update(overrides)
    return values


class TestTextConfigValidation:
    """Construction-time validation rejects unsupported configurations."""

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"hc_count": 1}, "hc_count > 1"),
            ({"hc_lowrank": 0}, "hc_lowrank must be positive"),
            ({"ngram_size": 1}, "ngram_size must be >= 2"),
            ({"heads_per_ngram": 0}, "heads_per_ngram must be positive"),
            ({"ple_embed_dim": 0}, "ple_embed_dim must be positive"),
            ({"ple_embed_dim": 6}, "divisible by total ngram heads"),
            ({"ple_conv_kernel_size": 0}, "ple_conv_kernel_size must be positive"),
            ({"ngram_vocab_size_base": 0}, "ngram_vocab_size_base must be positive"),
            (
                {"make_ngram_vocab_size_divisible_by": 0},
                "divisible_by must be positive",
            ),
        ],
    )
    def test_rejects_invalid_ple_parameters(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            _text_config(**overrides)

    @pytest.mark.parametrize("layer_id", [0, 2, -1])
    def test_ple_layer_ids_are_one_based_and_must_exist(self, layer_id):
        with pytest.raises(ValueError, match="1-based"):
            _text_config(ple_layer_ids=[layer_id])

    def test_ple_layer_ids_inside_layer_count_are_accepted(self):
        config = _text_config(ple_layer_ids=[1])
        assert config.ple_layer_ids == [1]


class TestTextConfigProperties:
    """Derived shapes follow the documented outer/inner HC layout."""

    def test_attention_block_types_map_full_attention(self):
        config = _text_config(
            num_hidden_layers=2,
            layer_types=["full_attention", "linear_attention"],
        )
        assert config.layers_block_type == ["attention", "linear_attention"]

    def test_short_conv_ids_are_zero_based_and_sorted(self):
        config = _text_config(
            num_hidden_layers=3,
            layer_types=["linear_attention"] * 3,
            ple_layer_ids=[3, 1, 2],
        )
        assert config.short_conv_layer_ids == [0, 1, 2]

    def test_short_conv_shape_uses_hc_channels_and_dilated_state(self):
        # channels = hidden * hc_count; state = (kernel - 1) * ngram_size.
        config = _text_config(ple_layer_ids=[1])
        assert config.short_conv_state_shape == (16 * 4, (3 - 1) * 3)

    def test_no_ple_layers_means_no_short_conv_state(self):
        config = _text_config(ple_layer_ids=[])
        assert config.short_conv_layer_ids == []
        assert config.short_conv_state_shape is None
        assert config.ngram_context_len == 0

    def test_ngram_context_len_and_spec_hidden_size(self):
        config = _text_config(ple_layer_ids=[1], ngram_size=4, ple_embed_dim=24)
        assert config.ngram_context_len == 3
        assert config.spec_hidden_size == 4 * 16

    def test_spec_decode_always_returns_a_tuple(self):
        assert _text_config().spec_decode_returns_tuple is True

    def test_group_attn_metadata_follows_indexer_heads(self):
        assert _text_config().uses_per_group_attn_metadata is False
        assert _text_config(**_QSA).uses_per_group_attn_metadata is True


class TestQsaConfigValidation:
    """QSA fields are all-or-nothing and constrained to the MQA operator ABIs."""

    def test_complete_supported_qsa_block_is_accepted(self):
        config = _text_config(**_QSA)
        assert config.indexer_n_heads == 2
        assert config.indexer_budget // config.indexer_compress_ratio == 512

    def test_partial_qsa_block_is_rejected(self):
        with pytest.raises(ValueError, match="missing required fields"):
            _text_config(indexer_n_heads=2)

    def test_only_one_indexer_kv_head_is_supported(self):
        with pytest.raises(ValueError, match="indexer_kv_heads=1"):
            _text_config(**{**_QSA, "indexer_kv_heads": 2})

    def test_budget_must_divide_by_compression_ratio(self):
        with pytest.raises(ValueError, match="divisible by indexer_compress_ratio"):
            _text_config(**{**_QSA, "indexer_budget": 2049})

    def test_block_topk_must_be_512_or_2048(self):
        with pytest.raises(ValueError, match="512 or 2048"):
            _text_config(**{**_QSA, "indexer_budget": 1000})

    def test_indexer_head_dim_must_cover_rotary_dimension(self):
        # head_dim (8) * partial_rotary_factor (0.25) == 2 > indexer_head_dim 1.
        with pytest.raises(ValueError, match="cover the attention rotary"):
            _text_config(**{**_QSA, "indexer_head_dim": 1})


class TestCompositeConfig:
    """The composite config wires typed text/vision sub-configs."""

    def test_dict_text_config_becomes_a_typed_sub_config(self):
        config = Qwen3_8FlashNextConfig(text_config=dict(_BASE))
        assert isinstance(config.text_config, Qwen3_8FlashNextTextConfig)
        assert config.text_config.hidden_size == 16
        assert isinstance(config.vision_config, Qwen3_8FlashNextVisionConfig)

    def test_top_level_kwargs_build_the_default_text_config(self):
        config = Qwen3_8FlashNextConfig(**_text_kwargs())
        assert isinstance(config.text_config, Qwen3_8FlashNextTextConfig)
        assert config.text_config.hidden_size == 16
        assert config.text_config.layer_types == ["full_attention"]
        assert config.text_config.ple_layer_ids == []

    def test_typed_instances_are_preserved_by_identity(self):
        text = _text_config()
        vision = Qwen3_8FlashNextVisionConfig()
        config = Qwen3_8FlashNextConfig(text_config=text, vision_config=vision)
        assert config.text_config is text
        assert config.vision_config is vision

    def test_vision_dict_is_built_and_media_tokens_are_stored(self):
        config = Qwen3_8FlashNextConfig(
            text_config=dict(_BASE),
            vision_config={},
            image_token_id=11,
            video_token_id=12,
            vision_start_token_id=13,
            vision_end_token_id=14,
        )
        assert isinstance(config.vision_config, Qwen3_8FlashNextVisionConfig)
        assert (
            config.image_token_id,
            config.video_token_id,
            config.vision_start_token_id,
            config.vision_end_token_id,
        ) == (11, 12, 13, 14)

    def test_tie_word_embeddings_is_applied_after_super_init(self):
        config = Qwen3_8FlashNextConfig(
            text_config=dict(_BASE), tie_word_embeddings=True
        )
        assert config.tie_word_embeddings is True

    def test_rope_parameters_fall_back_to_the_text_config(self):
        config = Qwen3_8FlashNextConfig(text_config=dict(_BASE))
        # The text config normalizes the same mapping it was constructed with.
        assert config.rope_parameters["rope_theta"] == 10000.0

    def test_explicit_rope_parameters_win_over_the_text_config(self):
        explicit = {"rope_theta": 1234.0}
        config = Qwen3_8FlashNextConfig(
            text_config=dict(_BASE), rope_parameters=explicit
        )
        assert config.rope_parameters is explicit

    def test_checkpoint_aliases_are_true_aliases(self):
        assert Qwen4ExpConfig is Qwen3_8FlashNextConfig
        assert Qwen4ExpTextConfig is Qwen3_8FlashNextTextConfig
        assert Qwen4ExpVisionConfig is Qwen3_8FlashNextVisionConfig
