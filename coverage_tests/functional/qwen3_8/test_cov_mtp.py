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

"""MTP checkpoint-name mapping and construction guards.

The MTP drafter translates checkpoint paths into the standalone draft model
and raises before building distributed submodules when the runtime mode is
unsupported. Every expected string below is derived from the documented
prefix-stripping/replacement order.

This module is skipped on the current CI image: vllm 0.24.0 does not export
``model_executor.models.utils.maybe_fuse_shared_experts``, which ``mtp.py``
imports unconditionally at module import time. The skip is import-scoped so
only the MTP tests are withheld; the causal-LM helper coverage continues in
``test_cov_model_helpers.py``.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)

# Drift-isolated at the supplementary entry (coverage_tests/run.py): the
# vllm 0.24 image lacks maybe_fuse_shared_experts, so this module is only
# imported and collected once the image ships vllm >= 0.33.
from vllm_fl.models.qwen3_8_flash_next.gpu import model as qwen_model, mtp


class TestMtpLayerRemap:
    def test_only_mtp_layer_indices_are_shifted(self):
        got = mtp._remap_ignored_layers(
            ["mtp.layers.0.self_attn", "mtp.layers.12.mlp", "other.layers.0"],
            5,
        )
        assert got == [
            "mtp.layers.5.self_attn",
            "mtp.layers.17.mlp",
            "other.layers.0",
        ]

    def test_empty_input_stays_empty(self):
        assert mtp._remap_ignored_layers([], 8) == []


class TestMtpWeightRemap:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("language_model.embed_tokens.weight", "model.embed_tokens.weight"),
            (
                "model.language_model.embed_tokens.weight",
                "model.embed_tokens.weight",
            ),
            ("embed_tokens.weight", "model.embed_tokens.weight"),
            ("model.mtp.layers.0.attn.weight", "model.layers.0.attn.weight"),
            ("mtp.shared_head.head.weight", "lm_head.weight"),
            ("model.shared_head.head.weight", "lm_head.weight"),
            ("shared_head.head.weight", "lm_head.weight"),
            ("model.lm_head.weight", "lm_head.weight"),
            ("mtp.layers.1.mlp.weight", "model.layers.1.mlp.weight"),
            ("model.embed_tokens.weight", "model.embed_tokens.weight"),
            ("lm_head.weight", "lm_head.weight"),
            (
                "model.language_model.mtp.layers.0.attn.weight",
                "model.layers.0.attn.weight",
            ),
            (
                "model.language_model.shared_head.head.weight",
                "lm_head.weight",
            ),
            ("visual.blocks.0.weight", None),
            ("model.mtp.layers.0.attn.weight", "model.layers.0.attn.weight"),
        ],
    )
    def test_checkpoint_paths_map_to_the_draft_model(self, name, expected):
        assert mtp._remap_mtp_weight_name(name) == expected

    def test_draft_config_requires_a_speculative_draft_model(self):
        with pytest.raises(ValueError, match="draft_model_config must be set"):
            mtp._make_draft_vllm_config(SimpleNamespace(speculative_config=None), 0)
        with pytest.raises(ValueError, match="draft_model_config must be set"):
            mtp._make_draft_vllm_config(
                SimpleNamespace(
                    speculative_config=SimpleNamespace(draft_model_config=None)
                ),
                0,
            )


class TestMtpGuards:
    def test_all_prefix_caching_mode_is_rejected(self):
        config = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=object()),
            cache_config=SimpleNamespace(mamba_cache_mode="all"),
        )
        with pytest.raises(NotImplementedError, match="mamba-cache-mode=align"):
            mtp.Qwen3_8FlashNextMTP(vllm_config=config)

    def test_moe_detection_checks_layer_and_mlp_types(self):
        instance = object.__new__(mtp.Qwen3_8FlashNextMTP)
        nn.Module.__init__(instance)
        decoder = object.__new__(qwen_model.Qwen3_8FlashNextDecoderLayer)
        nn.Module.__init__(decoder)
        decoder.mlp = object.__new__(qwen_model.Qwen3_8FlashNextSparseMoeBlock)
        dense = object.__new__(qwen_model.Qwen3_8FlashNextDecoderLayer)
        nn.Module.__init__(dense)
        dense.mlp = object()
        instance.model = SimpleNamespace(layers=[dense, decoder])
        assert instance._has_moe_layer() is True
        instance.model = SimpleNamespace(layers=[dense])
        assert instance._has_moe_layer() is False

    def test_forward_delegates_preserving_state_inputs(self):
        instance = object.__new__(mtp.Qwen3_8FlashNextMTP)
        nn.Module.__init__(instance)
        seen = {}

        def fake_model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=0,
            query_start_loc=None,
            ngram_context=None,
        ):
            seen.update(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                spec_step_idx=spec_step_idx,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
            )
            return "draft-out"

        instance.model = fake_model
        query_start_loc = torch.tensor([0, 1])
        ngram_context = torch.zeros(1, 2)
        out = instance.forward(
            input_ids=torch.zeros(1),
            positions=torch.zeros(1),
            hidden_states=torch.zeros(1, 2),
            intermediate_tensors=None,
            inputs_embeds=None,
            spec_step_idx=1,
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        assert out == "draft-out"
        assert seen["spec_step_idx"] == 1
        assert seen["query_start_loc"] is query_start_loc
        assert seen["ngram_context"] is ngram_context

    def test_embedding_delegates_to_the_predictor(self):
        instance = object.__new__(mtp.Qwen3_8FlashNextMTP)
        nn.Module.__init__(instance)
        instance.model = SimpleNamespace(embed_input_ids=lambda ids: ("emb", ids))
        token = torch.tensor([3])
        assert instance.embed_input_ids(token) == ("emb", token)

    def test_packed_module_mapping_matches_the_backbone(self):
        assert mtp.Qwen3_8FlashNextMTP.packed_modules_mapping == {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
            "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
            "in_proj_ba": ["in_proj_b", "in_proj_a"],
        }
