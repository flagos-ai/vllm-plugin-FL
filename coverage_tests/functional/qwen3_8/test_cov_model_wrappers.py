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

"""Causal-LM and conditional-generation wrappers around the Qwen4 backbone.

These constructors only wire upstream layers together, so each upstream owner
is replaced with a recording double and the expected wiring is asserted
directly.  Multimodal embedding merges use small explicit tensors so the
deepstack reshape/merge ordering is checked numerically.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.gpu import model as qwen_model


def _text_config(**overrides):
    values = dict(
        hidden_size=8,
        vocab_size=32,
        num_hidden_layers=1,
        num_experts=0,
        rms_norm_eps=1e-6,
        hc_count=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _causal_vllm_config(config, *, mamba_mode="align"):
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config, dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_mode),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )


class _FakeBackbone:
    instances = []

    def __init__(self, *, vllm_config, prefix=""):
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.make_empty_intermediate_tensors = "factory"
        self.forward_calls = []
        self.embed_calls = []
        self._mtp_hidden_buffer = "buffer"
        _FakeBackbone.instances.append(self)

    def __call__(self, *args, **kwargs):
        self.forward_calls.append((args, kwargs))
        return "backbone-output"

    def embed_input_ids(self, input_ids):
        self.embed_calls.append(input_ids)
        return ("embeds", input_ids)

    def get_mtp_target_hidden_states(self):
        return "mtp-hidden"


class _FakeHead:
    instances = []

    def __init__(self, vocab_size, hidden_size, prefix=""):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.prefix = prefix
        _FakeHead.instances.append(self)


class _FakeLogitsProcessor:
    instances = []

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.calls = []
        _FakeLogitsProcessor.instances.append(self)

    def __call__(self, *args):
        self.calls.append(args)
        return "logits"


def _install_causal_doubles(monkeypatch, *, moe_calls=None):
    _FakeBackbone.instances = []
    _FakeHead.instances = []
    _FakeLogitsProcessor.instances = []

    class FakeMoeBase:
        @staticmethod
        def set_moe_parameters(instance):
            if moe_calls is not None:
                moe_calls.append(instance)

    monkeypatch.setattr(qwen_model, "with_platform_quantization", lambda c: c)
    monkeypatch.setattr(qwen_model, "Qwen3_8FlashNextModel", _FakeBackbone)
    monkeypatch.setattr(qwen_model, "ParallelLMHead", _FakeHead)
    monkeypatch.setattr(qwen_model, "LogitsProcessor", _FakeLogitsProcessor)
    monkeypatch.setattr(qwen_model, "QwenNextMixtureOfExperts", FakeMoeBase)


class TestCausalLMConstruction:
    def test_wires_backbone_head_and_logits_processor(self, monkeypatch):
        moe_calls = []
        _install_causal_doubles(monkeypatch, moe_calls=moe_calls)
        config = _text_config(num_experts=4)
        model = qwen_model.Qwen3_8FlashNextForCausalLM(
            vllm_config=_causal_vllm_config(config), prefix="target"
        )
        assert _FakeBackbone.instances[0].prefix == "target.model"
        assert model.model is _FakeBackbone.instances[0]
        assert _FakeHead.instances[0].prefix == "target.lm_head"
        assert model.lm_head.vocab_size == 32
        assert model.logits_processor.vocab_size == 32
        assert model.make_empty_intermediate_tensors == "factory"
        assert moe_calls == [model]

    def test_dense_configuration_skips_moe_parameter_setup(self, monkeypatch):
        moe_calls = []
        _install_causal_doubles(monkeypatch, moe_calls=moe_calls)
        config = _text_config(num_experts=0)
        qwen_model.Qwen3_8FlashNextForCausalLM(vllm_config=_causal_vllm_config(config))
        assert moe_calls == []

    def test_all_mamba_cache_mode_is_rejected(self, monkeypatch):
        _install_causal_doubles(monkeypatch)
        with pytest.raises(NotImplementedError, match="mamba-cache-mode=align"):
            qwen_model.Qwen3_8FlashNextForCausalLM(
                vllm_config=_causal_vllm_config(_text_config(), mamba_mode="all")
            )

    def test_forward_delegates_every_argument(self, monkeypatch):
        _install_causal_doubles(monkeypatch)
        model = qwen_model.Qwen3_8FlashNextForCausalLM(
            vllm_config=_causal_vllm_config(_text_config())
        )
        ids = torch.tensor([1, 2])
        positions = torch.tensor([0, 1])
        intermediate = object()
        embeds = torch.zeros(2, 8)
        out = model.forward(
            ids,
            positions,
            intermediate_tensors=intermediate,
            inputs_embeds=embeds,
            query_start_loc=torch.tensor([0, 2]),
            ngram_context=torch.zeros(1, 2),
        )
        assert out == "backbone-output"
        args, kwargs = model.model.forward_calls[0]
        assert args[0] is ids
        assert args[1] is positions
        assert args[2] is intermediate
        assert args[3] is embeds
        assert set(kwargs) == {"query_start_loc", "ngram_context"}
        assert torch.equal(kwargs["query_start_loc"], torch.tensor([0, 2]))
        assert kwargs["ngram_context"].shape == (1, 2)

    def test_embedding_and_logits_delegate(self, monkeypatch):
        _install_causal_doubles(monkeypatch)
        model = qwen_model.Qwen3_8FlashNextForCausalLM(
            vllm_config=_causal_vllm_config(_text_config())
        )
        ids = torch.tensor([4, 5])
        assert model.embed_input_ids(ids) == ("embeds", ids)
        assert model.model.embed_calls[0] is ids
        hidden = torch.zeros(2, 8)
        assert model.compute_logits(hidden) == "logits"
        assert model.logits_processor.calls[0] == (model.lm_head, hidden)

    def test_mtp_hidden_state_lookup_passes_through(self, monkeypatch):
        _install_causal_doubles(monkeypatch)
        model = qwen_model.Qwen3_8FlashNextForCausalLM(
            vllm_config=_causal_vllm_config(_text_config())
        )
        assert model.get_mtp_target_hidden_states() == "buffer"

    def test_mamba_specs_cover_gdn_and_ple_state(self, monkeypatch):
        gdn_shape = ((4, 8), (8, 8))
        ple_shape = ((16, 24),)

        monkeypatch.setattr(
            qwen_model.Qwen3_8FlashNextForCausalLM,
            "get_gdn_mamba_state_shape_from_config",
            classmethod(lambda cls, cfg: gdn_shape),
        )
        monkeypatch.setattr(
            qwen_model.Qwen3_8FlashNextForCausalLM,
            "get_gdn_mamba_state_dtype_from_config",
            classmethod(lambda cls, cfg: (torch.bfloat16, torch.float32)),
        )
        monkeypatch.setattr(
            qwen_model.Qwen3_8FlashNextForCausalLM,
            "get_ple_mamba_state_shape_from_config",
            classmethod(lambda cls, cfg: ple_shape),
        )
        monkeypatch.setattr(
            qwen_model.Qwen3_8FlashNextForCausalLM,
            "get_ple_mamba_state_dtype_from_config",
            classmethod(lambda cls, cfg: (torch.float32,)),
        )
        specs = qwen_model.Qwen3_8FlashNextForCausalLM.get_mamba_specs_from_config(
            object()
        )
        assert len(specs) == 2
        assert specs[0].shapes == gdn_shape
        assert specs[0].dtypes == (torch.bfloat16, torch.float32)
        assert specs[0].block_size == -1
        assert specs[1].shapes == ple_shape
        assert specs[1].dtypes == (torch.float32,)
        monkeypatch.setattr(
            qwen_model.Qwen3_8FlashNextForCausalLM,
            "get_gdn_mamba_state_shape_from_config",
            classmethod(lambda cls, cfg: "gdn-shape"),
        )
        monkeypatch.setattr(
            qwen_model.Qwen3_8FlashNextForCausalLM,
            "get_gdn_mamba_state_dtype_from_config",
            classmethod(lambda cls, cfg: "gdn-dtype"),
        )
        cls = qwen_model.Qwen3_8FlashNextForCausalLM
        assert cls.get_mamba_state_shape_from_config(object()) == "gdn-shape"
        assert cls.get_mamba_state_dtype_from_config(object()) == "gdn-dtype"


def _multimodal_config(**overrides):
    values = dict(
        language_model_only=True,
        mm_encoder_tp_mode="data",
        is_multimodal_pruning_enabled=lambda: True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _conditional_config(*, deepstack=(), hidden_size=8):
    return SimpleNamespace(
        vision_config=SimpleNamespace(
            deepstack_visual_indexes=list(deepstack),
            out_hidden_size=6,
            rms_norm_eps=1e-6,
        ),
        text_config=SimpleNamespace(rms_norm_eps=1e-6, hidden_size=hidden_size),
    )


def _build_conditional(
    monkeypatch,
    *,
    language_model_only=True,
    deepstack=(),
    pp_first=True,
    vision=None,
):
    created = SimpleNamespace(vision=None, language=None, tokenizer_calls=[])
    config = _conditional_config(deepstack=deepstack)
    multimodal = _multimodal_config(language_model_only=language_model_only)

    class FakeBackbone(_FakeBackbone):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self.model = SimpleNamespace(start_layer=0, layers=[])
            created.language = self

    class FakeVision(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args = args
            self.kwargs = kwargs
            created.vision = self

    monkeypatch.setattr(qwen_model, "with_platform_quantization", lambda c: c)
    monkeypatch.setattr(qwen_model, "Qwen3_8FlashNextForCausalLM", FakeBackbone)
    monkeypatch.setattr(qwen_model, "Qwen3_VisionTransformer", vision or FakeVision)
    monkeypatch.setattr(
        qwen_model, "cached_tokenizer_from_config", lambda cfg: "tokenizer"
    )
    monkeypatch.setattr(
        qwen_model.Qwen3_8FlashNextForConditionalGeneration,
        "_mark_tower_model",
        lambda self, *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        qwen_model.Qwen3_8FlashNextForConditionalGeneration,
        "_mark_language_model",
        lambda self, *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        qwen_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=pp_first, is_last_rank=True),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, multimodal_config=multimodal),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )
    return (
        qwen_model.Qwen3_8FlashNextForConditionalGeneration(
            vllm_config=vllm_config, prefix="root"
        ),
        created,
    )


class TestConditionalGenerationConstruction:
    def test_multimodal_config_is_required(self, monkeypatch):
        monkeypatch.setattr(qwen_model, "with_platform_quantization", lambda c: c)
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=_conditional_config(), multimodal_config=None
            ),
            quant_config=None,
        )
        with pytest.raises(ValueError, match="requires multimodal_config"):
            qwen_model.Qwen3_8FlashNextForConditionalGeneration(vllm_config=vllm_config)

    def test_language_model_only_keeps_the_tower_missing(self, monkeypatch):
        model, created = _build_conditional(monkeypatch, language_model_only=True)
        assert created.vision is None
        assert isinstance(model.visual, qwen_model.StageMissingLayer)
        assert model.visual.stage_name == "vision_tower"
        assert model._tokenizer is None
        assert model.use_data_parallel is False
        assert model.use_deepstack is False
        assert model.deepstack_num_level == 0
        assert model.multiscale_dim == 0
        assert model.language_model is created.language
        assert model.make_empty_intermediate_tensors == "factory"

    def test_full_multimodal_construction_builds_the_vision_tower(self, monkeypatch):
        model, created = _build_conditional(monkeypatch, language_model_only=False)
        assert created.vision is not None
        assert created.vision.kwargs["prefix"] == "root.visual"
        assert model._tokenizer == "tokenizer"
        assert model.use_data_parallel is True
        assert model.is_multimodal_pruning_enabled is False
        assert model.use_deepstack is False
        assert model.visual_dim == 6
        assert model.multiscale_dim == 0
        assert not hasattr(model, "deepstack_input_embeds")

    def test_deepstack_allocates_one_buffer_per_level(self, monkeypatch):
        model, _ = _build_conditional(
            monkeypatch, language_model_only=False, deepstack=(0, 1, 2)
        )
        assert model.use_deepstack is True
        assert model.deepstack_num_level == 3
        assert model.multiscale_dim == 18
        assert len(model.deepstack_input_embeds) == 3
        assert model.deepstack_input_embeds[0].shape == (16, 8)
        assert model.deepstack_input_embeds_num_tokens == 0

    def test_pipeline_rank_must_start_after_the_deepstack_levels(self, monkeypatch):
        with pytest.raises(AssertionError, match="start_layer"):
            _build_conditional(
                monkeypatch,
                language_model_only=False,
                deepstack=(0, 1),
                pp_first=False,
            )

    def test_embed_input_ids_short_circuits_without_multimodal_embeddings(
        self, monkeypatch
    ):
        model, _ = _build_conditional(monkeypatch)
        model._embed_text_input_ids = lambda input_ids, embed, is_multimodal=None: (
            "text-embeds"
        )
        assert model.embed_input_ids(torch.tensor([1])) == "text-embeds"
        assert model.embed_input_ids(torch.tensor([1]), []) == "text-embeds"

    def test_language_model_only_rejects_multimodal_embeddings(self, monkeypatch):
        model, _ = _build_conditional(monkeypatch, language_model_only=True)
        model._embed_text_input_ids = lambda *args, **kwargs: "text-embeds"
        with pytest.raises(ValueError, match="does not accept multimodal"):
            model.embed_input_ids(
                torch.tensor([1]),
                [torch.zeros(1, 8)],
                is_multimodal=torch.tensor([True]),
            )

    def test_deepstack_branch_sets_and_merges_the_sidebar(self, monkeypatch):
        model, _ = _build_conditional(
            monkeypatch, language_model_only=False, deepstack=(0,)
        )
        model._embed_text_input_ids = lambda *args, **kwargs: "text-embeds"
        merged = {}

        def compute_deepstack(*, inputs_embeds, multimodal_embeddings, is_multimodal):
            assert inputs_embeds == "text-embeds"
            return {"deepstack_input_embeds_0": "sidebar"}, ["adjusted"]

        def merge(*, inputs_embeds, multimodal_embeddings, is_multimodal):
            merged.update(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
            return "merged"

        model._compute_deepstack_embeds = compute_deepstack
        model._set_deepstack_input_embeds = lambda value: merged.update(deepstack=value)
        monkeypatch.setattr(
            qwen_model, "_require_is_multimodal", lambda value: "validated"
        )
        monkeypatch.setattr(qwen_model, "_merge_multimodal_embeddings", merge)
        result = model.embed_input_ids(
            torch.tensor([1]),
            [torch.zeros(1, 8)],
            is_multimodal=torch.tensor([True]),
        )
        assert result == "merged"
        assert merged["multimodal_embeddings"] == ["adjusted"]
        assert merged["is_multimodal"] == "validated"
        assert merged["deepstack"] == {"deepstack_input_embeds_0": "sidebar"}

    def test_non_deepstack_branch_merges_directly(self, monkeypatch):
        model, _ = _build_conditional(monkeypatch, language_model_only=False)
        model._embed_text_input_ids = lambda *args, **kwargs: "text-embeds"
        called = {}
        monkeypatch.setattr(
            qwen_model, "_require_is_multimodal", lambda value: "validated"
        )
        monkeypatch.setattr(
            qwen_model,
            "_merge_multimodal_embeddings",
            lambda **kwargs: called.update(kwargs) or "merged",
        )
        assert (
            model.embed_input_ids(
                torch.tensor([1]),
                [torch.zeros(1, 8)],
                is_multimodal=torch.tensor([True]),
            )
            == "merged"
        )
        called_embeds = called["multimodal_embeddings"]
        assert len(called_embeds) == 1
        assert called_embeds[0].shape == (1, 8)
        assert torch.equal(called_embeds[0], torch.zeros(1, 8))

    def test_forward_routes_state_inputs_to_the_backbone(self, monkeypatch):
        model, _ = _build_conditional(monkeypatch)
        calls = {}

        class BackboneModel:
            def __init__(self):
                self.start_layer = 0
                self.layers = []

            def __call__(self, **kwargs):
                calls.update(kwargs)
                return "hidden"

        model.language_model.model = BackboneModel()
        model._get_deepstack_input_embeds = lambda tokens: ("deepstack", tokens)
        cleared = []
        model._clear_deepstack_input_embeds = cleared.append
        monkeypatch.setattr(
            qwen_model,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        )
        embeds = torch.zeros(3, 8)
        result = model.forward(
            torch.tensor([1, 2, 3]),
            torch.zeros(3),
            inputs_embeds=embeds,
            query_start_loc=torch.tensor([0, 3]),
            ngram_context=torch.zeros(1, 2),
        )
        assert result == "hidden"
        assert calls["inputs_embeds"] is embeds
        assert calls["deepstack_input_embeds"] == ("deepstack", 3)
        assert calls["query_start_loc"].tolist() == [0, 3]
        assert cleared == [3]

    def test_forward_clears_inputs_embeds_when_intermediate_tensors_exist(
        self, monkeypatch
    ):
        model, _ = _build_conditional(monkeypatch)
        calls = {}

        class BackboneModel:
            def __init__(self):
                self.start_layer = 0
                self.layers = []

            def __call__(self, **kwargs):
                calls.update(kwargs)
                return "hidden"

        model.language_model.model = BackboneModel()
        monkeypatch.setattr(
            qwen_model,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        )
        model.forward(
            torch.tensor([1]),
            torch.zeros(1),
            intermediate_tensors="intermediate",
            inputs_embeds=torch.zeros(1, 8),
        )
        assert calls["inputs_embeds"] is None
        assert calls["deepstack_input_embeds"] is None

    def test_load_weights_skips_visual_only_for_language_model_only(self, monkeypatch):
        calls = []

        class FakeLoader:
            def __init__(self, root, **kwargs):
                calls.append(kwargs)

            def load_weights(self, weights, mapper=None):
                return {"loaded"}

        monkeypatch.setattr(qwen_model, "AutoWeightsLoader", FakeLoader)
        monkeypatch.setattr(
            qwen_model,
            "pack_gated_hc_projection_weights",
            lambda model: calls.append("packed"),
        )
        model = object.__new__(qwen_model.Qwen3_8FlashNextForConditionalGeneration)
        from torch import nn

        nn.Module.__init__(model)
        model.language_model_only = True
        model.hf_to_vllm_mapper = "mapper"
        assert model.load_weights([]) == {"loaded"}
        assert calls[0]["skip_prefixes"] == ["visual."]
        calls.clear()
        model.language_model_only = False
        assert model.load_weights([]) == {"loaded"}
        assert calls[0]["skip_prefixes"] is None
        assert calls[-1] == "packed"

    def test_mamba_classmethods_delegate_to_the_causal_model(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            qwen_model.Qwen3_8FlashNextForCausalLM,
            "get_mamba_state_dtype_from_config",
            classmethod(lambda cls, cfg: calls.append("dtype") or "dtype"),
        )
        monkeypatch.setattr(
            qwen_model.Qwen3_8FlashNextForCausalLM,
            "get_mamba_state_shape_from_config",
            classmethod(lambda cls, cfg: calls.append("shape") or "shape"),
        )
        monkeypatch.setattr(
            qwen_model.Qwen3_8FlashNextForCausalLM,
            "get_mamba_specs_from_config",
            classmethod(lambda cls, cfg: calls.append("specs") or "specs"),
        )
        monkeypatch.setattr(
            qwen_model.Qwen3_8FlashNextForCausalLM,
            "get_mamba_state_copy_func",
            classmethod(lambda cls: calls.append("copy") or "copy"),
        )
        monkeypatch.setattr(
            qwen_model.Qwen3_8FlashNextForCausalLM,
            "get_mamba_state_copy_funcs",
            classmethod(lambda cls, types: calls.append(types) or "copies"),
        )
        cls = qwen_model.Qwen3_8FlashNextForConditionalGeneration
        assert cls.get_mamba_state_dtype_from_config(object()) == "dtype"
        assert cls.get_mamba_state_shape_from_config(object()) == "shape"
        assert cls.get_mamba_specs_from_config(object()) == "specs"
        assert cls.get_mamba_state_copy_func() == "copy"
        assert cls.get_mamba_state_copy_funcs({"gdn"}) == "copies"
        assert calls == ["dtype", "shape", "specs", "copy", {"gdn"}]
