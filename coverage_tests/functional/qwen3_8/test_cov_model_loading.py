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

"""Checkpoint-loading adapters: stacked names, fused experts and loaders.

The fused-expert adapter is driven with a recording parameter loader and a
3-D tensor whose shards are explicitly indexed, so the emitted
``(shard, expert)`` visits and loaded-name bookkeeping are fully determined.
The model-level tests substitute the auto-loader to observe the generated
weight stream and the legacy/native branch selection.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.gpu import model as qwen_model


@pytest.mark.parametrize(
    ("name", "expected_name", "expected_shard"),
    [
        ("x.q_proj.weight", "x.qkv_proj.weight", "q"),
        ("x.k_proj.weight", "x.qkv_proj.weight", "k"),
        ("x.v_proj.weight", "x.qkv_proj.weight", "v"),
        ("x.mlp.gate_proj.weight", "x.mlp.gate_up_proj.weight", 0),
        ("x.mlp.up_proj.weight", "x.mlp.gate_up_proj.weight", 1),
        (
            "x.shared_expert.gate_proj.weight",
            "x.shared_expert.gate_up_proj.weight",
            0,
        ),
        ("x.in_proj_qkv.weight", "x.in_proj_qkvz.weight", (0, 1, 2)),
        ("x.in_proj_z.weight", "x.in_proj_qkvz.weight", 3),
        ("x.in_proj_b.weight", "x.in_proj_ba.weight", 0),
        ("x.in_proj_a.weight", "x.in_proj_ba.weight", 1),
        ("x.other.weight", "x.other.weight", None),
    ],
)
def test_stacked_weight_mapping_attaches_shard_metadata(
    name, expected_name, expected_shard
):
    weight = torch.zeros(1)
    [(got_name, got_weight)] = list(qwen_model._map_stacked_weights([(name, weight)]))
    assert got_name == expected_name
    assert got_weight is weight
    assert getattr(got_weight, "shard_id", None) == expected_shard


class _Recorder:
    def __init__(self):
        self.calls = []

    def loader(self, param, value, name, *, shard_id, expert_id, return_success=True):
        self.calls.append((value.clone(), name, shard_id, expert_id))
        return True


class TestFusedExpertWeight:
    def _param(self, recorder):
        return SimpleNamespace(weight_loader=recorder.loader)

    def test_gate_up_tensor_is_split_into_w1_and_w3_experts(self):
        recorder = _Recorder()
        weight = torch.zeros(2, 4, 3)
        for expert in range(2):
            weight[expert] = expert + 1
        param = self._param(recorder)
        name = "mlp.experts.gate_up_proj"
        params = {"fused.w13_weight": param}
        mapping = [
            ("fused.w13_weight", "mlp.experts.gate_up_proj", 0, "w1"),
        ]
        handled, loaded = qwen_model._load_fused_expert_weight(
            name, weight, params, mapping, num_experts=2
        )
        assert handled is True
        assert loaded == {"fused.w13_weight"}
        assert [(call[1], call[2], call[3]) for call in recorder.calls] == [
            ("fused.w13_weight", "w1", 0),
            ("fused.w13_weight", "w1", 1),
            ("fused.w13_weight", "w3", 0),
            ("fused.w13_weight", "w3", 1),
        ]
        # w1 sees the first chunk, w3 the second (both [rows, cols] per expert).
        assert torch.equal(recorder.calls[0][0], weight.chunk(2, dim=-2)[0][0])
        assert torch.equal(recorder.calls[2][0], weight.chunk(2, dim=-2)[1][0])

    def test_down_projection_uses_the_mapped_shard(self):
        recorder = _Recorder()
        weight = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        param = self._param(recorder)
        name = "mlp.experts.down_proj"
        params = {"fused.w2_weight": param}
        mapping = [("fused.w2_weight", "mlp.experts.down_proj", 0, "w2")]
        handled, loaded = qwen_model._load_fused_expert_weight(
            name, weight, params, mapping, num_experts=2
        )
        assert handled is True
        assert loaded == {"fused.w2_weight"}
        assert [(call[1], call[2], call[3]) for call in recorder.calls] == [
            ("fused.w2_weight", "w2", 0),
            ("fused.w2_weight", "w2", 1),
        ]
        assert torch.equal(recorder.calls[1][0], weight[1])

    def test_unowned_parameter_is_skipped(self):
        recorder = _Recorder()
        handled, loaded = qwen_model._load_fused_expert_weight(
            "mlp.experts.down_proj",
            torch.zeros(2, 4, 3),
            {},
            [("fused.w2_weight", "mlp.experts.down_proj", 0, "w2")],
            2,
        )
        assert (handled, loaded) == (True, set())
        assert recorder.calls == []

    def test_unrelated_name_is_not_handled(self):
        handled, loaded = qwen_model._load_fused_expert_weight(
            "model.embed_tokens.weight", torch.zeros(1), {}, [], 2
        )
        assert (handled, loaded) == (False, set())

    def test_non_three_dimensional_tensor_is_rejected(self):
        with pytest.raises(ValueError, match="fused 3-D MoE tensor"):
            qwen_model._load_fused_expert_weight(
                "mlp.experts.gate_up_proj",
                torch.zeros(4, 3),
                {},
                [],
                2,
            )

    def test_mapping_without_a_matching_weight_name_is_handled(self):
        handled, loaded = qwen_model._load_fused_expert_weight(
            "mlp.experts.gate_up_proj",
            torch.zeros(1, 4, 3),
            {},
            [("fused.w2_weight", "mlp.experts.down_proj", 0, "w2")],
            1,
        )
        assert (handled, loaded) == (True, set())


class TestStackedAutoLoader:
    def _loader(self):
        return qwen_model._StackedAutoWeightsLoader(nn.Module())

    def test_packed_weight_calls_the_parameter_loader(self):
        loader = self._loader()
        calls = []
        param = SimpleNamespace(
            weight_loader=lambda *args: calls.append(args)
        )
        weight = torch.ones(1)
        weight.shard_id = "q"
        assert list(loader._load_param("p", param, [("", weight)])) == ["p"]
        assert calls == [(param, weight, "q")]

    def test_packed_weight_requires_a_loader(self):
        loader = self._loader()
        weight = torch.ones(1)
        weight.shard_id = "q"
        with pytest.raises(TypeError, match="no weight_loader"):
            list(loader._load_param("p", SimpleNamespace(), [("", weight)]))

    def test_nested_weight_is_rejected(self):
        loader = self._loader()
        weight = torch.ones(1)
        weight.shard_id = "q"
        with pytest.raises(ValueError, match="Attempted to load nested weight"):
            list(loader._load_param("p", SimpleNamespace(), [("child", weight)]))

    def test_ignored_suffix_and_aliased_names_are_skipped(self, monkeypatch):
        loader = self._loader()
        weight = torch.ones(1)
        weight.shard_id = "q"
        monkeypatch.setattr(loader, "_can_ignore_unexpected", lambda qual: True)
        assert (
            list(
                loader._load_param("p", SimpleNamespace(), [("child", weight)])
            )
            == []
        )
        monkeypatch.setattr(loader, "_can_skip", lambda qual: True)
        assert (
            list(loader._load_param("p", SimpleNamespace(), [("", weight)]))
            == []
        )

    def test_unpacked_weights_delegate_to_the_base_loader(self, monkeypatch):
        loader = self._loader()
        delegated = []

        def fake_base(self, base_prefix, param, weights):
            delegated.append((base_prefix, param, list(weights)))
            yield "base"

        monkeypatch.setattr(qwen_model.AutoWeightsLoader, "_load_param", fake_base)
        weight = torch.ones(1)
        assert list(loader._load_param("p", "param", [("", weight)])) == ["base"]
        assert delegated[0][0] == "p"
        assert delegated[0][2] == [("", weight)]


class _FakeLoader:
    instances = []

    def __init__(self, root, **kwargs):
        self.root = root
        self.kwargs = kwargs
        self.loaded = []
        _FakeLoader.instances.append(self)

    def load_weights(self, weights, mapper=None):
        self.loaded = list(weights)
        self.mapper = mapper
        return {"auto"}


class TestModelLoadWeights:
    def _model(self):
        model = object.__new__(qwen_model.Qwen3_8FlashNextModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(num_experts=2)
        model.num_redundant_experts = 1
        model._qsa_layer_ids = frozenset({2})
        model.hf_to_vllm_mapper = "mapper"
        return model

    def _install(self, monkeypatch):
        _FakeLoader.instances = []
        monkeypatch.setattr(qwen_model, "AutoWeightsLoader", _FakeLoader)
        monkeypatch.setattr(qwen_model, "_StackedAutoWeightsLoader", _FakeLoader)
        monkeypatch.setattr(
            qwen_model, "maybe_fuse_shared_experts", lambda weights, **kw: weights
        )
        packs = []
        monkeypatch.setattr(
            qwen_model,
            "pack_gated_hc_projection_weights",
            lambda model: packs.append(model),
        )
        return packs

    def test_native_mapper_path_remaps_and_loads(self, monkeypatch):
        packs = self._install(monkeypatch)
        monkeypatch.setattr(qwen_model, "_HAS_NATIVE_STACKED_WEIGHTS_MAPPER", True)
        model = self._model()
        loaded = model.load_weights(
            [
                ("layers.2.self_attn.k_scale", torch.zeros(1)),
                ("plain.weight", torch.ones(1)),
            ]
        )
        assert loaded == {"auto"}
        assert packs == [model]
        loader = _FakeLoader.instances[0]
        assert loader.kwargs["skip_substrs"] == [
            "hashstats_",
            "token_lookup",
            "hyper_connection_mixer.block_inject_weight",
        ]
        assert ".k_scale" in loader.kwargs["ignore_unexpected_suffixes"]
        assert loader.mapper == "mapper"
        assert [name for name, _ in loader.loaded] == [
            "layers.2.self_attn._k_scale",
            "plain.weight",
        ]

    def test_legacy_stacked_path_maps_names_and_forces_no_mapper(self, monkeypatch):
        packs = self._install(monkeypatch)
        monkeypatch.setattr(qwen_model, "_HAS_NATIVE_STACKED_WEIGHTS_MAPPER", False)
        monkeypatch.setattr(
            qwen_model, "_get_expert_mappings", lambda *args, **kwargs: None
        )
        model = self._model()
        weight = torch.zeros(1)
        assert model.load_weights([("layers.0.q_proj.weight", weight)]) == {"auto"}
        assert packs == [model]
        loader = _FakeLoader.instances[0]
        assert loader.mapper is None
        [(name, mapped)] = loader.loaded
        assert name == "layers.0.qkv_proj.weight"
        assert mapped.shard_id == "q"

    def test_legacy_expert_adapter_absorbs_fused_and_single_weights(
        self, monkeypatch
    ):
        packs = self._install(monkeypatch)
        monkeypatch.setattr(qwen_model, "_HAS_NATIVE_STACKED_WEIGHTS_MAPPER", False)
        monkeypatch.setattr(
            qwen_model,
            "_get_expert_mappings",
            lambda *args, **kwargs: ("regular", "fused"),
        )
        monkeypatch.setattr(
            qwen_model,
            "_load_fused_expert_weight",
            lambda name, *args: (
                (True, {"fused-name"})
                if "fused" in name
                else (False, set())
            ),
        )
        monkeypatch.setattr(
            qwen_model,
            "_load_single_expert_weight",
            lambda name, *args: (
                (True, {"single-name"})
                if "single" in name
                else (False, set())
            ),
        )
        model = self._model()
        loaded = model.load_weights(
            [
                ("fused", torch.zeros(1)),
                ("single", torch.zeros(1)),
                ("layers.0.q_proj.weight", torch.zeros(1)),
            ]
        )
        assert loaded == {"auto", "fused-name", "single-name"}
        assert packs == [model]
        loader = _FakeLoader.instances[0]
        assert loader.mapper is None
        assert [name for name, _ in loader.loaded] == ["layers.0.qkv_proj.weight"]

    def test_causal_lm_loader_skips_mtp_and_packs(self, monkeypatch):
        _FakeLoader.instances = []
        monkeypatch.setattr(qwen_model, "AutoWeightsLoader", _FakeLoader)
        packs = []
        monkeypatch.setattr(
            qwen_model,
            "pack_gated_hc_projection_weights",
            lambda model: packs.append(model),
        )
        model = object.__new__(qwen_model.Qwen3_8FlashNextForCausalLM)
        nn.Module.__init__(model)
        model.hf_to_vllm_mapper = "mapper"
        weight = torch.zeros(1)
        assert model.load_weights([("mtp.weight", weight)]) == {"auto"}
        loader = _FakeLoader.instances[0]
        assert loader.kwargs["skip_substrs"] == ["mtp."]
        assert loader.mapper == "mapper"
        assert loader.loaded == [("mtp.weight", weight)]
        assert packs == [model]
