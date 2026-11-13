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

"""Causal-LM checkpoint mapping, expert loading and MoE bookkeeping helpers.

The causal-LM model translates QSA cache-scale names, loads fused expert
tensors through either the fused or single-expert ABI, derives mamba state
geometry and exposes EPLB bookkeeping. The name mappings are pure string
functions with exact expected outputs; the loader probes are driven with
recording callables so both the success and the legacy-ABI fallback paths are
observed. Construction-time guards raise before any distributed module is
built.
"""

import operator
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.qwen3_8_flash_next.gpu import model as qwen_model
from vllm_fl.models.qwen3_8_flash_next.gpu.legacy_model_state import (
    Qwen3_8FlashNextLegacyModelState,
)
from vllm_fl.models.qwen3_8_flash_next.gpu.model_state import (
    Qwen3_8FlashNextModelState,
)


class TestQsaScaleNameRemap:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            (
                "model.layers.2.self_attn.k_proj.k_scale",
                "model.layers.2.self_attn._k_scale",
            ),
            (
                "model.layers.2.self_attn.k_proj.output_scale",
                "model.layers.2.self_attn._k_scale",
            ),
            (
                "model.layers.2.self_attn.attn._v_scale",
                "model.layers.2.self_attn._v_scale",
            ),
            (
                "model.layers.2.self_attn.v_scale",
                "model.layers.2.self_attn._v_scale",
            ),
            (
                "layers.2.self_attn.k_scale",
                "layers.2.self_attn._k_scale",
            ),
            (
                "model.layers.3.self_attn.k_scale",
                "model.layers.3.self_attn.k_scale",
            ),
            (
                "xlayers.2.self_attn.k_scale",
                "xlayers.2.self_attn.k_scale",
            ),
            (
                "model.layers.2.self_attn.unknown_scale",
                "model.layers.2.self_attn.unknown_scale",
            ),
        ],
    )
    def test_only_qsa_layer_scale_names_are_rewritten(self, name, expected):
        got = qwen_model._remap_qsa_cache_scale_name(name, frozenset({2}))
        assert got == expected


class TestExpertWeightLoader:
    def test_return_success_and_none_are_normalized_to_true(self):
        calls = []

        def loader(param, value, name, *, shard_id, expert_id, return_success=True):
            calls.append((param, value, name, shard_id, expert_id, return_success))
            return None

        param = SimpleNamespace(weight_loader=loader)
        value = torch.zeros(1)
        assert qwen_model._call_expert_weight_loader(param, value, "p", "w1", 3) is True
        assert calls == [(param, value, "p", "w1", 3, True)]

    def test_false_result_is_propagated(self):
        param = SimpleNamespace(
            weight_loader=lambda *args, **kwargs: False,
        )
        assert (
            qwen_model._call_expert_weight_loader(param, torch.zeros(1), "p", "w1", 0)
            is False
        )

    def test_missing_loader_is_a_type_error(self):
        with pytest.raises(TypeError, match="has no weight_loader"):
            qwen_model._call_expert_weight_loader(
                object(), torch.zeros(1), "p", "w1", 0
            )

    def test_legacy_loader_without_return_success_kwarg_is_retried(self):
        calls = []

        def loader(*args, **kwargs):
            if "return_success" in kwargs:
                raise TypeError("unexpected keyword argument 'return_success'")
            calls.append(kwargs)
            return None

        param = SimpleNamespace(weight_loader=loader)
        assert (
            qwen_model._call_expert_weight_loader(param, torch.zeros(1), "p", "w2", 1)
            is True
        )
        assert calls == [{"shard_id": "w2", "expert_id": 1}]

    def test_unrelated_type_error_is_reraised(self):
        param = SimpleNamespace(
            weight_loader=lambda *args, **kwargs: (_ for _ in ()).throw(
                TypeError("boom")
            )
        )
        with pytest.raises(TypeError, match="boom"):
            qwen_model._call_expert_weight_loader(param, torch.zeros(1), "p", "w1", 0)

    def test_single_expert_routes_through_the_named_parameter(self):
        calls = []
        param = SimpleNamespace(
            weight_loader=lambda *args, **kwargs: calls.append(kwargs) or True,
        )
        params = {"layers.0.mlp.experts.routed_experts.w2_weight": param}
        mapping = [
            ("mlp.experts.routed_experts.w2_weight", "mlp.experts.down_proj", 0, "w2")
        ]
        handled, loaded = qwen_model._load_single_expert_weight(
            "layers.0.mlp.experts.down_proj", torch.zeros(1), params, mapping
        )
        assert handled is True
        assert loaded == {"layers.0.mlp.experts.routed_experts.w2_weight"}
        assert calls == [{"shard_id": "w2", "expert_id": 0, "return_success": True}]

    def test_single_expert_skips_unrelated_and_missing_parameters(self):
        mapping = [
            ("mlp.experts.routed_experts.w2_weight", "mlp.experts.down_proj", 0, "w2")
        ]
        handled, loaded = qwen_model._load_single_expert_weight(
            "layers.0.mlp.experts.gate_up_proj", torch.zeros(1), {}, mapping
        )
        assert (handled, loaded) == (False, set())
        handled, loaded = qwen_model._load_single_expert_weight(
            "layers.0.mlp.experts.down_proj", torch.zeros(1), {}, mapping
        )
        assert (handled, loaded) == (True, set())


class TestExpertMappingProbe:
    def _helper(self):
        regular = [("p.w1_", "p.gate_proj", 0, "w1"), ("p.w3_", "p.up_proj", 0, "w3")]

        def helper(
            model,
            *,
            ckpt_gate_proj_name,
            ckpt_down_proj_name,
            ckpt_up_proj_name,
            num_experts,
            num_redundant_experts=None,
        ):
            if ckpt_gate_proj_name == "gate_proj":
                return regular
            return [
                ("p.w13_", "p.0.gate_up_proj", 0, "w1"),
                ("p.w2_", "p.0.down_proj", 0, "w2"),
            ]

        return regular, helper

    def test_fused_mapping_uses_parent_and_projection_names(self, monkeypatch):
        regular, helper = self._helper()
        monkeypatch.setattr(qwen_model, "fused_moe_make_expert_params_mapping", helper)
        mappings = qwen_model._get_expert_mappings(object(), 8, 1)
        assert mappings is not None
        assert mappings[0] == regular
        assert mappings[1] == [
            ("p.w13_weight", "p.gate_up_proj", 0, "w1"),
            ("p.w2_weight", "p.down_proj", 0, "w2"),
        ]

    def test_non_callable_helper_disables_the_legacy_path(self, monkeypatch):
        monkeypatch.setattr(qwen_model, "fused_moe_make_expert_params_mapping", None)
        assert qwen_model._get_expert_mappings(object(), 8, 0) is None

    def test_helper_type_errors_disable_the_legacy_path(self, monkeypatch):
        def helper(*args, **kwargs):
            raise TypeError("new ABI")

        monkeypatch.setattr(qwen_model, "fused_moe_make_expert_params_mapping", helper)
        assert qwen_model._get_expert_mappings(object(), 8, 0) is None

    def test_unexpected_checkpoint_name_shape_disables_the_legacy_path(
        self, monkeypatch
    ):
        def helper(*args, **kwargs):
            if kwargs.get("ckpt_gate_proj_name") == "gate_proj":
                return [("p.w1_", "p.gate_proj", 0, "w1")]
            return [("p.w13_", "a.b", 0, "w1")]

        monkeypatch.setattr(qwen_model, "fused_moe_make_expert_params_mapping", helper)
        assert qwen_model._get_expert_mappings(object(), 8, 0) is None

    def test_empty_mappings_disable_the_legacy_path(self, monkeypatch):
        monkeypatch.setattr(
            qwen_model, "fused_moe_make_expert_params_mapping", lambda *a, **k: []
        )
        assert qwen_model._get_expert_mappings(object(), 8, 0) is None


class TestMapperProbe:
    def test_native_qwen35_mapper_wins(self, monkeypatch):
        marker = object()
        monkeypatch.setattr(
            qwen_model,
            "Qwen3_5Model",
            SimpleNamespace(hf_to_vllm_mapper=marker),
        )
        assert qwen_model._get_qwen35_hf_to_vllm_mapper() is marker

    def test_qwen3_next_mapper_is_the_fallback(self, monkeypatch):
        marker = object()
        monkeypatch.setattr(
            qwen_model, "Qwen3_5Model", SimpleNamespace(hf_to_vllm_mapper=None)
        )
        monkeypatch.setattr(
            qwen_model,
            "Qwen3NextModel",
            SimpleNamespace(hf_to_vllm_mapper=marker),
        )
        assert qwen_model._get_qwen35_hf_to_vllm_mapper() is marker

    def test_default_mapper_is_created_when_neither_exists(self, monkeypatch):
        monkeypatch.setattr(
            qwen_model, "Qwen3_5Model", SimpleNamespace(hf_to_vllm_mapper=None)
        )
        monkeypatch.setattr(
            qwen_model, "Qwen3NextModel", SimpleNamespace(hf_to_vllm_mapper=None)
        )
        assert isinstance(
            qwen_model._get_qwen35_hf_to_vllm_mapper(), qwen_model.WeightsMapper
        )


class TestModelClassHooks:
    def test_mrope_positions_are_identical_on_all_three_axes(self):
        instance = object.__new__(qwen_model.Qwen3_8FlashNextForCausalLM)
        positions, delta = instance.get_mrope_input_positions([0, 1, 2], [])
        assert delta == 0
        assert positions.tolist() == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        assert positions.dtype == torch.long

    def test_model_state_classes_are_the_configured_owners(self):
        causal = qwen_model.Qwen3_8FlashNextForCausalLM
        assert causal.get_model_state_cls() is Qwen3_8FlashNextModelState
        assert causal.get_legacy_model_state_cls() is Qwen3_8FlashNextLegacyModelState
        conditional = qwen_model.Qwen3_8FlashNextForConditionalGeneration
        assert conditional.get_model_state_cls() is Qwen3_8FlashNextModelState
        assert (
            conditional.get_legacy_model_state_cls() is Qwen3_8FlashNextLegacyModelState
        )

    def test_mtp_hidden_buffer_is_returned_by_reference(self):
        instance = object.__new__(qwen_model.Qwen3_8FlashNextForCausalLM)
        nn.Module.__init__(instance)
        instance.model = SimpleNamespace(_mtp_hidden_buffer="buffer")
        assert instance.get_mtp_target_hidden_states() == "buffer"

    def test_conditional_generation_delegates_mtp_hidden_states(self):
        instance = object.__new__(qwen_model.Qwen3_8FlashNextForConditionalGeneration)
        nn.Module.__init__(instance)
        instance.language_model = SimpleNamespace(
            get_mtp_target_hidden_states=lambda: "nested"
        )
        assert instance.get_mtp_target_hidden_states() == "nested"


class TestParallelismGuards:
    def test_sparse_moe_rejects_sequence_parallel(self):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(use_sequence_parallel_moe=True)
        )
        with pytest.raises(NotImplementedError, match="sequence-parallel MoE"):
            qwen_model.Qwen3_8FlashNextSparseMoeBlock(vllm_config=config)

    def test_decoder_layer_rejects_sequence_parallel(self):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(use_sequence_parallel_moe=True),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(ple_layer_ids=[])
            ),
            cache_config=SimpleNamespace(),
            quant_config=None,
        )
        with pytest.raises(NotImplementedError, match="sequence-parallel MoE"):
            qwen_model.Qwen3_8FlashNextDecoderLayer(
                config, "full_attention", prefix="layers.0"
            )


def _sparse_moe():
    sparse = object.__new__(qwen_model.Qwen3_8FlashNextSparseMoeBlock)
    nn.Module.__init__(sparse)
    sparse.n_logical_experts = 8
    sparse.n_physical_experts = 8
    sparse.n_local_physical_experts = 8
    sparse.n_routed_experts = 8
    sparse.n_redundant_experts = 0
    sparse.experts = object()
    return sparse


def _decoder_with(mlp):
    layer = object.__new__(qwen_model.Qwen3_8FlashNextDecoderLayer)
    nn.Module.__init__(layer)
    layer.mlp = mlp
    return layer


class TestMoeBookkeeping:
    def test_collects_sparse_layers_and_example_counts(self):
        sparse = _sparse_moe()
        instance = object.__new__(qwen_model.Qwen3_8FlashNextMixtureOfExperts)
        instance.model = SimpleNamespace(
            layers=[_decoder_with(sparse), _decoder_with(object())]
        )
        instance._set_moe_parameters()
        assert instance.moe_mlp_layers == [sparse]
        assert instance.moe_layers == [sparse.experts]
        assert instance.num_moe_layers == 1
        assert instance.num_logical_experts == 8
        assert instance.num_physical_experts == 8
        assert instance.num_local_physical_experts == 8
        assert instance.num_routed_experts == 8
        assert instance.num_expert_groups == 1
        assert instance.num_shared_experts == 0

    def test_dense_model_zeroes_every_count(self):
        instance = object.__new__(qwen_model.Qwen3_8FlashNextMixtureOfExperts)
        instance.model = SimpleNamespace(layers=[_decoder_with(object())])
        instance._set_moe_parameters()
        assert instance.moe_layers == []
        assert instance.num_moe_layers == 0
        assert instance.num_logical_experts == 0
        assert instance.num_physical_experts == 0
        assert instance.num_routed_experts == 0
        assert instance.num_redundant_experts == 0

    def test_language_model_fallback_is_used_when_model_is_absent(self):
        sparse = _sparse_moe()
        instance = object.__new__(qwen_model.Qwen3_8FlashNextMixtureOfExperts)
        instance.language_model = SimpleNamespace(
            model=SimpleNamespace(layers=[_decoder_with(sparse)])
        )
        instance._set_moe_parameters()
        assert instance.moe_mlp_layers == [sparse]

    def test_physical_expert_update_propagates_to_every_moe(self):
        sparse = _sparse_moe()
        calls = []
        sparse.experts = SimpleNamespace(update_expert_map=lambda: calls.append(1))
        instance = object.__new__(qwen_model.Qwen3_8FlashNextMixtureOfExperts)
        instance.moe_mlp_layers = [sparse]
        instance.num_logical_experts = 8
        instance.update_physical_experts_metadata(10, 6)
        assert instance.num_physical_experts == 10
        assert instance.num_local_physical_experts == 6
        assert instance.num_redundant_experts == 2
        assert sparse.n_physical_experts == 10
        assert sparse.n_local_physical_experts == 6
        assert sparse.n_redundant_experts == 2
        assert calls == [1]


class TestMambaGeometry:
    def test_ple_state_shape_accounts_for_spec_tokens(self, monkeypatch):
        recorded = {}

        def fake_shape(*, tp_world_size, intermediate_size, conv_kernel):
            recorded.update(
                tp_world_size=tp_world_size,
                intermediate_size=intermediate_size,
                conv_kernel=conv_kernel,
            )
            return ("state",)

        monkeypatch.setattr(
            qwen_model.MambaStateShapeCalculator,
            "short_conv_state_shape",
            staticmethod(fake_shape),
        )
        config = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=2),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    ple_conv_kernel_size=3,
                    ngram_size=4,
                    hc_count=2,
                    hidden_size=8,
                )
            ),
        )
        result = qwen_model.Qwen3_8FlashNextForCausalLM.get_ple_mamba_state_shape_from_config(
            config
        )
        assert result == (("state",))
        # (3 - 1) * 4 == 8 state slots + 2 spec tokens + 1.
        assert recorded == {
            "tp_world_size": 1,
            "intermediate_size": 16,
            "conv_kernel": 11,
        }

    def test_ple_state_shape_without_speculation(self, monkeypatch):
        recorded = {}

        def fake_shape(*, tp_world_size, intermediate_size, conv_kernel):
            recorded["conv_kernel"] = conv_kernel
            return ()

        monkeypatch.setattr(
            qwen_model.MambaStateShapeCalculator,
            "short_conv_state_shape",
            staticmethod(fake_shape),
        )
        config = SimpleNamespace(
            speculative_config=None,
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    ple_conv_kernel_size=3,
                    ngram_size=4,
                    hc_count=2,
                    hidden_size=8,
                )
            ),
        )
        qwen_model.Qwen3_8FlashNextForCausalLM.get_ple_mamba_state_shape_from_config(
            config
        )
        assert recorded["conv_kernel"] == 9

    def test_ple_state_dtype_forwards_both_cache_dtypes(self, monkeypatch):
        recorded = {}

        def fake_dtype(model_dtype, mamba_cache_dtype):
            recorded.update(
                model_dtype=model_dtype, mamba_cache_dtype=mamba_cache_dtype
            )
            return ("dtype",)

        monkeypatch.setattr(
            qwen_model.MambaStateDtypeCalculator,
            "short_conv_state_dtype",
            staticmethod(fake_dtype),
        )
        config = SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            cache_config=SimpleNamespace(mamba_cache_dtype="float32"),
        )
        assert (
            qwen_model.Qwen3_8FlashNextForCausalLM.get_ple_mamba_state_dtype_from_config(
                config
            )
            == ("dtype",)
        )
        assert recorded == {
            "model_dtype": torch.bfloat16,
            "mamba_cache_dtype": "float32",
        }

    def test_gdn_shape_forwards_linear_head_geometry(self, monkeypatch):
        recorded = {}

        def fake_shape(*args):
            recorded["args"] = args
            return ((), ())

        monkeypatch.setattr(
            qwen_model.MambaStateShapeCalculator,
            "gated_delta_net_state_shape",
            staticmethod(fake_shape),
        )
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            speculative_config=SimpleNamespace(num_speculative_tokens=1),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    linear_num_key_heads=4,
                    linear_num_value_heads=8,
                    linear_key_head_dim=16,
                    linear_value_head_dim=8,
                    linear_conv_kernel_dim=4,
                )
            ),
        )
        qwen_model.Qwen3_8FlashNextForCausalLM.get_gdn_mamba_state_shape_from_config(
            config
        )
        assert recorded["args"] == (2, 4, 8, 16, 8, 4, 1)

    def test_gdn_dtype_forwards_ssm_cache_dtype(self, monkeypatch):
        recorded = {}

        def fake_dtype(model_dtype, mamba_cache_dtype, ssm_cache_dtype):
            recorded["args"] = (model_dtype, mamba_cache_dtype, ssm_cache_dtype)
            return ("a", "b")

        monkeypatch.setattr(
            qwen_model.MambaStateDtypeCalculator,
            "gated_delta_net_state_dtype",
            staticmethod(fake_dtype),
        )
        config = SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            cache_config=SimpleNamespace(
                mamba_cache_dtype="float32", mamba_ssm_cache_dtype="float32"
            ),
        )
        assert (
            qwen_model.Qwen3_8FlashNextForCausalLM.get_gdn_mamba_state_dtype_from_config(
                config
            )
            == ("a", "b")
        )
        assert recorded["args"] == (torch.bfloat16, "float32", "float32")

    def test_state_copy_dispatch_requires_every_mamba_type(self):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator,
        )
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

        causal = qwen_model.Qwen3_8FlashNextForCausalLM
        short = MambaAttentionBackendEnum.SHORT_CONV
        gdn = MambaAttentionBackendEnum.GDN_ATTN
        mapping = causal.get_mamba_state_copy_funcs({short, gdn})
        assert set(mapping) == {short, gdn}
        assert (
            mapping[short] == MambaStateCopyFuncCalculator.short_conv_state_copy_func()
        )
        assert (
            mapping[gdn]
            == MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
        )
        assert all(callable(func) for funcs in mapping.values() for func in funcs)
        with pytest.raises(AssertionError, match="missing state copy funcs"):
            causal.get_mamba_state_copy_funcs({object()})
        assert (
            causal.get_mamba_state_copy_func()
            == MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
        )


class TestPackageRegistration:
    def test_lazy_package_attributes_resolve_known_names(self):
        import vllm_fl.models.qwen3_8_flash_next as package
        from vllm_fl.models.qwen3_8_flash_next.common import hyperconnection

        assert package.HyperConnectionBase is hyperconnection.HyperConnectionBase
        assert package.HyperConnectionConfig is hyperconnection.HyperConnectionConfig
        with pytest.raises(AttributeError):
            operator.attrgetter("NotAModel")(package)

    def test_common_package_lazy_attributes(self):
        import vllm_fl.models.qwen3_8_flash_next.common as common
        from vllm_fl.models.qwen3_8_flash_next.common import hyperconnection

        assert common.GatedResidualSimple is hyperconnection.GatedResidualSimple
        assert common.GroupedGemmaRMSNorm is hyperconnection.GroupedGemmaRMSNorm
        with pytest.raises(AttributeError):
            operator.attrgetter("NotAModel")(common)
