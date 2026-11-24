# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ``vllm_fl.quantization.modelslim_w8a8``.

The core suite (``tests/unit_tests/quantization/test_modelslim_w8a8.py``)
covers the DeepSeek/GLM packed-name mapping, the static-scheme weight
creation and its apply math against reference oracles, and the row-parallel
bias handling. The baseline still misses:

* the dynamic linear scheme's ``create_weights`` registering the
  ``weight_offset`` parameter;
* the MoE method's ``create_weights`` registering both offset parameters
  with the channel-scale attrs;
* the config's prefix-group enumeration arms not exercised there:
  ``.fused_wqa_wkv`` and ``.fused_qkv_a_proj`` unpacking, the
  ``.gate_up_proj`` w1/w3 vs gate/up alternatives, the plain
  ``.down_proj``/w2 alternative, dedup across rooted prefixes, and the
  fully-missing kind (None) with the mixed/unsupported errors;
* ``get_quant_method``'s LinearBase arms (FLOAT -> unquantized, W8A8 ->
  static scheme, W8A8_DYNAMIC -> dynamic scheme) and its RoutedExperts
  arms (FLOAT -> unquantized MoE, W8A8 -> NotImplementedError,
  W8A8_DYNAMIC -> the offset MoE method);
* the static scheme's apply guards: pre-quantized int8 input with more than
  one logical matrix, and the tp_rank fallback through
  ``get_tensor_model_parallel_rank`` for a RowParallelLinear without an
  explicit tp_rank.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.parameter as vllm_parameter

pytestmark = [pytest.mark.supplementary]


@pytest.fixture(autouse=True)
def _single_rank_tensor_parallel(monkeypatch):
    monkeypatch.setattr(
        vllm_parameter, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        vllm_parameter, "get_tensor_model_parallel_world_size", lambda: 1
    )


def _config(description=None):
    from vllm_fl.quantization.modelslim_w8a8 import ModelSlimW8A8Config

    if description is None:
        description = {}
    return ModelSlimW8A8Config(description)


class TestQuantKindArms:
    """``_quant_kind`` / ``_checkpoint_prefix_groups`` branch coverage."""

    def test_missing_everywhere_returns_none(self):
        config = _config()
        assert config._quant_kind("model.layers.0.mlp") is None

    def test_fused_wqa_wkv_group(self):
        config = _config({
            "layers.0.attn.wq_a.weight": "W8A8",
            "layers.0.attn.wkv.weight": "W8A8",
        })
        assert config._quant_kind("model.layers.0.attn.fused_wqa_wkv") == "W8A8"
        # The non-rooted prefix resolves the same group.
        assert config._quant_kind("layers.0.attn.fused_wqa_wkv") == "W8A8"

    def test_fused_wqa_wkv_incomplete_raises(self):
        config = _config({"layers.0.attn.wq_a.weight": "W8A8"})
        with pytest.raises(ValueError, match="missing quantization entries"):
            config._quant_kind("model.layers.0.attn.fused_wqa_wkv")

    def test_fused_qkv_a_proj_group(self):
        config = _config({
            "layers.0.self_attn.q_a_proj.weight": "W8A8",
            "layers.0.self_attn.kv_a_proj_with_mqa.weight": "W8A8_DYNAMIC",
        })
        with pytest.raises(ValueError, match="mixes"):
            config._quant_kind("model.layers.0.self_attn.fused_qkv_a_proj")

    def test_experts_group_prefers_first_matching_set(self):
        # w1/w2/w3 and gate/up/down alternatives both describe the same
        # packed experts; only one set needs to be present.
        config = _config({
            "layers.0.ffn.experts.0.w1.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.w2.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.w3.weight": "W8A8_DYNAMIC",
        })
        assert config._quant_kind("model.layers.0.ffn.experts") == "W8A8_DYNAMIC"
        gate_up_config = _config({
            "layers.0.ffn.experts.0.gate_proj.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.up_proj.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.down_proj.weight": "W8A8_DYNAMIC",
        })
        assert (gate_up_config._quant_kind("model.layers.0.ffn.experts")
                == "W8A8_DYNAMIC")

    def test_experts_group_incomplete_raises(self):
        config = _config({
            "layers.0.ffn.experts.0.w1.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.w2.weight": "W8A8_DYNAMIC",
            # w3 missing
        })
        with pytest.raises(ValueError, match="missing quantization entries"):
            config._quant_kind("model.layers.0.ffn.experts")

    def test_gate_up_proj_group(self):
        config = _config({
            "layers.0.mlp.w1.weight": "W8A8",
            "layers.0.mlp.w3.weight": "W8A8",
        })
        assert config._quant_kind("model.layers.0.mlp.gate_up_proj") == "W8A8"
        gate_alt = _config({
            "layers.0.mlp.gate_proj.weight": "W8A8_DYNAMIC",
            "layers.0.mlp.up_proj.weight": "W8A8_DYNAMIC",
        })
        assert gate_alt._quant_kind("model.layers.0.mlp.gate_up_proj") == "W8A8_DYNAMIC"

    def test_down_proj_group(self):
        config = _config({"layers.0.mlp.down_proj.weight": "W8A8"})
        assert config._quant_kind("model.layers.0.mlp.down_proj") == "W8A8"
        w2_alt = _config({"layers.0.mlp.w2.weight": "W8A8_DYNAMIC"})
        assert w2_alt._quant_kind("model.layers.0.mlp.down_proj") == "W8A8_DYNAMIC"
        # down_proj group from down only: (down,) and (w2,) both accepted.
        assert w2_alt._checkpoint_prefix_groups("model.layers.0.mlp.down_proj") == [
            ("model.layers.0.mlp.down_proj",),
            ("model.layers.0.mlp.w2",),
            ("layers.0.mlp.down_proj",),
            ("layers.0.mlp.w2",),
        ]

    def test_unsupported_kind_raises(self):
        config = _config({"layers.0.mlp.other.weight": "INT4"})
        with pytest.raises(NotImplementedError, match="not supported"):
            config._quant_kind("model.layers.0.mlp.other")

    def test_plain_prefix_group(self):
        config = _config({"layers.0.ln.weight": "FLOAT"})
        assert config._checkpoint_prefix_groups("model.layers.0.ln") == [
            ("model.layers.0.ln",), ("layers.0.ln",)
        ]

    def test_is_static_and_dynamic_helpers(self):
        config = _config({"layers.0.mlp.down_proj.weight": "W8A8"})
        assert config._is_static_w8a8("model.layers.0.mlp.down_proj") is True
        assert config._is_dynamic_w8a8("model.layers.0.mlp.down_proj") is False


class TestConfigMetadata:
    def test_from_config_roundtrip(self):
        from vllm_fl.quantization.modelslim_w8a8 import ModelSlimW8A8Config

        config = ModelSlimW8A8Config.from_config({"a.weight": "FLOAT"})
        assert isinstance(config, ModelSlimW8A8Config)
        assert config.description == {"a.weight": "FLOAT"}

    def test_root_prefixes(self):
        from vllm_fl.quantization.modelslim_w8a8 import ModelSlimW8A8Config

        assert ModelSlimW8A8Config._root_prefixes("model.x") == ["model.x", "x"]
        assert ModelSlimW8A8Config._root_prefixes("x") == ["x", "model.x"]


class TestDynamicLinearSchemeOffset:
    def test_create_weights_registers_weight_offset(self):
        """The dynamic scheme adds weight_offset on top of the parent call.

        The parent ``CompressedTensorsW8A8Int8.create_weights`` needs a full
        vLLM linear-layer contract that varies across vLLM builds; rather
        than replicate it, call the subclass body with a stubbed parent
        whose registration is observable.
        """
        import vllm.model_executor.layers.quantization.compressed_tensors.\
            schemes as schemes_mod
        from vllm_fl.quantization.modelslim_w8a8 import (
            _ModelSlimW8A8LinearScheme,
            QuantizationStrategy,
        )

        parent_calls = []

        def stub_create_weights(self, layer, **kwargs):
            parent_calls.append(kwargs)

        original = schemes_mod.CompressedTensorsW8A8Int8.create_weights
        schemes_mod.CompressedTensorsW8A8Int8.create_weights = (
            stub_create_weights
        )
        try:
            scheme = _ModelSlimW8A8LinearScheme(
                strategy=QuantizationStrategy.CHANNEL,
                is_static_input_scheme=False,
                input_symmetric=True,
            )
            layer = torch.nn.Module()
            scheme.create_weights(
                layer,
                weight_loader=lambda *a, **k: None,
                output_partition_sizes=[8],
                input_size_per_partition=4,
                params_dtype=torch.bfloat16,
                layer_name="model.layers.0.q_proj",
            )
        finally:
            schemes_mod.CompressedTensorsW8A8Int8.create_weights = original
        assert parent_calls, "the parent create_weights must run"
        assert hasattr(layer, "weight_offset")
        assert layer.weight_offset.shape == (8, 1)
        assert layer.weight_offset.dtype == torch.float32


class TestMoeMethodOffsets:
    def _moe_config(self):
        from vllm.model_executor.layers.fused_moe.activation import (
            MoEActivation,
        )
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEConfig,
            FusedMoEParallelConfig,
            RoutingMethodType,
        )

        return FusedMoEConfig(
            num_experts=2,
            experts_per_token=2,
            hidden_dim=4,
            intermediate_size=6,
            num_local_experts=2,
            num_logical_experts=2,
            activation=MoEActivation.SILU,
            device="cpu",
            routing_method=RoutingMethodType.Renormalize,
            moe_parallel_config=FusedMoEParallelConfig(
                tp_size=1, pcp_size=1, dp_size=1, ep_size=1,
                tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0,
                sp_size=1, use_ep=False, all2all_backend="none",
                enable_eplb=False,
            ),
            in_dtype=torch.bfloat16,
        )

    def test_create_weights_registers_both_offsets(self):
        """The MoE method adds w13/w2 offsets around the parent call.

        The parent ``CompressedTensorsW8A8Int8MoEMethod.create_weights``
        needs a fully-populated method object (weight/input quant configs
        and a moe config); build the subclass instance through its real
        constructor with the config's args, stub the parent registration,
        and check the offsets plus their set attrs.
        """
        import vllm.model_executor.layers.quantization.compressed_tensors.\
            compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 as moe_mod
        from vllm_fl.quantization.modelslim_w8a8 import (
            _ModelSlimW8A8MoEMethod,
            ModelSlimW8A8Config,
        )

        config = ModelSlimW8A8Config({})
        parent_calls = []

        def stub_create_weights(self, layer, num_experts, hidden_size,
                                intermediate_size_per_partition, params_dtype,
                                **extra):
            parent_calls.append((num_experts, hidden_size,
                                 intermediate_size_per_partition))

        original = (
            moe_mod.CompressedTensorsW8A8Int8MoEMethod.create_weights
        )
        original_init = moe_mod.CompressedTensorsW8A8Int8MoEMethod.__init__
        moe_mod.CompressedTensorsW8A8Int8MoEMethod.create_weights = (
            stub_create_weights
        )

        def stub_init(self, weight_quant, input_quant, moe, layer_name=None):
            # Skip the upstream backend-oracle selection (unsupported on a
            # CPU host); the subclass under test only adds attributes around
            # the parent create_weights call.
            self.weight_quant = weight_quant
            self.input_quant = input_quant
            self.moe = moe
            self.layer_name = layer_name

        moe_mod.CompressedTensorsW8A8Int8MoEMethod.__init__ = stub_init
        try:
            method = _ModelSlimW8A8MoEMethod(
                config.weight_quant,
                config.input_quant,
                self._moe_config(),
                layer_name="model.layers.0.ffn.experts",
            )
            layer = torch.nn.Module()
            method.create_weights(
                layer, 2, 4, 6, torch.bfloat16,
                weight_loader=lambda *a, **k: None,
            )
        finally:
            moe_mod.CompressedTensorsW8A8Int8MoEMethod.create_weights = original
            moe_mod.CompressedTensorsW8A8Int8MoEMethod.__init__ = original_init
        assert parent_calls == [(2, 4, 6)]
        assert hasattr(layer, "w13_weight_offset")
        assert layer.w13_weight_offset.shape == (2, 12, 1)
        assert hasattr(layer, "w2_weight_offset")
        assert layer.w2_weight_offset.shape == (2, 4, 1)
        # set_weight_attrs recorded the channel quant method on both.
        assert layer.w13_weight_offset.quant_method == "channel"
        assert layer.w2_weight_offset.quant_method == "channel"


class TestGetQuantMethod:

    @pytest.fixture(autouse=True)
    def _routed_under_vllm_config(self):
        """RoutedExperts construction reads the current VllmConfig."""
        import contextlib

        with contextlib.ExitStack() as stack:
            from vllm.config import VllmConfig
            from vllm.config.vllm import set_current_vllm_config

            stack.enter_context(set_current_vllm_config(VllmConfig()))
            yield

    def _linear_layer(self):
        from vllm.model_executor.layers.linear import LinearBase

        class _Linear(LinearBase):
            def __init__(self):
                torch.nn.Module.__init__(self)

        return _Linear()

    def _routed_experts(self):
        """Build a real RoutedExperts instance with a minimal moe config."""
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.fused_moe.activation import (
            MoEActivation,
        )
        from vllm.model_executor.layers.fused_moe.expert_map_manager import (
            ExpertMapManager,
        )
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEConfig,
            FusedMoEParallelConfig,
            RoutingMethodType,
        )

        parallel = FusedMoEParallelConfig(
            tp_size=1, pcp_size=1, dp_size=1, ep_size=1,
            tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0,
            sp_size=1, use_ep=False, all2all_backend="none",
            enable_eplb=False,
        )
        config = FusedMoEConfig(
            num_experts=2,
            experts_per_token=2,
            hidden_dim=4,
            intermediate_size=6,
            num_local_experts=2,
            num_logical_experts=2,
            activation=MoEActivation.SILU,
            device="cpu",
            routing_method=RoutingMethodType.Renormalize,
            moe_parallel_config=parallel,
            in_dtype=torch.bfloat16,
        )
        expert_map_manager = ExpertMapManager(
            max_num_batched_tokens=8,
            top_k=2,
            global_num_experts=2,
            num_redundant_experts=0,
            num_expert_group=None,
            moe_parallel_config=parallel,
            placement_strategy="linear",
            enable_eplb=False,
        )
        return RoutedExperts(
            layer_name="model.layers.0.ffn.experts",
            params_dtype=torch.bfloat16,
            moe_config=config,
            quant_config=None,
            expert_map_manager=expert_map_manager,
        )

    def test_float_linear_gets_unquantized(self):
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        config = _config({"layers.0.mlp.weight": "FLOAT"})
        method = config.get_quant_method(self._linear_layer(), "layers.0.mlp")
        assert isinstance(method, UnquantizedLinearMethod)

    def test_missing_kind_linear_gets_unquantized(self):
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        config = _config({})
        method = config.get_quant_method(self._linear_layer(), "layers.0.mlp")
        assert isinstance(method, UnquantizedLinearMethod)

    def test_static_w8a8_linear_gets_static_scheme(self):
        from vllm.model_executor.layers.quantization.compressed_tensors.\
            compressed_tensors import CompressedTensorsLinearMethod

        from vllm_fl.quantization.modelslim_w8a8 import (
            _ModelSlimW8A8StaticLinearScheme,
        )

        config = _config({"layers.0.mlp.weight": "W8A8"})
        layer = self._linear_layer()
        method = config.get_quant_method(layer, "layers.0.mlp")
        assert isinstance(method, CompressedTensorsLinearMethod)
        assert isinstance(layer.scheme, _ModelSlimW8A8StaticLinearScheme)

    def test_dynamic_w8a8_linear_gets_dynamic_scheme(self):
        from vllm.model_executor.layers.quantization.compressed_tensors.\
            compressed_tensors import CompressedTensorsLinearMethod

        from vllm_fl.quantization.modelslim_w8a8 import (
            _ModelSlimW8A8LinearScheme,
        )

        config = _config({"layers.0.mlp.weight": "W8A8_DYNAMIC"})
        layer = self._linear_layer()
        method = config.get_quant_method(layer, "layers.0.mlp")
        assert isinstance(method, CompressedTensorsLinearMethod)
        assert isinstance(layer.scheme, _ModelSlimW8A8LinearScheme)

    def test_float_routed_experts_gets_unquantized_moe(self):
        from vllm.model_executor.layers.fused_moe import (
            UnquantizedFusedMoEMethod,
        )

        config = _config({
            "layers.0.ffn.experts.0.w1.weight": "FLOAT",
            "layers.0.ffn.experts.0.w2.weight": "FLOAT",
            "layers.0.ffn.experts.0.w3.weight": "FLOAT",
        })
        method = config.get_quant_method(
            self._routed_experts(), "layers.0.ffn.experts"
        )
        assert isinstance(method, UnquantizedFusedMoEMethod)
    def test_static_w8a8_routed_experts_raises(self):
        config = _config({
            "layers.0.ffn.experts.0.w1.weight": "W8A8",
            "layers.0.ffn.experts.0.w2.weight": "W8A8",
            "layers.0.ffn.experts.0.w3.weight": "W8A8",
        })
        with pytest.raises(NotImplementedError, match="static W8A8 routed"):
            config.get_quant_method(
                self._routed_experts(), "layers.0.ffn.experts"
            )

    def test_dynamic_w8a8_routed_experts_gets_offset_method(self):
        from vllm_fl.quantization.modelslim_w8a8 import (
            _ModelSlimW8A8MoEMethod,
        )

        config = _config({
            "layers.0.ffn.experts.0.w1.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.w2.weight": "W8A8_DYNAMIC",
            "layers.0.ffn.experts.0.w3.weight": "W8A8_DYNAMIC",
        })
        # The upstream int8 MoE oracle consults device backends unavailable
        # on a CPU host. The MoE method module holds the oracle as a direct
        # module attribute, so patch it there.
        import vllm.model_executor.layers.quantization.compressed_tensors.            compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 as moe_mod

        experts_layer = self._routed_experts()
        original_select = moe_mod.select_int8_moe_backend
        def fake_select(*args, **kwargs):
            return ("triton", object)

        moe_mod.select_int8_moe_backend = fake_select
        try:
            method = config.get_quant_method(
                experts_layer, "layers.0.ffn.experts"
            )
        finally:
            moe_mod.select_int8_moe_backend = original_select
        assert isinstance(method, _ModelSlimW8A8MoEMethod)


class TestStaticSchemeApplyEdges:
    def _loaded_layer(self, partitions=(3, 5), k=4):
        from vllm_fl.quantization.modelslim_w8a8 import (
            _ModelSlimW8A8StaticLinearScheme,
        )

        scheme = _ModelSlimW8A8StaticLinearScheme()
        layer = torch.nn.Module()
        scheme.create_weights(
            layer,
            output_partition_sizes=list(partitions),
            input_size_per_partition=k,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *a, **kw: None,
        )
        out_size = sum(partitions)
        layer.weight.data = torch.randint(
            -127, 127, (out_size, k), dtype=torch.int8
        )
        layer.input_scale.data = torch.rand(len(partitions)) + 0.25
        layer.input_offset.data = torch.randint(
            -3, 3, (len(partitions),), dtype=torch.float32
        )
        layer.deq_scale.data = torch.rand(out_size) + 0.5
        layer.quant_bias.data = torch.randint(
            -50, 50, (out_size,), dtype=torch.int32
        )
        scheme.process_weights_after_loading(layer)
        return scheme, layer

    def test_prequantized_input_with_multiple_partitions_rejected(self):
        scheme, layer = self._loaded_layer(partitions=(3, 5))
        with pytest.raises(ValueError, match="single logical matrix"):
            scheme.apply_weights(layer, torch.ones(2, 4, dtype=torch.int8), None)

    def test_row_parallel_without_tp_rank_uses_distributed_rank(self,
                                                                monkeypatch):
        from vllm_fl.quantization.modelslim_w8a8 import (
            _ModelSlimW8A8StaticLinearScheme,
        )
        import vllm.distributed as vllm_dist

        scheme, layer = self._loaded_layer(partitions=(8,))
        # Make the layer a RowParallelLinear instance without tp_rank so
        # apply_weights consults get_tensor_model_parallel_rank.
        import vllm_fl.quantization.modelslim_w8a8 as ms_module

        class _RowParallel(ms_module.RowParallelLinear):
            def __init__(self, inner):
                torch.nn.Module.__init__(self)
                self.__dict__.update(inner.__dict__)

        layer = _RowParallel(layer)
        # Patch the distributed rank used inside apply_weights (imported at
        # module scope from vllm.distributed).
        monkeypatch.setattr(
            ms_module, "get_tensor_model_parallel_rank", lambda: 0
        )

        def fake_quantize(x, scale, offset, dtype, axis, sqrt_mode):
            return (x.float() * scale.float() + offset.float()).round().clamp(
                -128, 127
            ).to(torch.int8)

        def fake_matmul(x, weight, deq_scale, *, bias, output_dtype):
            acc = x.to(torch.int32) @ weight.to(torch.int32)
            if bias is not None:
                acc = acc + bias
            return (acc.float() * deq_scale.float()).to(output_dtype)

        torch_npu = SimpleNamespace(
            npu_quantize=fake_quantize, npu_quant_matmul=fake_matmul
        )
        monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        out = scheme.apply_weights(layer, x, None)
        assert out.shape == (2, 8)
        del vllm_dist
