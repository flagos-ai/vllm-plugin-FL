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

from types import SimpleNamespace

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
    unquantized_fused_moe_method as unquantized_moe_module,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization import moe
from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.moe import (
    AscendModelSlimW8A8DynamicMoEMethod,
)
from vllm_fl.quantization.modelslim import AscendModelSlimConfig


def _weight_loader(*args, **kwargs):
    del args, kwargs


def _moe_config(**overrides):
    values = {
        "is_act_and_mul": True,
        "has_bias": False,
        "is_lora_enabled": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FusedMoEStub(FusedMoE):
    def __init__(self, moe_config):
        torch.nn.Module.__init__(self)
        self.moe_config = moe_config


def _make_layer(method, *, num_experts=2, hidden_size=2, intermediate_size=1):
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.mlp.experts"
    layer.activation = MoEActivation.SILU
    layer.apply_router_weight_on_input = False
    layer.expert_map = None
    method.create_weights(
        layer,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        params_dtype=torch.float32,
        weight_loader=_weight_loader,
    )
    return layer


def _mock_expert_kernels(monkeypatch):
    monkeypatch.setattr(moe, "_npu_format_nz", lambda tensor: tensor)
    monkeypatch.setattr(
        moe,
        "_npu_dynamic_quant",
        lambda tensor: (
            tensor,
            torch.ones(tensor.shape[0], dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(
        moe,
        "_npu_quant_matmul",
        lambda x, weight, scale, **kwargs: x.float() @ weight.float(),
    )
    monkeypatch.setattr(
        moe,
        "_npu_moe_activation",
        lambda activation, tensor: tensor[:, :1] * tensor[:, 1:],
    )


def _load_known_weights(layer):
    layer.w13_weight.data.copy_(
        torch.tensor(
            [
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]],
            ],
            dtype=torch.int8,
        )
    )
    layer.w2_weight.data.copy_(
        torch.tensor(
            [
                [[1], [2]],
                [[3], [4]],
            ],
            dtype=torch.int8,
        )
    )
    layer.w13_weight_scale.data.fill_(1)
    layer.w2_weight_scale.data.fill_(1)
    layer.w13_weight_offset.data.zero_()
    layer.w2_weight_offset.data.zero_()


def test_config_selects_dynamic_and_float_moe_methods(monkeypatch):
    monkeypatch.setattr(
        unquantized_moe_module,
        "select_unquantized_moe_backend",
        lambda **kwargs: (None, None),
    )
    mapping = {
        "experts": [
            "experts.0.w1",
            "experts.0.w2",
            "experts.0.w3",
        ]
    }
    dynamic_config = AscendModelSlimConfig(
        {
            "model.layers.0.mlp.experts.0.w1.weight": "W8A8_DYNAMIC",
            "model.layers.0.mlp.experts.0.w2.weight": "W8A8_DYNAMIC",
            "model.layers.0.mlp.experts.0.w3.weight": "W8A8_DYNAMIC",
        }
    )
    dynamic_config.packed_modules_mapping = mapping
    layer = _FusedMoEStub(_moe_config())

    vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    with set_current_vllm_config(vllm_config):
        method = dynamic_config.get_quant_method(
            layer,
            "model.layers.0.mlp.experts",
        )

        assert isinstance(method, AscendModelSlimW8A8DynamicMoEMethod)

        float_config = AscendModelSlimConfig(
            {
                "model.layers.0.mlp.experts.0.w1.weight": "FLOAT",
                "model.layers.0.mlp.experts.0.w2.weight": "FLOAT",
                "model.layers.0.mlp.experts.0.w3.weight": "FLOAT",
            }
        )
        float_config.packed_modules_mapping = mapping
        assert isinstance(
            float_config.get_quant_method(
                layer,
                "model.layers.0.mlp.experts",
            ),
            UnquantizedFusedMoEMethod,
        )


def test_static_w8a8_moe_is_rejected():
    config = AscendModelSlimConfig(
        {"model.layers.0.mlp.experts.weight": "W8A8"}
    )
    layer = _FusedMoEStub(_moe_config())

    with pytest.raises(NotImplementedError, match="W8A8 MoE is not supported"):
        config.get_quant_method(layer, "model.layers.0.mlp.experts")


def test_dynamic_moe_weight_and_channel_loader_contract():
    method = AscendModelSlimW8A8DynamicMoEMethod(_moe_config())
    layer = _make_layer(method)

    assert layer.w13_weight.shape == (2, 2, 2)
    assert layer.w2_weight.shape == (2, 2, 1)
    assert layer.w13_weight.dtype == torch.int8
    assert layer.w2_weight.dtype == torch.int8
    assert layer.w13_weight_scale.shape == (2, 2, 1)
    assert layer.w13_weight_offset.shape == (2, 2, 1)
    assert layer.w2_weight_scale.shape == (2, 2, 1)
    assert layer.w2_weight_offset.shape == (2, 2, 1)
    channel = FusedMoeWeightScaleSupported.CHANNEL.value
    assert layer.w13_weight_scale.quant_method == channel
    assert layer.w13_weight_offset.quant_method == channel
    assert layer.w2_weight_scale.quant_method == channel
    assert layer.w2_weight_offset.quant_method == channel
    assert layer.w13_weight.weight_loader is _weight_loader


def test_vllm_weight_loader_merges_w1_w3_channel_parameters():
    method = AscendModelSlimW8A8DynamicMoEMethod(_moe_config())
    layer = _FusedMoEStub(_moe_config())
    layer.prefix = "model.layers.0.mlp.experts"
    layer.quant_config = AscendModelSlimConfig({})
    layer.quant_method = method
    layer._expert_map = None
    layer.moe_parallel_config = SimpleNamespace(tp_rank=0, tp_size=1)
    method.create_weights(
        layer,
        num_experts=2,
        hidden_size=2,
        intermediate_size_per_partition=1,
        params_dtype=torch.float32,
        weight_loader=layer.weight_loader,
    )
    layer.w13_weight_scale.data.zero_()
    layer.w13_weight_offset.data.zero_()
    layer.w2_weight_scale.data.zero_()

    layer.weight_loader(
        layer.w13_weight_scale,
        torch.tensor([[2.0]]),
        "w13_weight_scale",
        "w1",
        expert_id=0,
    )
    layer.weight_loader(
        layer.w13_weight_scale,
        torch.tensor([[3.0]]),
        "w13_weight_scale",
        "w3",
        expert_id=0,
    )
    layer.weight_loader(
        layer.w13_weight_offset,
        torch.zeros(1, 1),
        "w13_weight_offset",
        "w3",
        expert_id=0,
    )
    layer.weight_loader(
        layer.w2_weight_scale,
        torch.tensor([[5.0], [7.0]]),
        "w2_weight_scale",
        "w2",
        expert_id=0,
    )

    assert torch.equal(
        layer.w13_weight_scale[0],
        torch.tensor([[2.0], [3.0]]),
    )
    assert torch.equal(
        layer.w2_weight_scale[0],
        torch.tensor([[5.0], [7.0]]),
    )


def test_post_load_and_local_topk_accumulation(monkeypatch):
    _mock_expert_kernels(monkeypatch)
    method = AscendModelSlimW8A8DynamicMoEMethod(_moe_config())
    layer = _make_layer(method)
    _load_known_weights(layer)
    method.process_weights_after_loading(layer)

    assert layer.w13_weight.shape == (2, 2, 2)
    assert layer.w2_weight.shape == (2, 1, 2)
    assert layer.w13_weight_scale.shape == (2, 2)
    assert layer.w2_weight_scale.shape == (2, 2)

    output = method.apply(
        layer,
        torch.tensor([[2.0, 3.0], [5.0, 7.0]]),
        topk_weights=torch.tensor([[0.25, 0.75], [0.4, 0.6]]),
        topk_ids=torch.tensor([[0, 1], [0, 0]]),
        shared_experts_input=None,
    )

    assert torch.allclose(
        output,
        torch.tensor([[21.25, 29.0], [25.0, 50.0]]),
    )


def test_expert_map_skips_non_local_experts(monkeypatch):
    _mock_expert_kernels(monkeypatch)
    method = AscendModelSlimW8A8DynamicMoEMethod(_moe_config())
    layer = _make_layer(method)
    _load_known_weights(layer)
    method.process_weights_after_loading(layer)
    # global expert 0 -> local 1, expert 1 is remote, expert 2 -> local 0
    layer.expert_map = torch.tensor([1, -1, 0])

    output = method.apply(
        layer,
        torch.tensor([[2.0, 3.0]]),
        topk_weights=torch.tensor([[0.5, 0.25, 0.25]]),
        topk_ids=torch.tensor([[0, 1, 2]]),
        shared_experts_input=None,
    )

    assert torch.allclose(output, torch.tensor([[14.5, 20.0]]))


def test_nonzero_moe_weight_offset_is_rejected(monkeypatch):
    monkeypatch.setattr(moe, "_npu_format_nz", lambda tensor: tensor)
    method = AscendModelSlimW8A8DynamicMoEMethod(_moe_config())
    layer = _make_layer(method)
    layer.w13_weight_offset.data.zero_()
    layer.w2_weight_offset.data.zero_()
    layer.w2_weight_offset.data[0, 0, 0] = 1

    with pytest.raises(ValueError, match="symmetric weights only"):
        method.process_weights_after_loading(layer)
