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

import pytest
import torch
import vllm.model_executor.parameter as parameter_module

from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization import w8a8
from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.linear import (
    AscendModelSlimLinearMethod,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.w8a8 import (
    AscendW8A8DynamicLinearScheme,
    AscendW8A8StaticLinearScheme,
)


@pytest.fixture(autouse=True)
def _mock_tensor_parallel_state(monkeypatch):
    """Construct vLLM parameters without a distributed process group."""

    monkeypatch.setattr(
        parameter_module,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        parameter_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )


def _weight_loader(*args, **kwargs):
    del args, kwargs


def _make_layer(method, output_partition_sizes, params_dtype=torch.bfloat16):
    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.proj"
    layer.params_dtype = params_dtype
    method.create_weights(
        layer,
        input_size_per_partition=4,
        output_partition_sizes=output_partition_sizes,
        input_size=4,
        output_size=sum(output_partition_sizes),
        params_dtype=params_dtype,
        weight_loader=_weight_loader,
    )
    return layer


def test_static_weight_and_scale_contract_for_packed_linear():
    method = AscendModelSlimLinearMethod(AscendW8A8StaticLinearScheme())
    layer = _make_layer(method, [3, 5])

    assert layer.weight.shape == (8, 4)
    assert layer.weight.dtype == torch.int8
    assert layer.input_scale.shape == (2,)
    assert layer.input_offset.shape == (2,)
    assert layer.quant_bias.shape == (8,)
    assert layer.deq_scale.shape == (8,)
    assert layer.deq_scale.dtype == torch.float32
    assert layer.weight_scale.shape == (8, 1)
    assert layer.weight_offset.shape == (8, 1)

    layer.input_scale.load_merged_column_weight(
        torch.tensor([0.5], dtype=torch.bfloat16),
        shard_id=0,
    )
    layer.input_scale.load_merged_column_weight(
        torch.tensor([0.5], dtype=torch.bfloat16),
        shard_id=1,
    )
    assert torch.equal(
        layer.input_scale,
        torch.tensor([0.5, 0.5], dtype=torch.bfloat16),
    )


def test_dynamic_weight_and_scale_contract():
    method = AscendModelSlimLinearMethod(AscendW8A8DynamicLinearScheme())
    layer = _make_layer(method, [8])

    assert layer.weight.shape == (8, 4)
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale.shape == (8, 1)
    assert layer.weight_offset.shape == (8, 1)
    assert not hasattr(layer, "input_scale")


def test_static_post_load_and_apply(monkeypatch):
    method = AscendModelSlimLinearMethod(AscendW8A8StaticLinearScheme())
    layer = _make_layer(method, [3, 5])
    layer.weight.data.zero_()
    layer.input_scale.data.copy_(torch.tensor([0.5, 0.5]))
    layer.input_offset.data.zero_()
    layer.quant_bias.data.zero_()
    layer.deq_scale.data.fill_(0.25)
    layer.weight_scale.data.fill_(0.5)
    layer.weight_offset.data.zero_()
    monkeypatch.setattr(w8a8, "_npu_format_nz", lambda tensor: tensor)

    method.process_weights_after_loading(layer)

    assert layer.weight.shape == (4, 8)
    assert layer.weight_scale.shape == (8,)
    assert layer.weight_offset.shape == (8,)
    assert layer.aclnn_input_scale.shape == (4,)
    assert torch.equal(
        layer.aclnn_input_scale_reciprocal,
        torch.full((4,), 2.0, dtype=torch.bfloat16),
    )

    calls = {}

    def fake_static_quant(x, scale_reciprocal, offset):
        calls["quant"] = (x, scale_reciprocal, offset)
        return torch.zeros_like(x, dtype=torch.int8)

    def fake_quant_matmul(x, weight, scale, **kwargs):
        calls["matmul"] = (x, weight, scale, kwargs)
        return torch.zeros(x.shape[0], weight.shape[1], dtype=torch.bfloat16)

    monkeypatch.setattr(w8a8, "_npu_static_quant", fake_static_quant)
    monkeypatch.setattr(w8a8, "_npu_quant_matmul", fake_quant_matmul)

    output = method.apply(layer, torch.ones(2, 4, dtype=torch.bfloat16))

    assert output.shape == (2, 8)
    assert calls["matmul"][3]["bias"] is layer.quant_bias
    assert calls["matmul"][3]["output_dtype"] == torch.bfloat16


def test_static_packed_input_quantization_must_match(monkeypatch):
    method = AscendModelSlimLinearMethod(AscendW8A8StaticLinearScheme())
    layer = _make_layer(method, [3, 5])
    layer.input_scale.data.copy_(torch.tensor([0.5, 0.75]))
    layer.input_offset.data.zero_()
    layer.weight_offset.data.zero_()
    monkeypatch.setattr(w8a8, "_npu_format_nz", lambda tensor: tensor)

    with pytest.raises(ValueError, match="different static input scales"):
        method.process_weights_after_loading(layer)


def test_dynamic_post_load_and_rank_preserving_apply(monkeypatch):
    method = AscendModelSlimLinearMethod(AscendW8A8DynamicLinearScheme())
    layer = _make_layer(method, [8])
    layer.weight.data.zero_()
    layer.weight_scale.data.fill_(0.5)
    layer.weight_offset.data.zero_()
    monkeypatch.setattr(w8a8, "_npu_format_nz", lambda tensor: tensor)

    method.process_weights_after_loading(layer)

    assert layer.weight.shape == (4, 8)
    assert layer.weight_scale.shape == (8,)
    assert layer.weight_scale_fp32.dtype == torch.float32

    calls = {}

    def fake_dynamic_quant(x):
        calls["dynamic_input"] = x
        return torch.zeros_like(x, dtype=torch.int8), torch.ones(
            x.shape[0],
            dtype=torch.float32,
        )

    def fake_quant_matmul(x, weight, scale, **kwargs):
        calls["matmul"] = (x, weight, scale, kwargs)
        return torch.zeros(x.shape[0], weight.shape[1], dtype=torch.bfloat16)

    monkeypatch.setattr(w8a8, "_npu_dynamic_quant", fake_dynamic_quant)
    monkeypatch.setattr(w8a8, "_npu_quant_matmul", fake_quant_matmul)

    output = method.apply(layer, torch.ones(2, 3, 4, dtype=torch.bfloat16))

    assert calls["dynamic_input"].shape == (6, 4)
    assert calls["matmul"][3]["pertoken_scale"].shape == (6,)
    assert output.shape == (2, 3, 8)


def test_nonzero_weight_offset_is_rejected(monkeypatch):
    method = AscendModelSlimLinearMethod(AscendW8A8DynamicLinearScheme())
    layer = _make_layer(method, [8])
    layer.weight_offset.data.zero_()
    layer.weight_offset.data[0, 0] = 1
    monkeypatch.setattr(w8a8, "_npu_format_nz", lambda tensor: tensor)

    with pytest.raises(ValueError, match="symmetric weights only"):
        method.process_weights_after_loading(layer)
