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
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.models.utils import WeightsMapper

from vllm_fl.quantization.modelslim import (
    MODELSLIM_CONFIG_FILENAME,
    AscendModelSlimConfig,
    register_modelslim_prefix_mapper,
    resolve_linear_quant_type,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.linear import (
    AscendModelSlimLinearMethod,
)
from vllm_fl.dispatch.backends.vendor.ascend.impl.quantization.w8a8 import (
    AscendW8A8DynamicLinearScheme,
)


class _LinearStub(LinearBase):
    def __init__(self):
        torch.nn.Module.__init__(self)


def test_registers_ascend_quantization_config():
    assert get_quantization_config("ascend") is AscendModelSlimConfig


def test_loads_descriptor_from_model_directory(tmp_path):
    descriptor = {
        "model.layers.0.self_attn.q_proj.weight": "W8A8_DYNAMIC",
        "model.embed_tokens.weight": "FLOAT",
    }
    (tmp_path / MODELSLIM_CONFIG_FILENAME).write_text(
        __import__("json").dumps(descriptor),
        encoding="utf-8",
    )
    config = AscendModelSlimConfig()

    config.maybe_update_config(
        str(tmp_path),
        hf_config=SimpleNamespace(model_type="example"),
    )

    assert config.quant_description == descriptor
    assert config.model_type == "example"


def test_missing_descriptor_fails_explicitly(tmp_path):
    config = AscendModelSlimConfig()
    with pytest.raises(ValueError, match=MODELSLIM_CONFIG_FILENAME):
        config.maybe_update_config(str(tmp_path))


def test_weight_packed_alias_is_normalized():
    config = AscendModelSlimConfig(
        {"model.layers.0.proj.weight_packed": "W8A8_DYNAMIC"}
    )
    assert (
        config.quant_description["model.layers.0.proj.weight"]
        == "W8A8_DYNAMIC"
    )


def test_resolves_packed_layer_only_when_all_shards_match():
    description = {
        "model.layers.0.q_proj.weight": "W8A8_DYNAMIC",
        "model.layers.0.k_proj.weight": "W8A8_DYNAMIC",
        "model.layers.0.v_proj.weight": "W8A8_DYNAMIC",
    }
    mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    assert (
        resolve_linear_quant_type(
            description,
            "model.layers.0.qkv_proj",
            mapping,
        )
        == "W8A8_DYNAMIC"
    )

    description["model.layers.0.v_proj.weight"] = "FLOAT"
    with pytest.raises(ValueError, match="mixes ModelSlim quantization types"):
        resolve_linear_quant_type(
            description,
            "model.layers.0.qkv_proj",
            mapping,
        )


def test_missing_or_unknown_linear_contract_fails_explicitly():
    with pytest.raises(ValueError, match="has no quantization entry"):
        resolve_linear_quant_type({}, "model.layers.0.proj")

    with pytest.raises(NotImplementedError, match="W4A8"):
        resolve_linear_quant_type(
            {"model.layers.0.proj.weight": "W4A8"},
            "model.layers.0.proj",
        )


def test_selects_float_and_dynamic_linear_methods():
    float_config = AscendModelSlimConfig({"model.proj.weight": "FLOAT"})
    float_method = float_config.get_quant_method(_LinearStub(), "model.proj")
    assert isinstance(float_method, UnquantizedLinearMethod)

    dynamic_config = AscendModelSlimConfig(
        {"model.proj.weight": "W8A8_DYNAMIC"}
    )
    dynamic_method = dynamic_config.get_quant_method(
        _LinearStub(),
        "model.proj",
    )
    assert isinstance(dynamic_method, AscendModelSlimLinearMethod)
    assert isinstance(dynamic_method.scheme, AscendW8A8DynamicLinearScheme)


def test_applies_vllm_weights_mapper_without_losing_entries():
    config = AscendModelSlimConfig(
        {
            "layers.0.proj.weight": "W8A8_DYNAMIC",
            "layers.0.proj.weight_scale": "W8A8_DYNAMIC",
        }
    )
    mapper = WeightsMapper(orig_to_new_prefix={"layers.": "model.layers."})

    config.apply_vllm_mapper(mapper)

    assert config.quant_description == {
        "model.layers.0.proj.weight": "W8A8_DYNAMIC",
        "model.layers.0.proj.weight_scale": "W8A8_DYNAMIC",
    }


def test_model_prefix_adapter_is_registered_outside_generic_config():
    model_type = "unit_test_modelslim_prefix_adapter"
    register_modelslim_prefix_mapper(
        model_type,
        lambda prefix: f"model.{prefix}",
    )
    config = AscendModelSlimConfig({"model.proj.weight": "FLOAT"})
    config.model_type = model_type

    method = config.get_quant_method(_LinearStub(), "proj")

    assert isinstance(method, UnquantizedLinearMethod)
