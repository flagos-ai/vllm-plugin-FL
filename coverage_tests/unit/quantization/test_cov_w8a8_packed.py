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

"""CPU tests for the packed INT8 W8A8 scheme adapter.

The vLLM compressed-tensors scheme base and the compressed-tensors metadata
are stand-ins from ``_w8a8_fakes``.  The adapter's own weight layout
(``[out, in // 4]`` int32 plus a two-element shape buffer) and its unpacking
result are checked against values computed in the test.
"""

from __future__ import annotations

import pytest
import torch

from ._w8a8_fakes import (
    QuantizationArgs,
    QuantizationStrategy,
    build_ct_environment,
    import_under_fakes,
)

MODULE = "vllm_fl.quantization.w8a8.packed"
INT8_MODE = "vllm_fl.quantization.w8a8.int8_mode"


@pytest.fixture
def packed():
    env = build_ct_environment()
    with import_under_fakes(MODULE, env.modules, purge=(MODULE, INT8_MODE)) as module:
        yield module, env


def _layer() -> torch.nn.Module:
    return torch.nn.Module()


def _expected_unpack(packed_weight: torch.Tensor, in_features: int) -> torch.Tensor:
    shifts = torch.arange(0, 32, 8, dtype=torch.int32)
    byte = (packed_weight.unsqueeze(-1) >> shifts) & 0xFF
    flat = byte.reshape(packed_weight.shape[0], -1).to(torch.int16) - 128
    return flat[:, :in_features].to(torch.int8)


def test_scheme_constructor_forwards_channel_w8a8_settings(packed):
    """Verifies the constructor forwards channel W8A8 settings to the base scheme."""
    module, env = packed
    scheme = module.CompressedTensorsPackedW8A8Int8(layer_name="layers.0")
    assert env.base_init_calls == [
        {
            "strategy": QuantizationStrategy.CHANNEL,
            "is_static_input_scheme": False,
            "input_symmetric": True,
        }
    ]
    assert scheme.input_size_per_partition == 0
    assert scheme.layer_name == "layers.0"


def test_create_weights_rejects_input_not_divisible_by_four(packed):
    """Verifies an input size not divisible by 4 raises ValueError without side effects."""
    module, env = packed
    scheme = module.CompressedTensorsPackedW8A8Int8()
    layer = _layer()
    with pytest.raises(ValueError, match="divisible by 4"):
        scheme.create_weights(
            layer=layer,
            output_partition_sizes=[16],
            input_size_per_partition=6,
            params_dtype=torch.float16,
            weight_loader=lambda *a, **k: None,
        )
    assert env.base_create_calls == []
    assert not hasattr(layer, "weight_packed")


def test_create_weights_registers_packed_int32_and_shape_buffers(packed):
    """Verifies create_weights registers the packed int32 weight and shape buffers."""
    module, env = packed
    scheme = module.CompressedTensorsPackedW8A8Int8()
    layer = _layer()

    def loader(*args, **kwargs):
        return None

    scheme.create_weights(
        layer=layer,
        output_partition_sizes=[8, 8],
        input_size_per_partition=8,
        params_dtype=torch.float16,
        weight_loader=loader,
        extra="kept",
    )

    assert scheme.input_size_per_partition == 8
    assert len(env.base_create_calls) == 1
    base_call = env.base_create_calls[0]
    assert base_call["output_partition_sizes"] == [8, 8]
    assert base_call["input_size_per_partition"] == 8
    assert base_call["params_dtype"] == torch.float16
    assert base_call["kwargs"] == {"extra": "kept"}
    assert not hasattr(layer, "weight")

    weight_packed = layer.weight_packed
    assert weight_packed.shape == (16, 2)
    assert weight_packed.dtype == torch.int32
    assert weight_packed.weight_loader is loader
    assert weight_packed.packed_factor == 4
    assert weight_packed.packed_dim == 1

    weight_shape = layer.weight_shape
    assert weight_shape.shape == (2,)
    assert weight_shape.dtype == torch.int64


def test_process_weights_unpacks_and_replaces_parameter(packed):
    """Verifies processing unpacks the int32 payload into an int8 weight parameter."""
    module, env = packed
    scheme = module.CompressedTensorsPackedW8A8Int8()
    layer = _layer()
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=[4],
        input_size_per_partition=8,
        params_dtype=torch.float16,
        weight_loader=lambda *a, **k: None,
    )
    payload = torch.tensor(
        [
            [0x00010203, 0x7F7E7D80],
            [0x10203040, 0x50607000],
            [0x01020304, 0x05060708],
            [0x7F7F7F7F, 0x00000000],
        ],
        dtype=torch.int32,
    )
    layer.weight_packed = torch.nn.Parameter(payload, requires_grad=False)

    scheme.process_weights_after_loading(layer)

    assert not hasattr(layer, "weight_packed")
    assert isinstance(layer.weight, torch.nn.Parameter)
    assert layer.weight.dtype == torch.int8
    assert torch.equal(layer.weight, _expected_unpack(payload, 8))
    assert env.base_process_calls == [layer]


def test_install_wrapper_selects_packed_scheme_for_channel_weights(packed):
    """Verifies the install wrapper selects the packed scheme for channel weights and is idempotent."""
    module, env = packed
    assert module.install_packed_w8a8_scheme() is True
    assert module.install_packed_w8a8_scheme() is True  # idempotent

    config = env.CompressedTensorsConfig()
    weight = QuantizationArgs()
    activation = QuantizationArgs(strategy=QuantizationStrategy.TOKEN, dynamic=True)
    scheme = config._get_scheme_from_parts(
        weight, activation, format="pack-quantized", layer_name="wrapped"
    )
    assert isinstance(scheme, module.CompressedTensorsPackedW8A8Int8)
    assert scheme.layer_name == "wrapped"
    assert env.original_scheme_calls == []


def test_install_wrapper_delegates_non_packed_metadata(packed):
    """Verifies non-packed metadata delegates to the original scheme selector."""
    module, env = packed
    assert module.install_packed_w8a8_scheme() is True
    config = env.CompressedTensorsConfig()
    weight = QuantizationArgs(num_bits=4)
    activation = QuantizationArgs(strategy=QuantizationStrategy.TOKEN, dynamic=True)
    result = config._get_scheme_from_parts(
        weight,
        activation,
        output_quant="out",
        format="pack-quantized",
        layer_name="native",
    )
    assert result == "original-scheme"
    assert env.original_scheme_calls == [
        {
            "weight_quant": weight,
            "input_quant": activation,
            "output_quant": "out",
            "format": "pack-quantized",
            "layer_name": "native",
        }
    ]


def test_install_wrapper_uses_quant_format_when_format_missing(packed):
    """Verifies a missing format falls back to the config's quant format."""
    module, env = packed
    assert module.install_packed_w8a8_scheme() is True
    config = env.CompressedTensorsConfig(quant_format="pack-quantized")
    scheme = config._get_scheme_from_parts(
        QuantizationArgs(),
        QuantizationArgs(strategy=QuantizationStrategy.TOKEN, dynamic=True),
    )
    assert isinstance(scheme, module.CompressedTensorsPackedW8A8Int8)
    assert env.original_scheme_calls == []


def test_install_wrapper_rejects_group_weight_strategy(packed):
    """Verifies a group weight strategy raises ValueError requiring per-channel weights."""
    module, env = packed
    assert module.install_packed_w8a8_scheme() is True
    config = env.CompressedTensorsConfig()
    with pytest.raises(ValueError, match="per-channel weights"):
        config._get_scheme_from_parts(
            QuantizationArgs(strategy=QuantizationStrategy.GROUP),
            QuantizationArgs(strategy=QuantizationStrategy.TOKEN, dynamic=True),
            format="pack-quantized",
        )
    assert env.original_scheme_calls == []
