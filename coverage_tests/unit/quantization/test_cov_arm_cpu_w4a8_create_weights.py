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

"""Supplementary coverage for the patched ARM W4A8 ``create_weights``.

For non-packed checkpoints the wrapper must forward every argument to the
original scheme method and register nothing itself.  For packed checkpoints
it must derive the effective group size, validate partition divisibility,
build the packed int32 payload plus bf16 scale/shape parameters, and create
an MP linear kernel restricted to the packed checkpoint path.

Expected shapes are derived independently from the model dimensions:
``weight_packed`` is ``(sum(output_partition_sizes), input_partition / 8)``,
``weight_scale`` is ``(sum(output_partition_sizes),
input_partition / effective_group_size)``, and ``weight_shape`` is a
two-element int64 tensor.
"""

import pytest
import torch

from ._vllm_fakes import build_environment, injected_modules, make_scheme
from vllm_fl.quantization.arm_cpu_w4a8 import install_arm_cpu_packed_w4a8


@pytest.fixture
def raw_env():
    """Fresh fake boundary without running the adapter installer."""
    env = build_environment()
    with injected_modules(env):
        yield env


@pytest.fixture
def fake_env(raw_env):
    """Fresh fake boundary with the adapter installed against it."""
    assert install_arm_cpu_packed_w4a8() is True
    return raw_env


INPUT = 256
OUTPUT = 100
PARTITIONS = [32, 16]
PARTITION_SUM = 48


def _loader(*args, **kwargs):
    return None


class TestUnpackedCreateWeights:
    """The non-packed path defers entirely to stock vLLM behavior."""

    def test_delegates_every_argument_to_original(self, fake_env):
        scheme = make_scheme(fake_env, packed=False)
        layer = torch.nn.Module()

        scheme.create_weights(
            layer,
            OUTPUT,
            INPUT,
            PARTITIONS,
            INPUT,
            torch.bfloat16,
            _loader,
            extra_option=7,
        )

        assert fake_env.original_create_weights_calls == [
            {
                "layer": layer,
                "output_size": OUTPUT,
                "input_size": INPUT,
                "output_partition_sizes": PARTITIONS,
                "input_size_per_partition": INPUT,
                "params_dtype": torch.bfloat16,
                "weight_loader": _loader,
                "kwargs": {"extra_option": 7},
            }
        ]
        assert not hasattr(layer, "weight_packed")
        assert not hasattr(layer, "weight_scale")
        assert not hasattr(layer, "weight_shape")
        assert not hasattr(scheme, "kernel")


class TestPackedCreateWeights:
    """The packed path builds packed parameters and a packed kernel."""

    def test_group_quantized_parameter_shapes_and_kernel(self, fake_env):
        scheme = make_scheme(fake_env, packed=True, group_size=128)
        layer = torch.nn.Module()

        scheme.create_weights(
            layer,
            OUTPUT,
            INPUT,
            PARTITIONS,
            INPUT,
            torch.bfloat16,
            _loader,
        )

        assert fake_env.original_create_weights_calls == []
        assert len(fake_env.choose_calls) == 1

        weight = layer.weight_packed
        assert isinstance(weight, fake_env.PackedvLLMParameter)
        assert weight.dtype is torch.int32
        assert tuple(weight.shape) == (PARTITION_SUM, INPUT // 8)
        assert weight.input_dim == 1
        assert weight.output_dim == 0
        assert weight.packed_factor == 8
        assert weight.packed_dim == 1
        assert weight.weight_loader is _loader
        assert weight.requires_grad is False

        scale = layer.weight_scale
        assert isinstance(scale, fake_env.GroupQuantScaleParameter)
        assert scale.dtype is torch.bfloat16
        assert tuple(scale.shape) == (PARTITION_SUM, INPUT // 128)
        assert scale.input_dim == 1
        assert scale.output_dim == 0
        assert scale.weight_loader is _loader

        weight_shape = layer.weight_shape
        assert isinstance(weight_shape, fake_env.BasevLLMParameter)
        assert weight_shape.dtype is torch.int64
        assert tuple(weight_shape.shape) == (2,)
        assert weight_shape.weight_loader is _loader

        config = fake_env.choose_calls[0]
        assert config.full_weight_shape == (INPUT, OUTPUT)
        assert config.partition_weight_shape == (INPUT, PARTITION_SUM)
        assert config.weight_type is torch.int8
        assert config.act_type is torch.bfloat16
        assert config.group_size == 128
        assert config.zero_points is False
        assert config.has_g_idx is False

        assert scheme.kernel.config is config
        assert scheme.kernel.init_names == {
            "w_q_param_name": "weight_packed",
            "w_s_param_name": "weight_scale",
            "w_zp_param_name": None,
            "w_gidx_param_name": None,
        }
        assert scheme.kernel._vllm_fl_arm_packed_checkpoint is True

    def test_row_parallel_group_size_minus_one_uses_partition(self, fake_env):
        scheme = make_scheme(fake_env, packed=True, group_size=-1)
        layer = torch.nn.Module()
        partition_input = 128

        scheme.create_weights(
            layer,
            PARTITION_SUM,
            INPUT,
            PARTITIONS,
            partition_input,
            torch.bfloat16,
            _loader,
        )

        assert fake_env.choose_calls[0].group_size == partition_input
        assert tuple(layer.weight_packed.shape) == (
            PARTITION_SUM,
            partition_input // 8,
        )
        assert tuple(layer.weight_scale.shape) == (PARTITION_SUM, 1)
        assert isinstance(layer.weight_scale, fake_env.ChannelQuantScaleParameter)

    def test_non_row_parallel_group_size_minus_one_uses_full_input(self, fake_env):
        scheme = make_scheme(fake_env, packed=True, group_size=-1)
        layer = torch.nn.Module()

        scheme.create_weights(
            layer,
            PARTITION_SUM,
            INPUT,
            PARTITIONS,
            INPUT,
            torch.bfloat16,
            _loader,
        )

        assert fake_env.choose_calls[0].group_size == INPUT
        assert tuple(layer.weight_scale.shape) == (PARTITION_SUM, 1)
        assert isinstance(layer.weight_scale, fake_env.GroupQuantScaleParameter)

    def test_explicit_group_size_controls_scale_columns(self, fake_env):
        scheme = make_scheme(fake_env, packed=True, group_size=64)
        layer = torch.nn.Module()

        scheme.create_weights(
            layer,
            PARTITION_SUM,
            INPUT,
            PARTITIONS,
            INPUT,
            torch.float32,
            _loader,
        )

        assert fake_env.choose_calls[0].group_size == 64
        assert layer.weight_scale.dtype is torch.float32
        assert tuple(layer.weight_scale.shape) == (PARTITION_SUM, INPUT // 64)

    def test_partition_not_divisible_by_group_size_raises(self, fake_env):
        scheme = make_scheme(fake_env, packed=True, group_size=128)
        layer = torch.nn.Module()

        with pytest.raises(
            ValueError,
            match="input partition 130 is not divisible by W4 group size 128",
        ):
            scheme.create_weights(
                layer,
                48,
                256,
                [48],
                130,
                torch.bfloat16,
                _loader,
            )

        assert not hasattr(layer, "weight_packed")
        assert fake_env.choose_calls == []
