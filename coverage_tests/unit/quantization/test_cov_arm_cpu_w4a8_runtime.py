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

"""Supplementary coverage for the ARM W4A8 runtime hooks.

``process_weights_after_loading`` runs only for kernels tagged by the packed
``create_weights`` path, delegates to stock behavior for non-G128 and
zero-point checkpoints, validates the loaded compressed-tensors parameters,
unpacks the int32 checkpoint payload, repacks it through FlagGems, and
clears the now-redundant scale/shape metadata.

``apply_weights`` delegates to stock behavior until ``process_weights`` has
recorded the ``(n, k)`` shape, then calls the FlagGems G128 kernel with a
bfloat16 activation, converts the result back to the caller's dtype, and
adds the bias in the output dtype.

The FlagGems stand-ins are deterministic CPU functions (see
``_vllm_fakes``), so every expected tensor below is recomputed directly
from the test inputs.
"""

import pytest
import torch

from ._vllm_fakes import (
    build_environment,
    injected_modules,
    make_kernel,
    make_layer,
)
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


K = 256
N = 48


def _payload(k=K, n=N):
    return torch.arange(n * (k // 8), dtype=torch.int32).reshape(n, k // 8)


def _scales(k=K, n=N, dtype=torch.float32):
    values = torch.linspace(0.5, 1.5, n * (k // 128), dtype=dtype)
    return values.reshape(n, k // 128)


def _unpacked_expected(payload):
    nibbles = payload.repeat_interleave(8, dim=-1)
    return nibbles.to(torch.int64).remainder(7).to(torch.int8)


class TestProcessWeightsDelegation:
    """Kernels outside the packed-checkpoint path keep stock behavior."""

    def test_untagged_kernel_delegates(self, fake_env):
        layer = make_layer(fake_env, _payload(), _scales())
        kernel = make_kernel(fake_env, packed=False)

        kernel.process_weights_after_loading(layer)

        assert fake_env.original_process_calls == [layer]
        assert not hasattr(kernel, "_vllm_fl_arm_w4a8_shape")
        assert fake_env.unpack_calls == []
        assert fake_env.pack_calls == []

    def test_non_g128_group_size_delegates(self, fake_env):
        layer = make_layer(fake_env, _payload(), _scales())
        kernel = make_kernel(fake_env, group_size=64)

        kernel.process_weights_after_loading(layer)

        assert fake_env.original_process_calls == [layer]
        assert not hasattr(kernel, "_vllm_fl_arm_w4a8_shape")
        assert fake_env.pack_calls == []

    def test_zero_points_checkpoint_delegates(self, fake_env):
        layer = make_layer(fake_env, _payload(), _scales())
        kernel = make_kernel(fake_env, zero_points=True)

        kernel.process_weights_after_loading(layer)

        assert fake_env.original_process_calls == [layer]
        assert not hasattr(kernel, "_vllm_fl_arm_w4a8_shape")
        assert fake_env.pack_calls == []


class TestProcessWeightsPacked:
    """The packed path unpacks, repacks and rewrites layer parameters."""

    def test_packed_checkpoint_is_unpacked_and_repacked(self, fake_env):
        payload = _payload()
        scales = _scales()
        layer = make_layer(
            fake_env,
            payload,
            scales,
            weight_shape=torch.tensor([N, K], dtype=torch.int64),
        )
        kernel = make_kernel(fake_env)

        kernel.process_weights_after_loading(layer)

        assert fake_env.original_process_calls == []

        assert len(fake_env.unpack_calls) == 1
        unpacked, num_bits, shape = fake_env.unpack_calls[0]
        assert torch.equal(unpacked, payload.detach())
        assert num_bits == 4
        assert shape == (N, K)

        assert len(fake_env.pack_calls) == 1
        repack_input, repack_scale = fake_env.pack_calls[0]
        assert torch.equal(repack_input, _unpacked_expected(payload))
        assert repack_input.dtype is torch.int8
        assert repack_scale.dtype is torch.bfloat16
        assert repack_scale.is_contiguous()
        assert torch.equal(repack_scale, scales.to(torch.bfloat16))

        assert isinstance(layer.weight_packed, torch.nn.Parameter)
        assert layer.weight_packed.dtype is torch.uint8
        assert tuple(layer.weight_packed.shape) == (N, K // 8)
        assert layer.weight_packed.requires_grad is False
        assert torch.equal(
            layer.weight_packed.data,
            torch.full((N, K // 8), 11, dtype=torch.uint8),
        )

        assert layer.weight_scale is None
        assert layer.weight_shape is None
        assert kernel._vllm_fl_arm_w4a8_shape == (N, K)
        assert [name for _, name, _ in fake_env.replace_calls] == ["weight_packed"]

    def test_missing_weight_shape_skips_packed_validation(self, fake_env):
        layer = make_layer(fake_env, _payload(), _scales())
        kernel = make_kernel(fake_env)

        kernel.process_weights_after_loading(layer)

        assert len(fake_env.unpack_calls) == 1
        assert len(fake_env.pack_calls) == 1
        assert layer.weight_shape is None
        assert kernel._vllm_fl_arm_w4a8_shape == (N, K)

    def test_non_packed_int8_checkpoint_still_repacks(self, fake_env):
        weight = torch.remainder(torch.arange(N * K).reshape(N, K), 7).to(torch.int8)
        scales = _scales()
        original_shape = torch.tensor([N, K], dtype=torch.int64)
        layer = make_layer(fake_env, weight, scales, weight_shape=original_shape)
        kernel = make_kernel(fake_env)

        kernel.process_weights_after_loading(layer)

        assert fake_env.unpack_calls == []
        assert len(fake_env.pack_calls) == 1
        assert torch.equal(fake_env.pack_calls[0][0], weight)
        assert layer.weight_packed.dtype is torch.uint8
        # A non-packed checkpoint keeps the loaded shape tensor.
        assert layer.weight_shape is not None
        assert torch.equal(layer.weight_shape, original_shape)
        assert kernel._vllm_fl_arm_w4a8_shape == (N, K)

    def test_non_contiguous_scales_are_made_contiguous(self, fake_env):
        scales = torch.arange(N * (K // 64), dtype=torch.float32).reshape(N, K // 64)[
            :, ::2
        ]
        assert not scales.is_contiguous()
        layer = make_layer(fake_env, _payload(), scales)
        kernel = make_kernel(fake_env)

        kernel.process_weights_after_loading(layer)

        _, repack_scale = fake_env.pack_calls[0]
        assert repack_scale.is_contiguous()
        assert torch.equal(repack_scale, scales.to(torch.bfloat16))

    @pytest.mark.parametrize(
        "loaded_shape",
        [
            [N, K - 1],
            [N],
            [0, K],
            [N + 1, K],
        ],
    )
    def test_invalid_weight_shape_raises(self, fake_env, loaded_shape):
        weight_shape = torch.tensor(loaded_shape, dtype=torch.int64)
        layer = make_layer(fake_env, _payload(), _scales(), weight_shape=weight_shape)
        kernel = make_kernel(fake_env)

        with pytest.raises(RuntimeError, match="unexpected packed-Q4 weight_shape"):
            kernel.process_weights_after_loading(layer)

        assert fake_env.unpack_calls == []
        assert fake_env.pack_calls == []

    def test_non_int8_unpacked_payload_raises(self, fake_env):
        weight = torch.zeros(N, K, dtype=torch.float16)
        layer = make_layer(fake_env, weight, _scales())
        kernel = make_kernel(fake_env)

        with pytest.raises(
            RuntimeError, match="unexpected compressed-tensors G128 parameter shape"
        ):
            kernel.process_weights_after_loading(layer)

        assert fake_env.pack_calls == []

    def test_wrong_scale_shape_raises(self, fake_env):
        weight = torch.zeros(N, K, dtype=torch.int8)
        scales = torch.zeros(N, K // 128 + 1, dtype=torch.float32)
        layer = make_layer(fake_env, weight, scales)
        kernel = make_kernel(fake_env)

        with pytest.raises(
            RuntimeError, match="unexpected compressed-tensors G128 parameter shape"
        ):
            kernel.process_weights_after_loading(layer)

        assert fake_env.pack_calls == []

    def test_aliased_weight_sharing_storage_is_emptied(self, fake_env):
        payload = _payload()
        layer = make_layer(fake_env, payload, _scales(), alias=payload)
        kernel = make_kernel(fake_env)

        kernel.process_weights_after_loading(layer)

        assert [name for _, name, _ in fake_env.replace_calls] == [
            "weight_packed",
            "weight",
        ]
        assert layer.weight.dtype is torch.int32
        assert layer.weight.numel() == 0
        assert layer.weight.requires_grad is False

    def test_distinct_weight_alias_is_left_untouched(self, fake_env):
        payload = _payload()
        alias = torch.zeros(N, K // 8, dtype=torch.int32)
        layer = make_layer(fake_env, payload, _scales(), alias=alias)
        kernel = make_kernel(fake_env)

        kernel.process_weights_after_loading(layer)

        assert [name for _, name, _ in fake_env.replace_calls] == ["weight_packed"]
        assert layer.weight.numel() == N * (K // 8)
        assert torch.equal(layer.weight, alias)

    def test_weight_named_quantized_parameter_skips_alias_emptying(self, fake_env):
        payload = _payload()
        layer = make_layer(fake_env, payload, _scales(), weight_name="weight")
        kernel = make_kernel(fake_env, w_q_name="weight")

        kernel.process_weights_after_loading(layer)

        assert [name for _, name, _ in fake_env.replace_calls] == ["weight"]
        assert layer.weight.dtype is torch.uint8
        assert tuple(layer.weight.shape) == (N, K // 8)


class TestApplyWeights:
    """The patched apply path wraps the FlagGems G128 linear kernel."""

    def _layer(self, env):
        layer = torch.nn.Module()
        layer.register_parameter(
            "weight_packed",
            env.BasevLLMParameter(data=torch.zeros(N, K // 8, dtype=torch.uint8)),
        )
        return layer

    def test_unset_shape_delegates_to_original(self, fake_env):
        layer = self._layer(fake_env)
        kernel = make_kernel(fake_env, packed=False)
        activations = torch.zeros(2, K, dtype=torch.float32)
        bias = torch.ones(N, dtype=torch.float32)

        output = kernel.apply_weights(layer, activations, bias)

        assert fake_env.original_apply_calls == [(layer, activations, bias)]
        assert fake_env.linear_calls == []
        assert torch.equal(output, torch.full((2, 2), -1.25))

    def test_bfloat16_input_keeps_dtype_and_passes_kernel_args(self, fake_env):
        layer = self._layer(fake_env)
        kernel = make_kernel(fake_env)
        kernel._vllm_fl_arm_w4a8_shape = (N, K)
        activations = torch.randn(3, K, dtype=torch.bfloat16)

        output = kernel.apply_weights(layer, activations)

        expected = activations.sum(dim=-1, keepdim=True).expand(-1, N)
        assert output.dtype is torch.bfloat16
        assert torch.equal(output, expected)

        assert len(fake_env.linear_calls) == 1
        kernel_input, kernel_weight, n, k = fake_env.linear_calls[0]
        assert kernel_input is activations
        assert kernel_weight is layer.weight_packed
        assert (n, k) == (N, K)

    def test_float32_input_is_converted_round_trip(self, fake_env):
        layer = self._layer(fake_env)
        kernel = make_kernel(fake_env)
        kernel._vllm_fl_arm_w4a8_shape = (N, K)
        activations = torch.randn(2, K, dtype=torch.float32)
        converted = activations.to(torch.bfloat16)

        output = kernel.apply_weights(layer, activations)

        expected = converted.sum(dim=-1, keepdim=True).expand(-1, N).to(torch.float32)
        assert output.dtype is torch.float32
        assert torch.equal(output, expected)

        kernel_input = fake_env.linear_calls[0][0]
        assert kernel_input.dtype is torch.bfloat16
        assert torch.equal(kernel_input, converted)

    def test_float32_bias_is_cast_to_bfloat16_output(self, fake_env):
        layer = self._layer(fake_env)
        kernel = make_kernel(fake_env)
        kernel._vllm_fl_arm_w4a8_shape = (N, K)
        activations = torch.randn(2, K, dtype=torch.bfloat16)
        bias = torch.randn(N, dtype=torch.float32)

        output = kernel.apply_weights(layer, activations, bias)

        base = activations.sum(dim=-1, keepdim=True).expand(-1, N)
        expected = base + bias.to(torch.bfloat16)
        assert output.dtype is torch.bfloat16
        assert torch.equal(output, expected)

    def test_bfloat16_bias_is_cast_to_float32_output(self, fake_env):
        layer = self._layer(fake_env)
        kernel = make_kernel(fake_env)
        kernel._vllm_fl_arm_w4a8_shape = (N, K)
        activations = torch.randn(2, K, dtype=torch.float32)
        bias = torch.randn(N, dtype=torch.bfloat16)
        bias_before = bias.clone()

        output = kernel.apply_weights(layer, activations, bias)

        base = activations.to(torch.bfloat16).sum(dim=-1, keepdim=True)
        expected = base.expand(-1, N).to(torch.float32) + bias.to(torch.float32)
        assert output.dtype is torch.float32
        assert torch.equal(output, expected)
        assert torch.equal(bias, bias_before)
