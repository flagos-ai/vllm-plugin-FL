# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from vllm.model_executor.kernels.linear import MPLinearLayerConfig
from vllm.scalar_type import scalar_types

from vllm_fl.quantization.wna16 import linear


def _config(
    weight_type,
    *,
    group_size: int,
) -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(128, 64),
        partition_weight_shape=(128, 64),
        weight_type=weight_type,
        act_type=torch.bfloat16,
        group_size=group_size,
        zero_points=False,
        has_g_idx=False,
    )


@pytest.mark.parametrize("group_size", [128, -1])
def test_w8a16_linear_accepts_group_and_channel(monkeypatch, group_size):
    monkeypatch.setattr(
        linear.kernels,
        "is_w8a16_gemm_available",
        lambda: True,
    )
    supported, reason = linear.FLWNA16LinearKernel.can_implement(
        _config(scalar_types.uint8b128, group_size=group_size)
    )
    assert supported is True
    assert reason is None


def test_w8a16_linear_does_not_depend_on_w4_operator(monkeypatch):
    monkeypatch.setattr(
        linear.kernels,
        "is_w8a16_gemm_available",
        lambda: True,
    )
    monkeypatch.setattr(
        linear.kernels,
        "is_wna16_gemm_available",
        lambda: False,
    )
    supported, _ = linear.FLWNA16LinearKernel.can_implement(
        _config(scalar_types.uint8b128, group_size=128)
    )
    assert supported


def test_w8a16_linear_dispatches_with_eight_bits(monkeypatch):
    monkeypatch.setattr(
        linear.kernels,
        "is_w8a16_gemm_available",
        lambda: True,
    )
    calls = []

    def gemm(x, weight, scale, group_size, bias, *, num_bits):
        calls.append((group_size, num_bits, bias))
        return torch.zeros((x.shape[0], weight.shape[0]), dtype=x.dtype)

    monkeypatch.setattr(linear.kernels, "wna16_gemm", gemm)
    kernel = linear.FLWNA16LinearKernel(
        _config(scalar_types.uint8b128, group_size=128),
        "weight_packed",
        "weight_scale",
    )
    layer = torch.nn.Module()
    layer.weight_packed = torch.nn.Parameter(
        torch.empty((64, 32), dtype=torch.int32),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.ones((64, 1)),
        requires_grad=False,
    )
    output = kernel.apply_weights(
        layer,
        torch.ones((2, 3, 128), dtype=torch.bfloat16),
    )

    assert output.shape == (2, 3, 64)
    assert calls == [(128, 8, None)]
