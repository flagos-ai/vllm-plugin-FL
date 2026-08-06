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

from vllm_fl.quantization.w8a8.reference import (
    dynamic_per_token_quant_int8,
    unpack_uint8b128_int32,
    w8a8_linear_reference,
)


def test_flagos_uint8b128_unpack_roundtrip():
    values = torch.tensor(
        [
            [-128, -127, -1, 0, 1, 126, 127, 42],
            [127, 0, -128, 1, -1, 64, -64, 7],
        ],
        dtype=torch.int8,
    )
    codes = (values.to(torch.int16) + 128).to(torch.uint8)
    packed = codes.contiguous().view(torch.int32)

    assert torch.equal(unpack_uint8b128_int32(packed), values)


def test_dynamic_per_token_quant_uses_independent_row_scales():
    x = torch.tensor(
        [
            [0.0, 1.0, -2.0, 0.5],
            [10.0, -5.0, 2.5, 0.0],
        ],
        dtype=torch.float32,
    )
    quantized, scales = dynamic_per_token_quant_int8(x)

    assert quantized.dtype == torch.int8
    assert scales.dtype == torch.float32
    assert scales.shape == (2, 1)
    assert torch.allclose(scales[:, 0], torch.tensor([2.0 / 127, 10.0 / 127]))
    assert quantized[0, 2] == -127
    assert quantized[1, 0] == 127
    assert scales[0] != scales[1]


def test_dynamic_per_token_quant_zero_row_is_finite():
    quantized, scales = dynamic_per_token_quant_int8(torch.zeros(2, 8))
    assert torch.count_nonzero(quantized) == 0
    assert torch.isfinite(scales).all()
    assert torch.count_nonzero(scales) == 0


def test_w8a8_reference_matches_explicit_qdq_matmul():
    x = torch.tensor(
        [[[0.0, 1.0, -2.0, 0.5], [3.0, -1.0, 0.25, 2.0]]],
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [[127, -64, 12, 0], [-32, 10, 80, -127]],
        dtype=torch.int8,
    )
    weight_scale = torch.tensor([[0.01], [0.025]], dtype=torch.float32)
    bias = torch.tensor([0.5, -0.25], dtype=torch.float32)

    x_q, x_scale = dynamic_per_token_quant_int8(x.reshape(-1, x.shape[-1]))
    expected = (
        (x_q.reshape(-1, 4).int() @ weight.int().t()).float()
        * x_scale.reshape(-1, 1)
        * weight_scale.reshape(1, -1)
        + bias
    ).reshape(1, 2, 2)
    actual = w8a8_linear_reference(x, weight, weight_scale, bias)
    assert torch.allclose(actual, expected)


def test_w8a8_reference_rejects_non_channel_scale():
    with pytest.raises(ValueError, match="one value per output channel"):
        w8a8_linear_reference(
            torch.ones(1, 4),
            torch.ones(2, 4, dtype=torch.int8),
            torch.ones(1),
        )
