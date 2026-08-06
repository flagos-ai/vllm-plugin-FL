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

from vllm_fl.quantization.w8a8.reference import (
    dynamic_per_token_quant_int8,
    w8a8_linear_reference,
)


def _vllm_dynamic_int8_quant_contract(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent transcription of vLLM dynamic_scaled_int8_quant."""
    original_shape = x.shape
    x_flat = x.reshape(-1, x.size(-1)).float()
    int8_info = torch.iinfo(torch.int8)
    absmax = x_flat.abs().amax(dim=-1, keepdim=True)
    scale = absmax / int8_info.max
    inv_scale = torch.where(
        absmax == 0,
        torch.zeros_like(absmax),
        int8_info.max / absmax,
    )
    quantized = (
        (x_flat * inv_scale).round().clamp(int8_info.min, int8_info.max).to(torch.int8)
    )
    return (
        quantized.reshape(original_shape),
        scale.reshape(original_shape[:-1] + (1,)),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 8), (7, 8)])
def test_dynamic_quant_matches_vllm_value_shape_and_dtype(dtype, shape):
    values = torch.tensor(
        [
            0.0,
            0.5,
            -0.5,
            1.0,
            -2.0,
            7.75,
            -8.0,
            0.03125,
        ],
        dtype=dtype,
    )
    x = values.repeat(shape[0]).reshape(shape)
    actual_q, actual_scale = dynamic_per_token_quant_int8(x)
    expected_q, expected_scale = _vllm_dynamic_int8_quant_contract(x)

    assert actual_q.dtype == torch.int8
    assert actual_scale.dtype == torch.float32
    assert actual_q.is_contiguous()
    assert actual_scale.is_contiguous()
    assert actual_q.shape == x.shape
    assert actual_scale.shape == (x.shape[0], 1)
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_scale, expected_scale)


def test_dynamic_quant_matches_vllm_for_noncontiguous_and_zero_rows():
    base = torch.tensor(
        [
            [0.0, 99.0, 0.0, 99.0, 0.0, 99.0, 0.0, 99.0],
            [1.0, 99.0, -2.0, 99.0, 3.0, 99.0, -4.0, 99.0],
        ],
        dtype=torch.float32,
    )
    x = base[:, ::2]
    assert not x.is_contiguous()

    actual = dynamic_per_token_quant_int8(x)
    expected = _vllm_dynamic_int8_quant_contract(x)

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[1][0, 0] == 0


def test_dynamic_quant_preserves_vllm_tiny_value_and_round_to_even_semantics():
    tiny = torch.tensor([[1e-12, -5e-13]], dtype=torch.float32)
    tiny_q, tiny_scale = dynamic_per_token_quant_int8(tiny)
    assert tiny_scale[0, 0] == pytest.approx(1e-12 / 127.0)
    assert torch.equal(tiny_q, torch.tensor([[127, -64]], dtype=torch.int8))

    ties = torch.tensor(
        [[1.0, 0.5 / 127.0, 1.5 / 127.0]],
        dtype=torch.float32,
    )
    ties_q, _ = dynamic_per_token_quant_int8(ties)
    assert torch.equal(ties_q, torch.tensor([[127, 0, 2]], dtype=torch.int8))


def test_dynamic_quant_rejects_unflattened_linear_input():
    with pytest.raises(ValueError, match="2D"):
        dynamic_per_token_quant_int8(torch.ones((2, 3, 8)))


def test_linear_reference_uses_int32_accumulation_and_flaggems_scale_order():
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]], dtype=torch.float32)
    weight = torch.tensor(
        [[127, -128, 127, -128], [-128, 127, -128, 127]],
        dtype=torch.int8,
    )
    weight_scale = torch.tensor([[0.01], [0.02]], dtype=torch.float32)
    bias = torch.tensor([0.25, -0.5], dtype=torch.float32)

    x_q, x_scale = _vllm_dynamic_int8_quant_contract(x)
    accumulator = x_q.to(torch.int32) @ weight.to(torch.int32).t()
    expected = (
        accumulator.to(torch.float32)
        * x_scale.reshape(-1, 1)
        * weight_scale.reshape(1, -1)
        + bias
    )

    actual = w8a8_linear_reference(x, weight, weight_scale, bias)
    assert torch.equal(actual, expected)
