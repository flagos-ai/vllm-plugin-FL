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

"""CPU tests for the portable W8A8 reference operations.

Every expectation below is derived from a plain-torch formula written in the
test (offset-binary byte extraction by shifts, symmetric per-token scaling,
integer accumulation followed by the two dequant scales).  The reference
module imports only ``torch``, so these tests run on any host.
"""

from __future__ import annotations

import pytest
import torch

from vllm_fl.quantization.w8a8 import (
    dynamic_per_token_quant_int8,
    unpack_uint8b128_int32,
    w8a8_linear_reference,
)


def _expected_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Little-endian byte extraction implemented with shifts, not views."""
    shifts = torch.arange(0, 32, 8, dtype=torch.int32)
    bytes_ = (packed.unsqueeze(-1) >> shifts) & 0xFF
    return (bytes_.reshape(packed.shape[0], -1).to(torch.int16) - 128).to(torch.int8)


def test_unpack_uint8b128_int32_matches_shift_reference():
    packed = torch.tensor(
        [
            [0x00010203, 0x7F7E7D7C],
            [0x7F7E7D80, 0x00FFFFFF],
        ],
        dtype=torch.int32,
    )
    unpacked = unpack_uint8b128_int32(packed)
    expected = _expected_unpack(packed)
    assert unpacked.dtype == torch.int8
    assert unpacked.shape == (2, 8)
    assert torch.equal(unpacked, expected)
    # Little-endian storage, then offset-binary (byte - 128) per field:
    # 0x00010203 -> 03 02 01 00 -> -125..-128 and
    # 0x7F7E7D7C -> 7C 7D 7E 7F -> -4..-1.
    assert unpacked[0].tolist() == [-125, -126, -127, -128, -4, -3, -2, -1]
    # The low byte of 0x7F7E7D80 is 0x80 -> 0.
    assert unpacked[1, 0].item() == 0


def test_unpack_uint8b128_int32_truncates_to_in_features():
    packed = torch.arange(4, dtype=torch.int32).reshape(2, 2)
    unpacked = unpack_uint8b128_int32(packed, in_features=5)
    assert unpacked.shape == (2, 5)
    assert torch.equal(unpacked, _expected_unpack(packed)[:, :5])


@pytest.mark.parametrize("in_features", [-1, 9])
def test_unpack_uint8b128_int32_rejects_bad_in_features(in_features):
    packed = torch.zeros(2, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match="in_features is incompatible"):
        unpack_uint8b128_int32(packed, in_features=in_features)


@pytest.mark.parametrize(
    "tensor",
    [
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, 2, dtype=torch.int64),
        torch.zeros(2, 2, 2, dtype=torch.int32),
    ],
)
def test_unpack_uint8b128_int32_requires_2d_int32(tensor):
    with pytest.raises(ValueError, match="2D int32 tensor"):
        unpack_uint8b128_int32(tensor)


def test_dynamic_per_token_quant_int8_matches_reference():
    x = torch.tensor(
        [
            [-128.0, 0.0, 64.0, 256.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, -2.0, 3.0, -4.0],
        ]
    )
    quantized, scale = dynamic_per_token_quant_int8(x)

    absmax = x.abs().amax(dim=-1, keepdim=True)
    expected_scale = absmax / 127.0
    safe = torch.where(absmax != 0, absmax, torch.ones_like(absmax))
    inv = torch.where(absmax != 0, 127.0 / safe, torch.zeros_like(safe))
    expected_q = (x * inv).round().clamp(-128, 127).to(torch.int8)

    assert quantized.dtype == torch.int8
    assert torch.equal(quantized, expected_q)
    assert torch.equal(scale, expected_scale)
    # The all-zero row keeps scale 0 (not a division-by-zero sentinel).
    assert scale[1, 0].item() == 0.0
    assert torch.equal(quantized[1], torch.zeros(4, dtype=torch.int8))


def test_dynamic_per_token_quant_int8_preserves_shape_metadata():
    x = torch.linspace(-3.0, 3.0, 12).reshape(3, 4)
    quantized, scale = dynamic_per_token_quant_int8(x)
    assert quantized.shape == x.shape
    assert scale.shape == (3, 1)
    assert scale.dtype == torch.float32


@pytest.mark.parametrize(
    "x",
    [torch.zeros(4), torch.zeros(2, 2, 2)],
)
def test_dynamic_per_token_quant_int8_requires_2d(x):
    with pytest.raises(ValueError, match="2D"):
        dynamic_per_token_quant_int8(x)


def test_dynamic_per_token_quant_int8_requires_float():
    with pytest.raises(TypeError, match="floating point"):
        dynamic_per_token_quant_int8(torch.zeros(2, 4, dtype=torch.int8))


def _linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Independent W8A8 formula: per-row scale, int32 matmul, two scales."""
    x_2d = x.reshape(-1, x.shape[-1]).to(torch.float32)
    absmax = x_2d.abs().amax(dim=-1, keepdim=True)
    scale = absmax / 127.0
    safe = torch.where(absmax != 0, absmax, torch.ones_like(absmax))
    inv = torch.where(absmax != 0, 127.0 / safe, torch.zeros_like(safe))
    x_q = (x_2d * inv).round().clamp(-128, 127).to(torch.int8).to(torch.int32)
    acc = x_q @ weight.to(torch.int32).t()
    out = acc.to(torch.float32) * scale * weight_scale.reshape(1, -1).float()
    if bias is not None:
        out = out + bias.reshape(1, -1)
    return out.to(x.dtype).reshape(*x.shape[:-1], weight.shape[0])


def test_w8a8_linear_reference_matches_independent_formula():
    torch.manual_seed(0)
    x = torch.randn(5, 16, dtype=torch.float16)
    weight = torch.randint(-128, 127, (8, 16), dtype=torch.int8)
    weight_scale = torch.linspace(0.05, 0.4, 8)
    out = w8a8_linear_reference(x, weight, weight_scale)
    expected = _linear_reference(x, weight, weight_scale)
    assert out.shape == (5, 8)
    assert out.dtype == torch.float16
    assert torch.allclose(out, expected, atol=0, rtol=0)


def test_w8a8_linear_reference_adds_bias_and_flattens_leading_dims():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 16, dtype=torch.float32)
    weight = torch.randint(-128, 127, (4, 16), dtype=torch.int8)
    weight_scale = torch.tensor([0.1, 0.2, 0.3, 0.4])
    bias = torch.tensor([1.0, -1.0, 0.5, 0.0])
    out = w8a8_linear_reference(x, weight, weight_scale, bias)
    expected = _linear_reference(x, weight, weight_scale, bias)
    assert out.shape == (2, 3, 4)
    assert torch.equal(out, expected)


def test_w8a8_linear_reference_rejects_bad_weight():
    with pytest.raises(ValueError, match="2D int8"):
        w8a8_linear_reference(
            torch.zeros(2, 4),
            torch.zeros(4, dtype=torch.int8),
            torch.ones(4),
        )
    with pytest.raises(ValueError, match="2D int8"):
        w8a8_linear_reference(
            torch.zeros(2, 4),
            torch.zeros(4, 4, dtype=torch.int32),
            torch.ones(4),
        )


def test_w8a8_linear_reference_rejects_incompatible_input():
    weight = torch.zeros(4, 8, dtype=torch.int8)
    with pytest.raises(ValueError, match="incompatible input dimensions"):
        w8a8_linear_reference(torch.zeros(2, 7), weight, torch.ones(4))
    with pytest.raises(ValueError, match="incompatible input dimensions"):
        w8a8_linear_reference(torch.zeros(8), weight, torch.ones(4))


def test_w8a8_linear_reference_validates_scale_and_bias_width():
    weight = torch.zeros(4, 8, dtype=torch.int8)
    x = torch.zeros(2, 8)
    with pytest.raises(ValueError, match="one value per output channel"):
        w8a8_linear_reference(x, weight, torch.ones(3))
    with pytest.raises(ValueError, match="bias must contain one value"):
        w8a8_linear_reference(x, weight, torch.ones(4), torch.ones(5))
