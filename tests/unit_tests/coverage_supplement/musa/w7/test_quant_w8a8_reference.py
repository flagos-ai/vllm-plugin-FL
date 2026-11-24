# Coverage tests: vllm_fl/quantization/w8a8/reference.py (CPU only).

import pytest
import torch

from vllm_fl.quantization.w8a8 import reference as ref


def test_unpack_roundtrip_and_shape():
    values = torch.tensor(
        [
            [-128, -127, -1, 0, 1, 126, 127, 42],
            [127, 0, -128, 1, -1, 64, -64, 7],
        ],
        dtype=torch.int8,
    )
    codes = (values.to(torch.int16) + 128).to(torch.uint8)
    packed = codes.contiguous().view(torch.int32)

    unpacked = ref.unpack_uint8b128_int32(packed)

    assert unpacked.dtype == torch.int8
    assert unpacked.shape == values.shape
    assert torch.equal(unpacked, values)


def test_unpack_in_features_slices_and_validates():
    values = torch.arange(-8, 8, dtype=torch.int8).reshape(2, 8)
    codes = (values.to(torch.int16) + 128).to(torch.uint8)
    packed = codes.contiguous().view(torch.int32)

    sliced = ref.unpack_uint8b128_int32(packed, in_features=6)
    assert torch.equal(sliced, values[:, :6])

    with pytest.raises(ValueError):
        ref.unpack_uint8b128_int32(packed, in_features=-1)
    with pytest.raises(ValueError):
        ref.unpack_uint8b128_int32(packed, in_features=9)


def test_unpack_rejects_non_int32_or_non_2d():
    with pytest.raises(ValueError):
        ref.unpack_uint8b128_int32(torch.zeros(4, dtype=torch.int32))
    with pytest.raises(ValueError):
        ref.unpack_uint8b128_int32(torch.zeros(2, 4, dtype=torch.int8))


def test_dynamic_per_token_quant_zero_and_nonzero_rows():
    x = torch.tensor(
        [
            [0.0, 1.0, -2.0, 0.5],
            [10.0, -5.0, 2.5, 0.0],
        ],
        dtype=torch.float32,
    )
    quantized, scales = ref.dynamic_per_token_quant_int8(x)

    assert quantized.dtype == torch.int8
    assert scales.dtype == torch.float32
    assert scales.shape == (2, 1)
    assert quantized[0, 2] == -127
    assert quantized[1, 0] == 127
    assert torch.allclose(
        scales[:, 0], torch.tensor([2.0 / 127, 10.0 / 127], dtype=torch.float32)
    )

    zeros_q, zeros_s = ref.dynamic_per_token_quant_int8(torch.zeros(3, 4))
    assert torch.count_nonzero(zeros_q) == 0
    assert torch.count_nonzero(zeros_s) == 0
    assert torch.isfinite(zeros_s).all()


def test_dynamic_quant_validates_inputs():
    with pytest.raises(ValueError):
        ref.dynamic_per_token_quant_int8(torch.ones(2, 3, 4))
    with pytest.raises(TypeError):
        ref.dynamic_per_token_quant_int8(torch.ones(2, 4, dtype=torch.int8))


def test_normalize_channel_scale_shapes_and_validation():
    scale, ok = ref._normalize_channel_scale(torch.ones(4), 4), True
    assert scale.shape == (1, 4)
    assert scale.dtype == torch.float32
    assert ok

    with pytest.raises(ValueError, match="one value per output channel"):
        ref._normalize_channel_scale(torch.ones(3), 4)


def test_linear_reference_matches_int32_reference_and_shapes():
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

    x_q, x_scale = ref.dynamic_per_token_quant_int8(x.reshape(-1, 4))
    expected = (
        (x_q.reshape(-1, 4).int() @ weight.int().t()).float()
        * x_scale.reshape(-1, 1)
        * weight_scale.reshape(1, -1)
        + bias
    ).reshape(1, 2, 2)

    actual = ref.w8a8_linear_reference(x, weight, weight_scale, bias)
    assert actual.shape == (1, 2, 2)
    assert torch.allclose(actual, expected)

    no_bias = ref.w8a8_linear_reference(x.reshape(-1, 4), weight, weight_scale)
    assert no_bias.shape == (2, 2)


def test_linear_reference_rejects_bad_weight_and_x():
    x = torch.ones(2, 4)
    with pytest.raises(ValueError, match="2D int8"):
        ref.w8a8_linear_reference(x, torch.ones(4), torch.ones(1))
    with pytest.raises(ValueError, match="2D int8"):
        ref.w8a8_linear_reference(x, torch.ones(2, 4), torch.ones(2))
    with pytest.raises(ValueError, match="incompatible input dimensions"):
        ref.w8a8_linear_reference(torch.ones(4), torch.ones(2, 4, dtype=torch.int8), torch.ones(2))
    with pytest.raises(ValueError, match="incompatible input dimensions"):
        ref.w8a8_linear_reference(
            torch.ones(2, 3), torch.ones(2, 4, dtype=torch.int8), torch.ones(2)
        )


def test_linear_reference_rejects_bad_bias():
    with pytest.raises(ValueError, match="one value per output channel"):
        ref.w8a8_linear_reference(
            torch.ones(2, 4),
            torch.ones(2, 4, dtype=torch.int8),
            torch.ones(2),
            bias=torch.ones(3),
        )
