# Copyright (c) 2026 BAAI. All rights reserved.

import pytest
import torch

from vllm_fl.quantization.wna16.reference import (
    unpack_uint4b8,
    wna16_gemm_reference,
)


def _pack_uint4b8(values: torch.Tensor) -> torch.Tensor:
    codes = (values.to(torch.int32) + 8).reshape(values.shape[0], -1, 8)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    return torch.sum(codes << shifts, dim=-1).to(torch.int32)


def test_wna16_reference_matches_dequantized_matmul():
    values = torch.tensor(
        [
            [-8, -7, -1, 0, 1, 4, 6, 7],
            [7, 6, 4, 1, 0, -1, -7, -8],
        ],
        dtype=torch.int8,
    )
    packed = _pack_uint4b8(values)
    scales = torch.tensor([[0.5, 2.0], [1.5, 0.25]])
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 8
    bias = torch.tensor([0.25, -0.5])

    expected_weight = values.float() * scales.repeat_interleave(4, dim=1)
    expected = x @ expected_weight.t() + bias
    actual = wna16_gemm_reference(x, packed, scales, 4, bias)

    assert torch.equal(unpack_uint4b8(packed), values)
    assert torch.allclose(actual, expected)


def test_wna16_reference_rejects_incompatible_scales():
    x = torch.ones((1, 8))
    packed = torch.zeros((2, 1), dtype=torch.int32)
    with pytest.raises(ValueError, match="weight_scale"):
        wna16_gemm_reference(x, packed, torch.ones((2, 1)), 4)
