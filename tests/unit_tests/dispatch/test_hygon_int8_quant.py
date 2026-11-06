from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("vllm")

from vllm_fl.dispatch.backends.vendor.hygon.impl.native_ops.int8_quant import (
    dynamic_scaled_int8_quant_hygon,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP device")
def test_dynamic_symmetric_int8_quant_matches_reference():
    device = torch.device("cuda")
    input = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0], [2.0, -1.25, 0.75, -2.0]],
            [[0.25, -0.5, 1.0, -0.75], [-3.0, 1.0, 2.5, -2.0]],
        ],
        device=device,
        dtype=torch.float16,
    )
    result = torch.empty_like(input, dtype=torch.int8)
    scales = torch.empty((input.numel() // input.shape[-1], 1), device=device)

    dynamic_scaled_int8_quant_hygon(result, input, scales, None)

    rows = input.float().view(-1, input.shape[-1])
    expected_scales = rows.abs().amax(dim=1, keepdim=True) / 127.0
    safe_scales = torch.where(expected_scales == 0, 1.0, expected_scales)
    expected = torch.where(
        expected_scales == 0,
        torch.zeros_like(rows),
        torch.round(rows / safe_scales),
    ).to(torch.int8)

    torch.testing.assert_close(scales, expected_scales)
    torch.testing.assert_close(result.view_as(expected), expected, rtol=0, atol=0)


def test_dynamic_asymmetric_int8_quant_is_explicitly_unsupported():
    input = torch.empty((1, 4))
    result = torch.empty_like(input, dtype=torch.int8)
    scales = torch.empty((1, 1))
    azp = torch.empty((1, 1), dtype=torch.int32)

    with pytest.raises(NotImplementedError, match="dynamic symmetric"):
        dynamic_scaled_int8_quant_hygon(result, input, scales, azp)
