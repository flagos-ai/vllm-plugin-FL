import pytest
import torch

from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    per_token_quant_int8,
)

from vllm_fl.ops.deepseek_v4_silu_int8 import silu_mul_dynamic_int8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("rows", [1, 17, 257])
def test_silu_mul_dynamic_int8_matches_native_marlin(rows: int) -> None:
    torch.manual_seed(123)
    x = torch.randn((rows, 4096), device="cuda", dtype=torch.bfloat16)
    activated = torch.empty((rows, 2048), device="cuda", dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul(activated, x)
    expected_q, expected_scale = per_token_quant_int8(activated)

    actual_q, actual_scale = silu_mul_dynamic_int8(x)

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
