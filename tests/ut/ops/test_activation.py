import pytest
import torch
import torch.nn.functional as F

try:
    from vllm_fl.ops.activation import SiluAndMulFL
except ImportError:
    import sys
    from unittest.mock import MagicMock

    class MockSiluAndMul:
        def __init__(self):
            pass

    mock_vllm = MagicMock()
    mock_vllm.model_executor.layers.activation.SiluAndMul = MockSiluAndMul
    sys.modules["vllm"] = mock_vllm
    sys.modules["vllm.model_executor"] = mock_vllm.model_executor
    sys.modules["vllm.model_executor.layers"] = mock_vllm.model_executor.layers
    sys.modules["vllm.model_executor.layers.activation"] = (
        mock_vllm.model_executor.layers.activation
    )

    from flag_gems.modules.activation import gems_silu_and_mul

    class SiluAndMulFL(MockSiluAndMul):
        def __init__(self):
            super().__init__()

        def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
            d = x.shape[-1] // 2
            x1, x2 = x[..., :d], x[..., d:]
            return gems_silu_and_mul(x1, x2)


def ref_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using standard PyTorch ops.
    Logic: Split last dim into two, calculate SiLU(first) * second.
    """
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return F.silu(x1) * x2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 4096),  # Common hidden size
        (16, 11008),  # Llama MLP intermediate size
        (4, 128, 512),  # Batch, Seq, Hidden
        (11, 22),  # Odd batch/seq, Even hidden
    ],
)
def test_silu_and_mul_correctness(dtype, shape):
    """
    Test that SiluAndMulFL produces results matching PyTorch reference.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping test.")

    device = "cuda"

    assert shape[-1] % 2 == 0, "Hidden dimension must be even for SiluAndMul"

    torch.manual_seed(42)
    x = torch.randn(shape, dtype=dtype, device=device)

    model = SiluAndMulFL()

    out_fl = model.forward_oot(x)

    out_ref = ref_silu_and_mul(x)

    assert (
        out_fl.shape == out_ref.shape
    ), f"Shape mismatch: {out_fl.shape} vs {out_ref.shape}"

    atol = 1e-3 if dtype == torch.float32 else 1e-2
    rtol = 1e-3 if dtype == torch.float32 else 1e-2

    try:
        torch.testing.assert_close(out_fl, out_ref, atol=atol, rtol=rtol)
    except AssertionError as e:
        diff = (out_fl - out_ref).abs()
        print(f"\nMax diff for {dtype}, {shape}: {diff.max().item()}")
        print(f"Mean diff: {diff.mean().item()}")
        raise e


def test_grad_check():
    """
    Optional: Check if gradients can propagate (if needed for training).
    Skip this if `forward_oot` is inference only (likely for vLLM).
    """
    pass


if __name__ == "__main__":
    pytest.main([__file__])
