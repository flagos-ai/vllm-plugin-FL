import pytest
import torch

from vllm_fl.ops.layernorm import RMSNormFL


def manual_rms_norm(x, weight, eps, residual=None):
    if residual is not None:
        x = x + residual
        updated_residual = x
    else:
        updated_residual = None

    orig_dtype = x.dtype
    x_f32 = x.float()

    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = x_f32 * torch.rsqrt(variance + eps)

    output = (hidden_states * weight.float()).to(orig_dtype)

    if residual is not None:
        return output, updated_residual
    return output


def assert_allclose(actual, expected, dtype, msg=""):
    if dtype == torch.bfloat16:
        rtol, atol = 2e-2, 2e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=msg)
    except AssertionError as e:
        diff = (actual.float() - expected.float()).abs().max().item()
        print(
            f"\n[Fail Info] Dtype: {dtype} | Max Diff: {diff:.6f} | Tolerance: {rtol}"
        )
        raise e


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA")
@pytest.mark.parametrize("batch_size", [2, 8])
@pytest.mark.parametrize("seq_len", [128, 2048])
@pytest.mark.parametrize("hidden_size", [1024, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rms_norm_forward_oot(batch_size, seq_len, hidden_size, dtype):
    torch.manual_seed(42)
    device = "cuda"
    eps = 1e-6

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

    model = RMSNormFL(hidden_size, eps=eps, dtype=dtype).to(device)

    out_fl = model.forward_oot(x, residual=None)

    out_ref = manual_rms_norm(x, model.weight, eps, residual=None)

    assert_allclose(out_fl, out_ref, dtype, msg=f"Basic Norm Mismatch {dtype}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA")
@pytest.mark.parametrize("hidden_size", [1024, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_with_residual(hidden_size, dtype):
    torch.manual_seed(42)
    device = "cuda"
    batch_size = 4
    seq_len = 128
    eps = 1e-6

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    residual = torch.randn_like(x)

    x_ref = x.clone()
    res_ref = residual.clone()

    model = RMSNormFL(hidden_size, eps=eps, dtype=dtype).to(device)

    out_fl, res_fl = model.forward_oot(x, residual=residual)

    out_ref, res_ref_out = manual_rms_norm(x_ref, model.weight, eps, residual=res_ref)

    assert_allclose(res_fl, res_ref_out, dtype, msg="Residual Value Mismatch")

    assert_allclose(out_fl, out_ref, dtype, msg="Norm Output with Residual Mismatch")


if __name__ == "__main__":
    import sys

    from pytest import main

    sys.exit(main(["-v", __file__]))
