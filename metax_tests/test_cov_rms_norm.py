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

"""Functional tests for MetaX MacaBackend.rms_norm (with and without residual)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.utils.platform_config import PlatformConfig

pytestmark = [pytest.mark.gpu, pytest.mark.functional]


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Verify accelerator visibility via MACA_VISIBLE_DEVICES and bind cuda:0."""
    assert torch.cuda.is_available(), (
        "No accelerator detected by torch.cuda.is_available(); "
        "ensure MACA_VISIBLE_DEVICES is set and /dev/dri, /dev/mxcd are mounted."
    )
    return torch.device("cuda:0")


@pytest.fixture(scope="session")
def metax_tolerance() -> tuple[float, float]:
    """Load MetaX C550 inference default tolerance from tests/platforms/metax.yaml."""
    tol = PlatformConfig.load("metax", "c550").get_tolerance("inference", "default")
    return float(tol.rtol), float(tol.atol)


def _init_maca_backend(required_op: str):
    """Initialize vLLM/vLLM-FL MetaX runtime and verify the target _C op exists."""
    try:
        import vllm  # noqa: F401

        import vllm_fl  # noqa: F401
        from vllm_fl.dispatch.backends.vendor.metax.metax import MacaBackend
    except Exception as exc:
        pytest.fail(f"Failed to initialize MetaX vLLM-FL runtime: {exc!r}")

    if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, required_op):
        pytest.fail(
            f"Missing MetaX operator torch.ops._C.{required_op}; "
            "verify mcoplib is installed and loaded in the CI image."
        )
    return MacaBackend()


def _reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute reference RMSNorm in fp32 and cast back to input dtype."""
    x_fp32 = x.float()
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * rms * weight.float()).to(x.dtype)


def test_maca_rms_norm_without_residual(
    device: torch.device,
    metax_tolerance: tuple[float, float],
) -> None:
    """Verify MacaBackend.rms_norm returns normalized output and leaves x unchanged."""
    backend = _init_maca_backend("rms_norm")

    dtype = torch.float16
    eps = 1e-6
    x = torch.tensor(
        [[1.0, -2.0, 0.5, 1.5], [-1.0, 0.5, 2.0, -1.5]],
        device=device,
        dtype=dtype,
    )
    x_before = x.clone()
    weight = torch.tensor([1.0, 0.5, 1.5, 2.0], device=device, dtype=dtype)
    obj = SimpleNamespace(
        weight=torch.nn.Parameter(weight, requires_grad=False),
        variance_epsilon=eps,
    )

    expected = _reference_rms_norm(x_before, weight, eps)
    out = backend.rms_norm(obj, x, residual=None)

    assert isinstance(out, torch.Tensor)
    assert out.shape == x_before.shape
    assert out.dtype == dtype
    assert torch.equal(x, x_before)

    rtol, atol = metax_tolerance
    assert torch.allclose(out, expected, rtol=rtol, atol=atol), (
        f"RMSNorm output mismatch: max_diff={(out - expected).abs().max().item():.6f}"
    )


def test_maca_rms_norm_with_residual(
    device: torch.device,
    metax_tolerance: tuple[float, float],
) -> None:
    """Verify MacaBackend.rms_norm updates residual and x in-place using pre-call copies."""
    backend = _init_maca_backend("fused_add_rms_norm")

    dtype = torch.float16
    eps = 1e-6
    x = torch.tensor(
        [[1.0, -2.0, 0.5, 1.5], [-1.0, 0.5, 2.0, -1.5]],
        device=device,
        dtype=dtype,
    )
    residual = torch.tensor(
        [[0.5, 1.0, -0.25, 0.5], [1.5, -0.5, -1.0, 0.5]],
        device=device,
        dtype=dtype,
    )
    x_before = x.clone()
    residual_before = residual.clone()
    weight = torch.tensor([1.0, 0.5, 1.5, 2.0], device=device, dtype=dtype)
    obj = SimpleNamespace(
        weight=torch.nn.Parameter(weight, requires_grad=False),
        variance_epsilon=eps,
    )

    expected_residual = (x_before.float() + residual_before.float()).to(dtype)
    expected_x = _reference_rms_norm(expected_residual, weight, eps)

    out_x, out_residual = backend.rms_norm(obj, x, residual=residual)

    assert out_x is x
    assert out_residual is residual

    rtol, atol = metax_tolerance
    assert torch.allclose(residual, expected_residual, rtol=rtol, atol=atol), (
        f"Residual in-place sum mismatch: "
        f"max_diff={(residual - expected_residual).abs().max().item():.6f}"
    )
    assert torch.allclose(x, expected_x, rtol=rtol, atol=atol), (
        f"RMSNorm in-place output mismatch: "
        f"max_diff={(x - expected_x).abs().max().item():.6f}"
    )
