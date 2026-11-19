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

"""Functional tests for MetaX MacaBackend activation operators."""

from __future__ import annotations

import math

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


def _reference_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Compute reference SiLU-and-mul in fp32 and cast back to input dtype."""
    half = x.shape[-1] // 2
    x_fp32 = x.float()
    gate = x_fp32[..., :half]
    up = x_fp32[..., half:]
    silu_gate = gate * torch.sigmoid(gate)
    return (silu_gate * up).to(x.dtype)


def _reference_gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Compute reference exact GELU (approximate='none') and mul in fp32."""
    half = x.shape[-1] // 2
    x_fp32 = x.float()
    gate = x_fp32[..., :half]
    up = x_fp32[..., half:]
    gelu_gate = 0.5 * gate * (1.0 + torch.erf(gate * math.sqrt(0.5)))
    return (gelu_gate * up).to(x.dtype)


def _fixed_activation_input(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build a small deterministic [2, 8] input tensor for gated activations."""
    return torch.tensor(
        [
            [1.0, -2.0, 0.5, 0.0, 1.5, -0.5, 2.0, -1.0],
            [-1.5, 0.25, 2.0, -0.75, -0.5, 1.25, -1.5, 0.5],
        ],
        device=device,
        dtype=dtype,
    )


def test_maca_silu_and_mul(
    device: torch.device,
    metax_tolerance: tuple[float, float],
) -> None:
    """Verify MacaBackend.silu_and_mul matches the fp32 SiLU-and-mul reference."""
    from vllm.config import VllmConfig, set_current_vllm_config

    backend = _init_maca_backend("silu_and_mul")

    dtype = torch.float16
    x = _fixed_activation_input(device, dtype)
    x_before = x.clone()

    expected = _reference_silu_and_mul(x_before)
    with set_current_vllm_config(VllmConfig()):
        out = backend.silu_and_mul(None, x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (x_before.shape[0], x_before.shape[1] // 2)
    assert out.dtype == dtype
    assert torch.equal(x, x_before)

    rtol, atol = metax_tolerance
    assert torch.allclose(out, expected, rtol=rtol, atol=atol), (
        f"silu_and_mul output mismatch: max_diff={(out - expected).abs().max().item():.6f}"
    )


def test_maca_gelu_and_mul(
    device: torch.device,
    metax_tolerance: tuple[float, float],
) -> None:
    """Verify MacaBackend.gelu_and_mul matches the fp32 exact GELU-and-mul reference."""
    from vllm.config import VllmConfig, set_current_vllm_config

    backend = _init_maca_backend("gelu_and_mul")

    dtype = torch.float16
    x = _fixed_activation_input(device, dtype)
    x_before = x.clone()

    expected = _reference_gelu_and_mul(x_before)
    with set_current_vllm_config(VllmConfig()):
        out = backend.gelu_and_mul(None, x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (x_before.shape[0], x_before.shape[1] // 2)
    assert out.dtype == dtype
    assert torch.equal(x, x_before)

    rtol, atol = metax_tolerance
    assert torch.allclose(out, expected, rtol=rtol, atol=atol), (
        f"gelu_and_mul output mismatch: max_diff={(out - expected).abs().max().item():.6f}"
    )
