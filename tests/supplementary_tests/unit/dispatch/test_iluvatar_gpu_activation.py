# Copyright 2025 BAAI. All rights reserved.
"""Real-BI-V150 coverage for the FlagGems activation custom-op paths.

Two activation files sit at 0 coverage on the CPU baseline because their
compute kernels only execute on a real device:

* ``vllm_fl/dispatch/backends/vendor/iluvatar/activation.py`` — the
  ``torch.library.custom_op("vllm_fl::iluvatar_gems_silu_and_mul")``
  wrapping ``flaggems_vllm.silu_and_mul`` (used by the FlagGems backend when
  the iluvatar vendor path is active), plus its fake/meta registration;
* ``vllm_fl/dispatch/backends/flaggems/impl/activation.py::silu_and_mul_flaggems``
  — the vendor dispatch that routes into the custom op on iluvatar, and the
  generic (non-iluvatar) flaggems path exercised through
  ``gelu_and_mul_flaggems``.

All expectations are computed with independent torch oracles
(``F.silu(a) * b``, erf-based GELU). Tolerances follow
``tests/platforms/iluvatar.yaml``: default rtol/atol 1e-3; bf16/fp16
activations compared after upcasting, using the yaml's bfloat16 tolerance
1e-2 for bf16 inputs and the default 1e-3 for fp16/fp32 (fp16 is not granted
the bf16 slack — it is held to the default tolerance as the yaml does not
list a separate fp16 entry).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.supplementary,
    pytest.mark.platforms("iluvatar"),
    pytest.mark.gpu,
]

# tests/platforms/iluvatar.yaml tolerance.inference
RTOL_DEFAULT = 1e-3
ATOL_DEFAULT = 1e-3
RTOL_BF16 = 1e-2
ATOL_BF16 = 1e-2


def _device():
    from tests.utils.device_utils import get_device

    return get_device()


def _tol_for(dtype):
    if dtype == torch.bfloat16:
        return RTOL_BF16, ATOL_BF16
    return RTOL_DEFAULT, ATOL_DEFAULT


def _oracle_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    a = x[..., :half].float()
    b = x[..., half:].float()
    return F.silu(a) * b


def _oracle_gelu_and_mul(x: torch.Tensor, approximate: str) -> torch.Tensor:
    half = x.shape[-1] // 2
    a = x[..., :half].float()
    b = x[..., half:].float()
    return F.gelu(a, approximate=approximate) * b


class TestIluvatarSiluAndMulCustomOp:
    """The ``vllm_fl::iluvatar_gems_silu_and_mul`` custom op on the device."""

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=str
    )
    def test_matches_torch_oracle(self, dtype):
        from vllm_fl.dispatch.backends.vendor.iluvatar.activation import (
            _iluvatar_gems_silu_and_mul,
        )

        device = _device()
        x = torch.randn(64, 128, dtype=dtype, device=device)
        out = _iluvatar_gems_silu_and_mul(x)
        assert out.shape == (64, 64)
        assert out.dtype == dtype
        assert out.device == device

        rtol, atol = _tol_for(dtype)
        torch.testing.assert_close(
            out.float(), _oracle_silu_and_mul(x), rtol=rtol, atol=atol
        )

    @pytest.mark.parametrize("shape", [(1, 64), (7, 256), (128, 512)])
    def test_shapes(self, shape):
        from vllm_fl.dispatch.backends.vendor.iluvatar.activation import (
            _iluvatar_gems_silu_and_mul,
        )

        device = _device()
        x = torch.randn(*shape, dtype=torch.float32, device=device)
        out = _iluvatar_gems_silu_and_mul(x)
        assert out.shape == (*shape[:-1], shape[-1] // 2)

    def test_fake_kernel_shape_only(self):
        """The registered fake/meta kernel must produce the right shape.

        Runs under ``torch._subclasses.FakeTensorMode`` so the real kernel
        is never launched — this covers the ``register_fake`` body.
        """
        from vllm_fl.dispatch.backends.vendor.iluvatar.activation import (
            _iluvatar_gems_silu_and_mul,
        )

        from torch._subclasses.fake_tensor import FakeTensorMode

        device = _device()
        with FakeTensorMode():
            x = torch.randn(3, 20, dtype=torch.bfloat16, device=device)
            out = _iluvatar_gems_silu_and_mul(x)
            assert out.shape == (3, 10)
            assert out.dtype == torch.bfloat16

    def test_op_registered_in_torch_namespace(self):
        """The custom op is a real torch.library op, callable via torch.ops."""
        device = _device()
        x = torch.randn(8, 32, dtype=torch.float32, device=device)
        out = torch.ops.vllm_fl.iluvatar_gems_silu_and_mul(x)
        torch.testing.assert_close(
            out.float(), _oracle_silu_and_mul(x), rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT
        )


class TestFlaggemsActivationDispatch:
    """``silu_and_mul_flaggems`` vendor routing on the real platform."""

    def test_iluvatar_vendor_routes_to_custom_op(self):
        """On the iluvatar platform the FlagGems backend uses the custom op.

        The platform resolves with vendor_name == "iluvatar" in the container,
        so ``silu_and_mul_flaggems`` must take the iluvatar branch and produce
        results identical to the torch oracle.
        """
        from vllm.platforms import current_platform

        from vllm_fl.dispatch.backends.flaggems.impl.activation import (
            silu_and_mul_flaggems,
        )

        assert getattr(current_platform, "vendor_name", None) == "iluvatar"

        device = _device()
        x = torch.randn(32, 96, dtype=torch.bfloat16, device=device)
        out = silu_and_mul_flaggems(None, x)
        assert out.shape == (32, 48)
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(
            out.float(), _oracle_silu_and_mul(x), rtol=RTOL_BF16, atol=ATOL_BF16
        )

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=str
    )
    def test_silu_and_mul_flaggems_matches_oracle(self, dtype):
        from vllm_fl.dispatch.backends.flaggems.impl.activation import (
            silu_and_mul_flaggems,
        )

        device = _device()
        x = torch.randn(16, 64, dtype=dtype, device=device)
        out = silu_and_mul_flaggems(None, x)
        assert out.dtype == dtype
        rtol, atol = _tol_for(dtype)
        torch.testing.assert_close(
            out.float(), _oracle_silu_and_mul(x), rtol=rtol, atol=atol
        )

    @pytest.mark.parametrize("approximate", ["none", "tanh"], ids=str)
    def test_gelu_and_mul_flaggems_matches_oracle(self, approximate):
        from vllm_fl.dispatch.backends.flaggems.impl.activation import (
            gelu_and_mul_flaggems,
        )

        device = _device()
        x = torch.randn(16, 64, dtype=torch.float32, device=device)
        layer = type("Layer", (), {"approximate": approximate})()
        out = gelu_and_mul_flaggems(layer, x)
        assert out.dtype == torch.float32
        torch.testing.assert_close(
            out.float(),
            _oracle_gelu_and_mul(x, approximate),
            rtol=RTOL_DEFAULT,
            atol=ATOL_DEFAULT,
        )

    def test_gelu_and_mul_default_approximate_none(self):
        """``obj=None`` falls back to the "none" (exact erf) approximation."""
        from vllm_fl.dispatch.backends.flaggems.impl.activation import (
            gelu_and_mul_flaggems,
        )

        device = _device()
        x = torch.randn(8, 32, dtype=torch.float32, device=device)
        out = gelu_and_mul_flaggems(None, x)
        torch.testing.assert_close(
            out.float(),
            _oracle_gelu_and_mul(x, "none"),
            rtol=RTOL_DEFAULT,
            atol=ATOL_DEFAULT,
        )
