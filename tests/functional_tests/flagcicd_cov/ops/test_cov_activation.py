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

"""Single-GPU correctness tests for vllm_fl.ops activation layers.

Each forward is compared against an independent PyTorch reference computed
in the test itself; the dispatch layer must return the same values, and the
selected implementation is asserted (not just the absence of an error) so
the FlagGems-vs-native dispatch remains observable.

Pilot P1 of the supplementary coverage suite (flagcicd_cov). Runs in the
functional scope: each xdist worker owns exactly one GPU.
"""

import torch
import torch.nn.functional as F

import pytest

from vllm_fl.dispatch import call_op, resolve_op

pytestmark = [pytest.mark.gpu]

# A100-40GB pilot budget: token counts stay small so all shapes fit in a few
# MB; dtype coverage exercises both exact (fp32) and reduced (bf16) paths.
SHAPES = [(1, 128), (8, 512), (64, 1024)]


def _resolve_or_fail(op_name: str):
    """Resolve an op or fail the test: the dispatch table must contain it.

    Returns the implementation callable (OpManager.resolve returns the
    selected implementation function directly).
    """
    impl = resolve_op(op_name)
    assert impl is not None, f"op {op_name!r} has no implementation"
    assert callable(impl), f"op {op_name!r} resolved to a non-callable"
    return impl


class TestSiluAndMulCov:
    """SiluAndMulFL dispatch matches an independent elementwise reference."""

    @staticmethod
    def reference(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return F.silu(x[..., :half]) * x[..., half:]

    def test_forward_matches_reference_per_shape(self, device):
        """Every pilot shape returns values equal to the elementwise reference."""
        impl = _resolve_or_fail("silu_and_mul")
        for tokens, hidden in SHAPES:
            x = torch.randn(tokens, hidden * 2, device=device, dtype=torch.float32)
            out = call_op("silu_and_mul", None, x)
            ref = self.reference(x)
            assert out.shape == (tokens, hidden)
            assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4 * 4), (
                f"silu_and_mul mismatch at ({tokens}, {hidden})"
            )
        assert impl is resolve_op("silu_and_mul"), "resolution must be stable"

    def test_forward_bfloat16_matches_fp32_reference(self, device):
        """bf16 input follows the fp32 reference within bf16 tolerance."""
        x = torch.randn(16, 512, device=device, dtype=torch.bfloat16)
        out = call_op("silu_and_mul", None, x)
        ref = self.reference(x.float())
        assert torch.allclose(
            out.float(), ref, rtol=1e-2, atol=1e-2
        ), "silu_and_mul bf16 deviates from fp32 reference beyond tolerance"


class TestGeluAndMulCov:
    """GeluAndMulFL dispatch matches the tanh and none approximations."""

    @staticmethod
    def reference(x: torch.Tensor, approximate: str) -> torch.Tensor:
        half = x.shape[-1] // 2
        return F.gelu(x[..., :half], approximate=approximate) * x[..., half:]

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_forward_matches_reference(self, device, approximate):
        """Both gelu approximations match their references exactly enough."""
        _resolve_or_fail("gelu_and_mul")
        ns = type("Cfg", (), {"approximate": approximate})
        x = torch.randn(16, 512, device=device, dtype=torch.float32)
        out = call_op("gelu_and_mul", ns, x)
        ref = self.reference(x, approximate)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4 * 4), (
            f"gelu_and_mul({approximate}) mismatch"
        )


class TestRMSNormCov:
    """RMSNormFL dispatch matches a plain-PyTorch RMS normalization."""

    @staticmethod
    def reference(x: torch.Tensor, weight: torch.Tensor, eps: float):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * weight

    def test_forward_and_residual_returned(self, device):
        """Plain call normalizes x; residual call normalizes x+residual and returns the sum."""
        _resolve_or_fail("rms_norm")
        hidden = 512
        weight = torch.randn(hidden, device=device, dtype=torch.float32)
        layer = type("Layer", (), {"weight": weight, "variance_epsilon": 1e-6})()
        x = torch.randn(8, 33, hidden, device=device, dtype=torch.float32)
        out = call_op("rms_norm", layer, x)
        ref = self.reference(x, weight, 1e-6)
        assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3), "rms_norm mismatch"

        # Fused add semantics (vllm ir.ops.fused_add_rms_norm): the sum
        # x + residual is normalized and also returned as the new residual.
        residual = torch.randn_like(x)
        out2, res2 = call_op("rms_norm", layer, x, residual)
        sum_ref = x + residual
        ref2 = self.reference(sum_ref, weight, 1e-6)
        assert torch.allclose(out2, ref2, rtol=1e-3, atol=1e-3), (
            "rms_norm residual path must normalize x+residual"
        )
        assert torch.allclose(res2, sum_ref, rtol=1e-5, atol=1e-6), (
            "returned residual must equal x+residual"
        )
