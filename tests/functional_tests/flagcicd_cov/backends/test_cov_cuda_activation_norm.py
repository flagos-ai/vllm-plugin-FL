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

"""Single-GPU numerical coverage for vendor/cuda/impl activation,
normalization and MLA operator discovery modules.

Every kernel output is compared against an independent PyTorch reference
computed in the test. Selection is asserted through the manager's selected
impl id (forced to vendor.cuda via an explicit per-op policy) rather than
assumed, because FlagGems dispatch is enabled by default on this image.
"""

import pytest
import torch
import torch.nn.functional as F

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from vllm_fl.dispatch import get_default_manager
from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

pytestmark = [pytest.mark.gpu]


def _force_vendor_cuda(op_name):
    """Context manager selecting vendor.cuda for one op."""
    return policy_context(
        SelectionPolicy.from_dict(per_op_order={op_name: ["vendor:cuda"]})
    )


def _namespace(**attrs):
    return type("Layer", (), attrs)()


class TestSiluAndMulCuda:
    """silu_and_mul_cuda matches an independent elementwise reference."""

    @staticmethod
    def reference(x):
        half = x.shape[-1] // 2
        return F.silu(x[..., :half]) * x[..., half:]

    def test_selected_impl_is_cuda_delegate(self):
        """silu_and_mul resolves to vendor.cuda under the per-op policy."""
        with _force_vendor_cuda("silu_and_mul"):
            assert (
                get_default_manager().get_selected_impl_id("silu_and_mul")
                == "vendor.cuda"
            )

    @pytest.mark.parametrize("tokens,hidden", [(1, 128), (8, 512), (64, 1024)])
    def test_fp32_matches_reference(self, device, tokens, hidden):
        """fp32 silu(x1)*x2 matches the reference elementwise."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.activation import (
            silu_and_mul_cuda,
        )

        x = torch.randn(tokens, hidden * 2, device=device, dtype=torch.float32)
        out = silu_and_mul_cuda(None, x)
        ref = self.reference(x)
        assert out.shape == (tokens, hidden)
        assert torch.allclose(out, ref, rtol=1e-4, atol=4e-4)

    def test_bfloat16_matches_fp32_reference(self, device):
        """bf16 silu_and_mul stays close to the fp32 reference."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.activation import (
            silu_and_mul_cuda,
        )

        x = torch.randn(16, 512, device=device, dtype=torch.bfloat16)
        out = silu_and_mul_cuda(None, x)
        ref = self.reference(x.float())
        assert torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-2)

    def test_three_dimensional_input(self, device):
        """A 3D input reduces the last dim as the gate/up pair."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.activation import (
            silu_and_mul_cuda,
        )

        x = torch.randn(2, 3, 256, device=device, dtype=torch.float32)
        out = silu_and_mul_cuda(None, x)
        ref = self.reference(x)
        assert out.shape == (2, 3, 128)
        assert torch.allclose(out, ref, rtol=1e-4, atol=4e-4)


class TestGeluAndMulCuda:
    """gelu_and_mul_cuda honors the approximate attribute of the layer."""

    @staticmethod
    def reference(x, approximate):
        half = x.shape[-1] // 2
        return F.gelu(x[..., :half], approximate=approximate) * x[..., half:]

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_approximations_match_reference(self, device, approximate):
        """Both gelu approximations match the reference."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.activation import (
            gelu_and_mul_cuda,
        )

        x = torch.randn(16, 512, device=device, dtype=torch.float32)
        obj = _namespace(approximate=approximate)
        out = gelu_and_mul_cuda(obj, x)
        ref = self.reference(x, approximate)
        assert out.shape == (16, 256)
        assert torch.allclose(out, ref, rtol=1e-4, atol=4e-4), (
            f"gelu_and_mul(approximate={approximate!r}) mismatch"
        )

    def test_none_obj_defaults_to_none_approximation(self, device):
        """obj=None resolves to the default 'none' gelu (no attribute read)."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.activation import (
            gelu_and_mul_cuda,
        )

        x = torch.randn(8, 256, device=device, dtype=torch.float32)
        out = gelu_and_mul_cuda(None, x)
        ref = self.reference(x, "none")
        assert torch.allclose(out, ref, rtol=1e-4, atol=4e-4)

    def test_obj_without_attribute_defaults_to_none(self, device):
        """An object lacking `approximate` falls back to the default."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.activation import (
            gelu_and_mul_cuda,
        )

        x = torch.randn(8, 256, device=device, dtype=torch.float32)
        out = gelu_and_mul_cuda(_namespace(), x)
        ref = self.reference(x, "none")
        assert torch.allclose(out, ref, rtol=1e-4, atol=4e-4)

    def test_bfloat16_tanh_matches_fp32_reference(self, device):
        """bf16 gelu_tanh stays close to the fp32 reference."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.activation import (
            gelu_and_mul_cuda,
        )

        x = torch.randn(16, 512, device=device, dtype=torch.bfloat16)
        obj = _namespace(approximate="tanh")
        out = gelu_and_mul_cuda(obj, x)
        ref = self.reference(x.float(), "tanh")
        assert torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-2)


class TestRmsNormCuda:
    """rms_norm_cuda plain and residual paths match plain-PyTorch RMS norm."""

    @staticmethod
    def reference(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * weight

    def _make_layer(self, device, hidden, dtype):
        weight = (1 + 0.05 * torch.randn(hidden, device=device)).to(dtype)
        return _namespace(weight=weight, variance_epsilon=1e-6)

    def test_fp32_matches_reference(self, device):
        """fp32 rms_norm matches the plain PyTorch reference."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.normalization import (
            rms_norm_cuda,
        )

        hidden = 512
        layer = self._make_layer(device, hidden, torch.float32)
        x = torch.randn(8, 33, hidden, device=device, dtype=torch.float32)
        out = rms_norm_cuda(layer, x)
        ref = self.reference(x, layer.weight, 1e-6)
        assert torch.allclose(out, ref, rtol=1e-4, atol=4e-4)

    def test_bfloat16_matches_fp32_reference(self, device):
        """bf16 rms_norm stays close to the fp32 reference."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.normalization import (
            rms_norm_cuda,
        )

        hidden = 512
        layer = self._make_layer(device, hidden, torch.bfloat16)
        x = torch.randn(8, 17, hidden, device=device, dtype=torch.bfloat16)
        out = rms_norm_cuda(layer, x)
        ref = self.reference(x.float(), layer.weight.float(), 1e-6)
        assert torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-2)

    def test_residual_path_mutates_inputs_in_place(self, device):
        """The residual path fuses the add: x and residual update in place."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.normalization import (
            rms_norm_cuda,
        )

        hidden = 512
        layer = self._make_layer(device, hidden, torch.float32)
        x = torch.randn(4, 9, hidden, device=device, dtype=torch.float32)
        residual = torch.randn_like(x)
        x_in, res_in = x.clone(), residual.clone()
        out, out_residual = rms_norm_cuda(layer, x_in, res_in)
        total = x + residual
        ref = self.reference(total, layer.weight, 1e-6)
        assert out is x_in and out_residual is res_in
        assert torch.allclose(out, ref, rtol=1e-4, atol=4e-4)
        assert torch.allclose(out_residual, total, rtol=1e-5, atol=1e-6)

    def test_bfloat16_residual_path(self, device):
        """The bf16 residual path fuses the add and normalizes."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.normalization import (
            rms_norm_cuda,
        )

        hidden = 256
        layer = self._make_layer(device, hidden, torch.bfloat16)
        x = torch.randn(4, 5, hidden, device=device, dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        out, out_residual = rms_norm_cuda(layer, x, residual)
        total = x.float() + residual.float()
        ref = self.reference(total, layer.weight.float(), 1e-6)
        assert torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-2)
        assert torch.allclose(out_residual.float(), total, rtol=1e-2, atol=1e-2)


class TestMlaOperatorDiscovery:
    """get_operator/is_available mapping for native MLA entry points."""

    def test_unknown_name_raises_key_error(self):
        """An unknown MLA operator name raises KeyError."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.mla import get_operator

        raised = False
        try:
            get_operator("definitely_not_an_mla_operator")
        except KeyError:
            raised = True
        assert raised, "unknown MLA operator name must raise KeyError"

    def test_known_names_resolve_to_callables(self):
        """Known MLA operator names resolve to callables."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.mla import get_operator

        for name in (
            "concat_and_cache_mla",
            "concat_mla_q",
            "per_token_group_quant_fp8",
            "mla_prefill_backend",
        ):
            op = get_operator(name)
            assert op is not None, f"{name} resolved to None"
            assert callable(op), f"{name} resolved to a non-callable"

    def test_is_available_false_for_non_callable(self):
        """is_available is False for a non-callable target."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.mla import is_available

        assert is_available("concat_and_cache_mla", None) is False

    def test_is_available_true_for_kernel_backed_cache_op(self):
        """is_available is True for the cache concat op."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.mla import (
            get_operator,
            is_available,
        )

        op = get_operator("concat_and_cache_mla")
        assert op is not None, "concat_and_cache_mla missing from _custom_ops"
        assert is_available("concat_and_cache_mla", op) is True
