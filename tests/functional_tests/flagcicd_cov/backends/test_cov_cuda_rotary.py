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

"""Single-GPU coverage for vendor/cuda/impl/rotary.py.

apply_rotary_emb_cuda delegates to the layer object's forward_cuda and is
checked against an independent PyTorch rotation reference for both Neox and
interleaved styles and partial rotary dimensions.

# vllm_fl/dispatch/backends/vendor/cuda/impl/rotary.py:49
# rotary_embedding_cuda forwards (position_ids, query, key, cos, sin,
# rotary_interleaved) to vLLM's rotary_embedding, whose real signature is
# (positions, query, key, head_size: int, cos_sin_cache, is_neox). The `cos`
# tensor binds to the int head_size parameter, so any real invocation
# mismatches the schema. Per the source-bug protocol the numerical case for
# rotary_embedding_cuda is removed; the delegation path is covered through
# apply_rotary_emb_cuda below.
"""

import pytest
import torch

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from vllm_fl.dispatch import get_default_manager
from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

pytestmark = [pytest.mark.gpu]


def _make_apply_rotary_emb(is_neox_style=True, enable_fp32_compute=False):
    """Build an ApplyRotaryEmb without a running vLLM config.

    The CustomOp constructor needs engine state; the tests exercise only the
    backend method, matching the construction pattern of the unit suite.
    """
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    op = ApplyRotaryEmb.__new__(ApplyRotaryEmb)
    torch.nn.Module.__init__(op)
    op.is_neox_style = is_neox_style
    op.enable_fp32_compute = enable_fp32_compute
    return op


def _reference_rotate(x, cos, sin, is_neox_style):
    """Independent rotation of the full head_size (cos/sin already sized)."""
    origin_dtype = x.dtype
    x32 = x.float()
    cos = cos.unsqueeze(-2).to(torch.float32)
    sin = sin.unsqueeze(-2).to(torch.float32)
    if is_neox_style:
        x1, x2 = torch.chunk(x32, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        out = torch.cat((o1, o2), dim=-1)
    else:
        x1 = x32[..., ::2]
        x2 = x32[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        out = torch.stack((o1, o2), dim=-1).flatten(-2)
    return out.to(origin_dtype)


class TestApplyRotaryEmbCuda:
    """apply_rotary_emb_cuda delegates to forward_cuda and stays correct."""

    def test_selected_impl_is_cuda(self):
        """apply_rotary_emb resolves to vendor.cuda under the policy."""
        with policy_context(
            SelectionPolicy.from_dict(per_op_order={"apply_rotary_emb": ["vendor:cuda"]})
        ):
            assert (
                get_default_manager().get_selected_impl_id("apply_rotary_emb")
                == "vendor.cuda"
            )

    @pytest.mark.parametrize("is_neox_style", [True, False])
    def test_matches_reference_full_dim(self, device, is_neox_style):
        """Full-dim rotation matches the reference for both styles."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.rotary import (
            apply_rotary_emb_cuda,
        )

        torch.manual_seed(3)
        num_tokens, num_heads, head_size = 5, 2, 80
        op = _make_apply_rotary_emb(is_neox_style=is_neox_style)
        x = torch.randn(
            num_tokens, num_heads, head_size, device=device, dtype=torch.float32
        )
        angles = torch.randn(num_tokens, head_size // 2, device=device)
        cos, sin = angles.cos(), angles.sin()

        out = apply_rotary_emb_cuda(op, x, cos, sin)
        ref = _reference_rotate(x, cos, sin, is_neox_style)
        assert out.shape == x.shape
        assert torch.allclose(out, ref, rtol=1e-4, atol=4e-4)

    def test_fp16_matches_reference(self, device):
        """fp16 rotation stays close to the fp32 reference."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.rotary import (
            apply_rotary_emb_cuda,
        )

        torch.manual_seed(4)
        op = _make_apply_rotary_emb(is_neox_style=True)
        x = torch.randn(4, 3, 64, device=device, dtype=torch.float16)
        angles = torch.randn(4, 32, device=device)
        cos, sin = angles.cos(), angles.sin()
        out = apply_rotary_emb_cuda(op, x, cos, sin)
        ref = _reference_rotate(x.float(), cos, sin, True)
        assert torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-2)

    def test_forward_cuda_partial_rotary_dimension(self, device):
        """vLLM's CUDA path supports partial rotary via the layer object."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.rotary import (
            apply_rotary_emb_cuda,
        )

        torch.manual_seed(6)
        head_size, rotary_dim = 80, 78
        op = _make_apply_rotary_emb(is_neox_style=True)
        x = torch.randn(2, 2, head_size, device=device, dtype=torch.float32)
        angles = torch.randn(2, rotary_dim // 2, device=device)
        cos, sin = angles.cos(), angles.sin()

        out = apply_rotary_emb_cuda(op, x, cos, sin)
        # The vllm_flash_attn path rotates only the first rotary_dim elements
        # of each head and copies the tail unchanged.
        rotated = _reference_rotate(x[..., :rotary_dim], cos, sin, True)
        expected = torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
        assert torch.allclose(out, expected, rtol=1e-4, atol=4e-4)
        assert torch.equal(out[..., rotary_dim:], x[..., rotary_dim:])
