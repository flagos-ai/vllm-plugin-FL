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

"""Single-GPU tests for vllm_fl/models/glm5/vision_attention.py (M1).

``_packed_math_sdpa`` composes existing torch math ops into a varlen
attention. Every test compares against an independent per-sequence
scaled-dot-product reference computed with plain torch ops in the test —
no shared implementation is reused.
"""

import math

import pytest
import torch

from . import _import_shim  # noqa: F401  (idempotent; no-op on CI)
from vllm_fl.models.glm5.vision_attention import _packed_math_sdpa

pytestmark = [pytest.mark.gpu]


def _reference(q, k, v, cu_seqlens, scale):
    """Independent per-sequence attention in float32."""
    ref = torch.empty_like(q)
    for s in range(cu_seqlens.numel() - 1):
        lo, hi = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
        qs = q[lo:hi].transpose(0, 1).float()
        ks = k[lo:hi].transpose(0, 1).float()
        vs = v[lo:hi].transpose(0, 1).float()
        scores = (qs @ ks.transpose(1, 2)) * scale
        ref[lo:hi] = (scores.softmax(dim=-1) @ vs).transpose(0, 1)
    return ref


def _run(q, k, v, cu_seqlens, scale, **extra):
    kwargs = {
        "dropout_p": 0,
        "causal": False,
        "cu_seqlens_q": cu_seqlens,
        "cu_seqlens_k": cu_seqlens,
        "softmax_scale": scale,
    }
    kwargs.update(extra)
    return _packed_math_sdpa(q, k, v, **kwargs)


class TestPackedMathSdpa:
    """Varlen attention matches the per-sequence reference."""

    @pytest.mark.parametrize(
        ("lengths", "dtype"),
        [
            ([5, 9, 10], torch.float32),
            ([1, 1, 1, 1], torch.float32),
            ([16], torch.float32),
            ([3, 29, 4, 33], torch.bfloat16),
        ],
    )
    def test_matches_per_sequence_reference(self, device, lengths, dtype):
        """Sequence sets spanning the internal 32-row query tiling."""
        torch.manual_seed(9)
        total = sum(lengths)
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        )
        heads, dim = 4, 16
        q = torch.randn(total, heads, dim, device=device, dtype=dtype)
        k = torch.randn(total, heads, dim, device=device, dtype=dtype)
        v = torch.randn(total, heads, dim, device=device, dtype=dtype)
        scale = 1.0 / math.sqrt(dim)
        out = _run(q, k, v, cu_seqlens, scale)
        assert out.shape == q.shape
        assert out.dtype == q.dtype
        ref = _reference(q, k, v, cu_seqlens.cpu(), scale)
        # bf16 path materializes float32 intermediates, so tolerance is
        # driven by the bf16 inputs (rtol 1e-2 per the dtype contract).
        rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)
        assert torch.allclose(out.float(), ref.float(), rtol=rtol, atol=atol)

    def test_output_is_not_aliased_to_inputs(self, device):
        """The result buffer is freshly allocated per call."""
        torch.manual_seed(4)
        cu_seqlens = torch.tensor([0, 6, 11], dtype=torch.int32, device=device)
        q = torch.randn(11, 2, 8, device=device)
        k = torch.randn(11, 2, 8, device=device)
        v = torch.randn(11, 2, 8, device=device)
        out = _run(q, k, v, cu_seqlens, 0.25)
        assert out.data_ptr() != q.data_ptr()
        assert out.data_ptr() != v.data_ptr()
        snapshot = out.clone()
        q += 1.0
        assert torch.equal(out, snapshot)

    def test_dropout_nonzero_rejected(self, device):
        """vllm_fl/models/glm5/vision_attention.py:21 guards the contract."""
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
        q = torch.randn(8, 2, 8, device=device)
        k = torch.randn(8, 2, 8, device=device)
        v = torch.randn(8, 2, 8, device=device)
        with pytest.raises(AssertionError):
            _run(q, k, v, cu_seqlens, 0.25, dropout_p=0.1)

    def test_causal_flag_rejected(self, device):
        """Causal flag rejected."""
        cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
        q = torch.randn(8, 2, 8, device=device)
        k = torch.randn(8, 2, 8, device=device)
        v = torch.randn(8, 2, 8, device=device)
        with pytest.raises(AssertionError):
            _run(q, k, v, cu_seqlens, 0.25, causal=True)

    def test_mismatched_cu_seqlens_rejected(self, device):
        """q and k boundary tensors must be the same object."""
        cu_a = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
        cu_b = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
        q = torch.randn(8, 2, 8, device=device)
        k = torch.randn(8, 2, 8, device=device)
        v = torch.randn(8, 2, 8, device=device)
        with pytest.raises(AssertionError):
            _packed_math_sdpa(
                q,
                k,
                v,
                dropout_p=0,
                causal=False,
                cu_seqlens_q=cu_a,
                cu_seqlens_k=cu_b,
                softmax_scale=0.25,
            )
