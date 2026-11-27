# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU numeric tests for reference backend impl helpers.

These tests cover ``vllm_fl/dispatch/backends/reference/impl/*`` directly:
rotary embedding (both layouts plus every validation error), the RMS norm
bf16 round-trip, and activation ``approximate`` handling.  All expectations
are computed with independent oracles that spell the math out differently
from the implementation.

The basic/residual/variance-size/weight-flag RMS norm paths and the 4D
rotary layouts are covered by ``tests/unit_tests/ops/test_numerical.py``
and are deliberately not duplicated here.
"""

import math

import pytest
import torch

from vllm_fl.dispatch.backends.reference.impl.activation import (
    gelu_and_mul_torch,
    silu_and_mul_torch,
)
from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch
from vllm_fl.dispatch.backends.reference.impl.rotary import rotary_embedding_torch


class _NormObj:
    """Minimal RMSNorm layer double with the attributes the impl reads."""

    def __init__(self, weight, variance_epsilon=1e-6, **overrides):
        self.weight = weight
        self.variance_epsilon = variance_epsilon
        self.pass_weight = True
        self.pass_weight_add = True
        for key, value in overrides.items():
            setattr(self, key, value)


class _RotaryObj:
    """Placeholder obj — rotary impl reads no attributes from it."""

    pass


def _rotary_cache(head_dim, num_positions, base=10000.0):
    """Build half-width cos/sin caches the way vLLM layers build them."""
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float32) * 2 / head_dim)
    )
    positions = torch.arange(num_positions, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [pos, half]
    return freqs.cos(), freqs.sin()


def _neox_oracle(x, cos_sel, sin_sel):
    """NeoX rotation: rotate_half pairs (i, i+d/2) against cat-duplicated cos."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat((-x2, x1), dim=-1)
    c = torch.cat([cos_sel, cos_sel], dim=-1)
    s = torch.cat([sin_sel, sin_sel], dim=-1)
    return x * c + rot * s


def _interleaved_oracle(x, cos_sel, sin_sel):
    """Interleaved rotation: (even, odd) neighbour pairs, repeated cos."""
    c = cos_sel.repeat_interleave(2, dim=-1)
    s = sin_sel.repeat_interleave(2, dim=-1)
    even, odd = x[..., ::2], x[..., 1::2]
    rot = torch.stack((-odd, even), dim=-1).flatten(-2)
    return x * c + rot * s


class TestRotaryEmbedding:
    """rotary_embedding_torch numeric contract for NeoX and interleaved."""

    @pytest.mark.parametrize("interleaved", [False, True])
    def test_3d_inputs_match_style_oracle(self, interleaved):
        """3D [seq, heads, dim] inputs match the layout-matching oracle."""
        seq, heads, head_dim = 4, 2, 8
        torch.manual_seed(0)
        q = torch.randn(seq, heads, head_dim)
        k = torch.randn(seq, heads, head_dim)
        pos = torch.arange(seq)
        cos, sin = _rotary_cache(head_dim, seq)

        q_out, k_out = rotary_embedding_torch(
            _RotaryObj(), q, k, cos, sin, pos, rotary_interleaved=interleaved
        )
        # The impl inserts a heads axis; mirror that in the oracle inputs.
        cos_sel = cos[pos].unsqueeze(1)
        sin_sel = sin[pos].unsqueeze(1)
        oracle = _interleaved_oracle if interleaved else _neox_oracle
        torch.testing.assert_close(
            q_out, oracle(q, cos_sel, sin_sel), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            k_out, oracle(k, cos_sel, sin_sel), rtol=1e-5, atol=1e-6
        )


    def test_full_width_cache_needs_no_expansion(self):
        """A cache already head_dim wide is used as-is (no cat/repeat)."""
        seq, heads, head_dim = 3, 1, 8
        torch.manual_seed(3)
        q = torch.randn(seq, heads, head_dim)
        k = torch.randn(seq, heads, head_dim)
        pos = torch.arange(seq)
        cos_half, sin_half = _rotary_cache(head_dim, seq)
        cos = torch.cat([cos_half, cos_half], dim=-1)
        sin = torch.cat([sin_half, sin_half], dim=-1)

        q_out, _ = rotary_embedding_torch(_RotaryObj(), q, k, cos, sin, pos)
        cos_sel = cos[pos].unsqueeze(1)
        sin_sel = sin[pos].unsqueeze(1)
        # Full-width cache: use it directly (no cat) with rotate_half.
        d = q.shape[-1]
        q1, q2 = q[..., : d // 2], q[..., d // 2 :]
        rot = torch.cat((-q2, q1), dim=-1)
        expected = q * cos_sel + rot * sin_sel
        torch.testing.assert_close(q_out, expected, rtol=1e-5, atol=1e-6)

    def test_interleaved_uses_adjacent_coordinate_pairs(self):
        """Hand-computed check: interleaved rotation swaps (even, odd) pairs."""
        # head_dim=2 with a width-1 cache: repeat_interleave gives c=[1,1].
        cos = torch.tensor([[1.0]])
        sin = torch.tensor([[0.0]])
        q = torch.tensor([[[1.0, 2.0]]])  # [seq=1, heads=1, dim=2]
        k = torch.tensor([[[3.0, 4.0]]])
        pos = torch.tensor([0])

        q_out, _ = rotary_embedding_torch(
            _RotaryObj(), q, k, cos, sin, pos, rotary_interleaved=True
        )
        torch.testing.assert_close(q_out, q, rtol=1e-6, atol=1e-7)

        # c=[0,0], s=[1,1]: result = rotate_interleaved(q) = [-b, a] = [-2, 1].
        cos0 = torch.tensor([[0.0]])
        sin1 = torch.tensor([[1.0]])
        q_out2, _ = rotary_embedding_torch(
            _RotaryObj(), q, k, cos0, sin1, pos, rotary_interleaved=True
        )
        torch.testing.assert_close(
            q_out2, torch.tensor([[[-2.0, 1.0]]]), rtol=1e-6, atol=1e-7
        )

    def test_neox_uses_half_offset_pairs(self):
        """Hand-computed check: NeoX rotation pairs (i, i+d/2)."""
        # head_dim=2, width-1 cache duplicated to [c, c] by the impl.
        cos = torch.tensor([[0.0]])
        sin = torch.tensor([[1.0]])
        q = torch.tensor([[[3.0, 4.0]]])  # [seq=1, heads=1, dim=2]
        k = torch.tensor([[[5.0, 6.0]]])
        pos = torch.tensor([0])

        q_out, _ = rotary_embedding_torch(_RotaryObj(), q, k, cos, sin, pos)
        # c=[0,0], s=[1,1]; rotate_half([a, b]) = [-b, a] → result = [-4, 3].
        torch.testing.assert_close(
            q_out, torch.tensor([[[-4.0, 3.0]]]), rtol=1e-6, atol=1e-7
        )

    def test_cos_sin_promoted_to_query_dtype(self):
        """Float64 cos/sin still produce float32 outputs for float32 inputs."""
        seq, head_dim = 2, 4
        q = torch.randn(seq, 1, head_dim)
        k = torch.randn(seq, 1, head_dim)
        pos = torch.arange(seq)
        cos, sin = _rotary_cache(head_dim, seq)
        q_out, k_out = rotary_embedding_torch(
            _RotaryObj(), q, k, cos.double(), sin.double(), pos
        )
        assert q_out.dtype == torch.float32 and k_out.dtype == torch.float32

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (
                lambda q, k, c, s, p: (q.double(), k, c, s, p),
                "query and key must have the same device and dtype",
            ),
            (
                lambda q, k, c, s, p: (q.unsqueeze(0), k, c, s, p),
                "query and key must both be 3D or both be 4D",
            ),
            (
                lambda q, k, c, s, p: (q, k[..., :4], c, s, p),
                "query and key must have the same head dimension",
            ),
            (
                lambda q, k, c, s, p: (q[..., :3], k[..., :3], c, s, p),
                "the rotary head dimension must be a positive even number",
            ),
            (
                lambda q, k, c, s, p: (q, k, c, s, p.float()),
                "position_ids must have dtype int32 or int64",
            ),
            (
                lambda q, k, c, s, p: (q, k, c, s, p.unsqueeze(0)),
                "3D query and key require 1D position_ids",
            ),
            (
                lambda q, k, c, s, p: (q[:1], k[:1], c, s, p),
                "query, key, and position_ids must have equal token counts",
            ),
            (
                lambda q, k, c, s, p: (q, k, c.unsqueeze(0), s, p),
                "cos and sin must be same-shaped 2D caches",
            ),
            (
                lambda q, k, c, s, p: (q, k, c, s[:2], p),
                "cos and sin must be same-shaped 2D caches",
            ),
            (
                lambda q, k, c, s, p: (q, k, c[:, :2], s[:, :2], p),
                "cos and sin width must equal the head dimension or its half",
            ),
        ],
    )
    def test_validation_errors(self, mutate, message):
        """Malformed inputs raise ValueError with the documented message."""
        seq, head_dim = 3, 8
        torch.manual_seed(4)
        q = torch.randn(seq, 2, head_dim)
        k = torch.randn(seq, 2, head_dim)
        pos = torch.arange(seq)
        cos, sin = _rotary_cache(head_dim, seq)
        q2, k2, c2, s2, p2 = mutate(q, k, cos, sin, pos)
        with pytest.raises(ValueError, match=message):
            rotary_embedding_torch(_RotaryObj(), q2, k2, c2, s2, p2)

    def test_4d_bad_position_shape_raises(self):
        """4D inputs with mis-shaped positions are rejected."""
        q = torch.randn(2, 1, 3, 8)
        k = torch.randn(2, 1, 3, 8)
        pos = torch.zeros(5)  # neither [seq] nor [batch, seq]
        cos, sin = _rotary_cache(8, 5)
        with pytest.raises(ValueError, match="position_ids"):
            rotary_embedding_torch(_RotaryObj(), q, k, cos, sin, pos)

    def test_4d_mismatched_batch_raises(self):
        """4D query/key with different batch sizes are rejected."""
        q = torch.randn(2, 1, 3, 8)
        k = torch.randn(3, 1, 3, 8)
        pos = torch.arange(3)
        cos, sin = _rotary_cache(8, 3)
        with pytest.raises(ValueError, match="batch/sequence"):
            rotary_embedding_torch(_RotaryObj(), q, k, cos, sin, pos)


class TestRmsNorm:
    """rms_norm_torch flag combinations with independent oracles."""


    def test_bfloat16_roundtrip(self):
        """bf16 inputs return bf16 outputs computed via float32 internally."""
        x = torch.randn(3, 8, dtype=torch.bfloat16)
        weight = torch.randn(8, dtype=torch.bfloat16)
        obj = _NormObj(weight)

        out = rms_norm_torch(obj, x)
        assert out.dtype == torch.bfloat16
        # Compare against the same math in float32 with a bf16-appropriate tol.
        x32, w32 = x.float(), weight.float()
        expected = x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + 1e-6) * w32
        torch.testing.assert_close(
            out.float(), expected.to(torch.bfloat16).float(), rtol=2e-2, atol=1e-2
        )


class TestActivations:
    """silu_and_mul_torch / gelu_and_mul_torch contracts."""

    def test_silu_and_mul_split_and_multiply(self):
        """First half passes SiLU, second half multiplies element-wise."""
        x = torch.tensor([[1.0, 2.0, -1.0, 0.5]])
        out = silu_and_mul_torch(None, x)
        silu_1 = 1.0 / (1.0 + math.exp(-1.0))
        silu_2 = 2.0 / (1.0 + math.exp(-2.0))
        torch.testing.assert_close(
            out, torch.tensor([[silu_1 * -1.0, silu_2 * 0.5]]), rtol=1e-5, atol=1e-6
        )

    def test_gelu_approximate_none_and_tanh(self):
        """approximate is read from obj; none/tanh produce different values."""
        # head_dim=4: first half [0.7, 1.3] activates, second half multiplies.
        x = torch.tensor([[0.7, 1.3, -0.4, 2.0]])
        gelu_exact = 0.5 * 0.7 * (1.0 + math.erf(0.7 / math.sqrt(2.0)))
        gelu_tanh = (
            0.5
            * 0.7
            * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (0.7 + 0.044715 * 0.7**3)))
        )
        gelu_exact_2 = 0.5 * 1.3 * (1.0 + math.erf(1.3 / math.sqrt(2.0)))
        gelu_tanh_2 = (
            0.5
            * 1.3
            * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (1.3 + 0.044715 * 1.3**3)))
        )

        class Obj:
            approximate = "none"

        out_none = gelu_and_mul_torch(Obj(), x)

        class ObjTanh:
            approximate = "tanh"

        out_tanh = gelu_and_mul_torch(ObjTanh(), x)
        torch.testing.assert_close(
            out_none,
            torch.tensor([[gelu_exact * -0.4, gelu_exact_2 * 2.0]]),
            rtol=1e-4,
            atol=1e-6,
        )
        torch.testing.assert_close(
            out_tanh,
            torch.tensor([[gelu_tanh * -0.4, gelu_tanh_2 * 2.0]]),
            rtol=1e-4,
            atol=1e-6,
        )
        # The two approximations genuinely differ at this point.
        assert not torch.equal(out_none, out_tanh)

    def test_gelu_default_approximate_when_obj_lacks_attr(self):
        """An obj without `approximate` (or None) defaults to 'none'."""
        x = torch.tensor([[0.5, 1.0, 0.25, -0.75]])
        out = gelu_and_mul_torch(None, x)
        gelu_1 = 0.5 * 0.5 * (1.0 + math.erf(0.5 / math.sqrt(2.0)))
        gelu_2 = 0.5 * 1.0 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
        torch.testing.assert_close(
            out, torch.tensor([[gelu_1 * 0.25, gelu_2 * -0.75]]), rtol=1e-4, atol=1e-6
        )

        class Bare:
            pass

        out_bare = gelu_and_mul_torch(Bare(), x)
        torch.testing.assert_close(out, out_bare)

    def test_activations_handle_3d_inputs(self):
        """Both activations work on arbitrary leading dimensions."""
        x = torch.randn(2, 3, 10)
        assert silu_and_mul_torch(None, x).shape == (2, 3, 5)
        assert gelu_and_mul_torch(None, x).shape == (2, 3, 5)
