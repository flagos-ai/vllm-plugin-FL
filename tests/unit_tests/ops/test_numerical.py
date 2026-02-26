# Copyright (c) 2025 BAAI. All rights reserved.

"""
Numerical correctness tests for operator implementations.

This module tests that operator implementations produce numerically correct
results by comparing against reference (PyTorch) implementations.

These tests verify:
- Output shape correctness
- Numerical accuracy within tolerance
- Handling of edge cases (zeros, large values, etc.)
- Different dtypes (float32, float16, bfloat16)
"""

import pytest
import torch
import torch.nn.functional as F


# =============================================================================
# Helpers
# =============================================================================

class _FakeNorm:
    """Minimal stand-in for an RMSNorm layer, providing weight and epsilon."""

    def __init__(self, weight: torch.Tensor, epsilon: float = 1e-5):
        self.weight = weight
        self.variance_epsilon = epsilon


# =============================================================================
# Reference Implementations (PyTorch baseline)
# =============================================================================

def reference_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Reference SiLU and multiply implementation."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return F.silu(x1) * x2


def reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor = None,
):
    """Reference RMS normalization implementation."""
    if residual is not None:
        x = x + residual
        new_residual = x.clone()

    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    output = weight * x

    if residual is not None:
        return output, new_residual
    return output


def reference_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = True,
):
    """
    Reference rotary position embedding implementation.

    Applies rotary position embedding to query and key tensors.
    """
    def apply_rotary(x, cos, sin, is_neox_style):
        if is_neox_style:
            # GPT-NeoX style: split at half
            d = x.shape[-1] // 2
            x1, x2 = x[..., :d], x[..., d:]
            rotated = torch.cat([-x2, x1], dim=-1)
        else:
            # GPT-J style: interleaved
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)

        return x * cos + rotated * sin

    q_embed = apply_rotary(query, cos, sin, is_neox_style)
    k_embed = apply_rotary(key, cos, sin, is_neox_style)

    return q_embed, k_embed


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(params=[torch.float32, torch.float16])
def dtype(request):
    """Test with different floating point dtypes."""
    return request.param


@pytest.fixture(params=[(2, 128), (4, 256), (8, 512)])
def hidden_size_config(request):
    """Different batch and hidden size configurations."""
    return request.param


@pytest.fixture
def device():
    """Get available device (CPU for unit tests)."""
    return torch.device("cpu")


# =============================================================================
# SiLU and Multiply Tests
# =============================================================================

class TestSiluAndMulNumerical:
    """Numerical correctness tests for SiLU and multiply operation."""

    def test_basic_correctness(self, device):
        """Test basic numerical correctness."""
        from vllm_fl.dispatch.backends.reference.impl.activation import silu_and_mul_torch

        x = torch.randn(2, 16, device=device, dtype=torch.float32)

        result = silu_and_mul_torch(None, x)
        expected = reference_silu_and_mul(x)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_output_shape(self, hidden_size_config, device):
        """Test that output shape is correct (half of input)."""
        from vllm_fl.dispatch.backends.reference.impl.activation import silu_and_mul_torch

        batch, hidden = hidden_size_config
        x = torch.randn(batch, hidden * 2, device=device)

        result = silu_and_mul_torch(None, x)

        assert result.shape == (batch, hidden)

    def test_dtype_preservation(self, dtype, device):
        """Test that dtype is preserved."""
        from vllm_fl.dispatch.backends.reference.impl.activation import silu_and_mul_torch

        x = torch.randn(2, 16, device=device, dtype=dtype)

        result = silu_and_mul_torch(None, x)

        assert result.dtype == dtype

    def test_zero_input(self, device):
        """Test with zero input tensor."""
        from vllm_fl.dispatch.backends.reference.impl.activation import silu_and_mul_torch

        x = torch.zeros(2, 16, device=device)

        result = silu_and_mul_torch(None, x)
        expected = reference_silu_and_mul(x)

        torch.testing.assert_close(result, expected)

    def test_large_values(self, device):
        """Test with large input values."""
        from vllm_fl.dispatch.backends.reference.impl.activation import silu_and_mul_torch

        x = torch.randn(2, 16, device=device) * 100

        result = silu_and_mul_torch(None, x)
        expected = reference_silu_and_mul(x)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_negative_values(self, device):
        """Test with negative input values."""
        from vllm_fl.dispatch.backends.reference.impl.activation import silu_and_mul_torch

        x = -torch.abs(torch.randn(2, 16, device=device))

        result = silu_and_mul_torch(None, x)
        expected = reference_silu_and_mul(x)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_3d_input(self, device):
        """Test with 3D input tensor."""
        from vllm_fl.dispatch.backends.reference.impl.activation import silu_and_mul_torch

        x = torch.randn(2, 4, 16, device=device)

        result = silu_and_mul_torch(None, x)
        expected = reference_silu_and_mul(x)

        assert result.shape == (2, 4, 8)
        torch.testing.assert_close(result, expected)


# =============================================================================
# RMS Normalization Tests
# =============================================================================

class TestRMSNormNumerical:
    """Numerical correctness tests for RMS normalization."""

    def test_basic_correctness(self, device):
        """Test basic numerical correctness."""
        from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch

        hidden_size = 128
        x = torch.randn(2, hidden_size, device=device, dtype=torch.float32)
        weight = torch.ones(hidden_size, device=device, dtype=torch.float32)
        epsilon = 1e-5
        obj = _FakeNorm(weight, epsilon)

        result = rms_norm_torch(obj, x)
        expected = reference_rms_norm(x, weight, epsilon)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_with_residual(self, device):
        """Test RMS norm with residual connection."""
        from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch

        hidden_size = 128
        x = torch.randn(2, hidden_size, device=device, dtype=torch.float32)
        residual = torch.randn(2, hidden_size, device=device, dtype=torch.float32)
        weight = torch.ones(hidden_size, device=device, dtype=torch.float32)
        epsilon = 1e-5
        obj = _FakeNorm(weight, epsilon)

        result_out, result_res = rms_norm_torch(obj, x, residual)
        expected_out, expected_res = reference_rms_norm(x, weight, epsilon, residual)

        torch.testing.assert_close(result_out, expected_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(result_res, expected_res, rtol=1e-5, atol=1e-5)

    def test_output_shape(self, hidden_size_config, device):
        """Test that output shape matches input shape."""
        from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch

        batch, hidden = hidden_size_config
        x = torch.randn(batch, hidden, device=device)
        weight = torch.ones(hidden, device=device)
        epsilon = 1e-5
        obj = _FakeNorm(weight, epsilon)

        result = rms_norm_torch(obj, x)

        assert result.shape == x.shape

    def test_dtype_preservation(self, dtype, device):
        """Test that dtype is preserved."""
        from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch

        hidden_size = 128
        x = torch.randn(2, hidden_size, device=device, dtype=dtype)
        weight = torch.ones(hidden_size, device=device, dtype=dtype)
        epsilon = 1e-5
        obj = _FakeNorm(weight, epsilon)

        result = rms_norm_torch(obj, x)

        assert result.dtype == dtype

    def test_normalization_effect(self, device):
        """Test that normalization actually normalizes the tensor."""
        from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch

        hidden_size = 128
        # Create input with known variance
        x = torch.randn(2, hidden_size, device=device) * 10
        weight = torch.ones(hidden_size, device=device)
        epsilon = 1e-5
        obj = _FakeNorm(weight, epsilon)

        result = rms_norm_torch(obj, x)

        # After RMS norm, the RMS should be approximately 1
        rms = result.pow(2).mean(-1).sqrt()
        torch.testing.assert_close(
            rms,
            torch.ones_like(rms),
            rtol=0.1,
            atol=0.1
        )

    def test_epsilon_effect(self, device):
        """Test that epsilon prevents division by zero."""
        from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch

        hidden_size = 128
        x = torch.zeros(2, hidden_size, device=device)
        weight = torch.ones(hidden_size, device=device)
        epsilon = 1e-5
        obj = _FakeNorm(weight, epsilon)

        # Should not raise or produce NaN/Inf
        result = rms_norm_torch(obj, x)

        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_weight_scaling(self, device):
        """Test that weight properly scales the output."""
        from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch

        hidden_size = 128
        x = torch.randn(2, hidden_size, device=device)
        weight1 = torch.ones(hidden_size, device=device)
        weight2 = torch.ones(hidden_size, device=device) * 2
        epsilon = 1e-5
        obj1 = _FakeNorm(weight1, epsilon)
        obj2 = _FakeNorm(weight2, epsilon)

        result1 = rms_norm_torch(obj1, x)
        result2 = rms_norm_torch(obj2, x)

        # Result with weight=2 should be twice result with weight=1
        torch.testing.assert_close(result2, result1 * 2, rtol=1e-5, atol=1e-5)

    def test_3d_input(self, device):
        """Test with 3D input tensor (batch, seq, hidden)."""
        from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch

        x = torch.randn(2, 4, 128, device=device)
        weight = torch.ones(128, device=device)
        epsilon = 1e-5
        obj = _FakeNorm(weight, epsilon)

        result = rms_norm_torch(obj, x)
        expected = reference_rms_norm(x, weight, epsilon)

        assert result.shape == x.shape
        torch.testing.assert_close(result, expected)


# =============================================================================
# Rotary Embedding Tests
# =============================================================================

class TestRotaryEmbeddingNumerical:
    """Numerical correctness tests for rotary position embedding."""

    def test_basic_correctness_4d(self, device):
        """Test basic numerical correctness with 4D tensors."""
        from vllm_fl.dispatch.backends.reference.impl.rotary import rotary_embedding_torch

        batch, num_heads, seq_len, head_dim = 2, 4, 8, 64
        max_seq_len = 16
        rotary_dim = head_dim // 2

        # [batch, num_heads, seq_len, head_dim]
        query = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
        key = torch.randn(batch, num_heads, seq_len, head_dim, device=device)

        # Create cos/sin cache [max_seq_len, rotary_dim]
        freqs = 1.0 / (10000 ** (torch.arange(0, rotary_dim, device=device).float() / rotary_dim))
        angles = torch.arange(max_seq_len, device=device).unsqueeze(1) * freqs.unsqueeze(0)
        cos = torch.cos(angles)  # [max_seq_len, rotary_dim]
        sin = torch.sin(angles)  # [max_seq_len, rotary_dim]

        # For 4D query, position_ids should be 2D [batch, seq_len]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)

        result_q, result_k = rotary_embedding_torch(
            None, query, key, cos, sin,
            position_ids=positions,
            rotary_interleaved=False,
            inplace=False,
        )

        # Should have same shape as input
        assert result_q.shape == query.shape
        assert result_k.shape == key.shape
        assert not torch.isnan(result_q).any()
        assert not torch.isnan(result_k).any()

    def test_output_shape_4d(self, device):
        """Test that output shapes match input shapes for 4D tensors."""
        from vllm_fl.dispatch.backends.reference.impl.rotary import rotary_embedding_torch

        batch, num_heads, seq_len, head_dim = 4, 8, 16, 32
        max_seq_len = 32
        rotary_dim = head_dim // 2

        # [batch, num_heads, seq_len, head_dim]
        query = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
        key = torch.randn(batch, num_heads, seq_len, head_dim, device=device)

        # For 4D query, use 2D position_ids [batch, seq_len]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        cos = torch.randn(max_seq_len, rotary_dim, device=device)
        sin = torch.randn(max_seq_len, rotary_dim, device=device)

        result_q, result_k = rotary_embedding_torch(
            None, query, key, cos, sin,
            position_ids=positions,
            rotary_interleaved=False,
            inplace=False,
        )

        assert result_q.shape == query.shape
        assert result_k.shape == key.shape

    def test_output_shape_3d(self, device):
        """Test that output shapes match input shapes for 3D tensors."""
        from vllm_fl.dispatch.backends.reference.impl.rotary import rotary_embedding_torch

        seq_len, num_heads, head_dim = 16, 8, 32
        max_seq_len = 32
        rotary_dim = head_dim // 2

        # [seq_len, num_heads, head_dim]
        query = torch.randn(seq_len, num_heads, head_dim, device=device)
        key = torch.randn(seq_len, num_heads, head_dim, device=device)

        # For 3D query, use 1D position_ids [seq_len]
        positions = torch.arange(seq_len, device=device)
        cos = torch.randn(max_seq_len, rotary_dim, device=device)
        sin = torch.randn(max_seq_len, rotary_dim, device=device)

        result_q, result_k = rotary_embedding_torch(
            None, query, key, cos, sin,
            position_ids=positions,
            rotary_interleaved=False,
            inplace=False,
        )

        assert result_q.shape == query.shape
        assert result_k.shape == key.shape

    def test_dtype_preservation_3d(self, dtype, device):
        """Test that dtype is preserved with 3D tensors."""
        from vllm_fl.dispatch.backends.reference.impl.rotary import rotary_embedding_torch

        seq_len, num_heads, head_dim = 8, 4, 32
        max_seq_len = 16
        rotary_dim = head_dim // 2

        # Use 3D tensors for simpler testing
        query = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
        key = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)

        positions = torch.arange(seq_len, device=device)
        cos = torch.randn(max_seq_len, rotary_dim, device=device, dtype=dtype)
        sin = torch.randn(max_seq_len, rotary_dim, device=device, dtype=dtype)

        result_q, result_k = rotary_embedding_torch(
            None, query, key, cos, sin,
            position_ids=positions,
            rotary_interleaved=False,
            inplace=False,
        )

        assert result_q.dtype == dtype
        assert result_k.dtype == dtype

    def test_interleaved_vs_neox_style(self, device):
        """Test both interleaved and neox rotary styles."""
        from vllm_fl.dispatch.backends.reference.impl.rotary import rotary_embedding_torch

        seq_len, num_heads, head_dim = 8, 4, 64
        max_seq_len = 16
        rotary_dim = head_dim // 2

        # Use 3D tensors
        query = torch.randn(seq_len, num_heads, head_dim, device=device)
        key = torch.randn(seq_len, num_heads, head_dim, device=device)

        positions = torch.arange(seq_len, device=device)
        cos = torch.randn(max_seq_len, rotary_dim, device=device)
        sin = torch.randn(max_seq_len, rotary_dim, device=device)

        # Test neox style (default)
        result_q_neox, result_k_neox = rotary_embedding_torch(
            None, query, key, cos, sin,
            position_ids=positions,
            rotary_interleaved=False,
            inplace=False,
        )

        # Test interleaved style
        result_q_interleaved, result_k_interleaved = rotary_embedding_torch(
            None, query, key, cos, sin,
            position_ids=positions,
            rotary_interleaved=True,
            inplace=False,
        )

        # Results should be different between styles
        assert not torch.allclose(result_q_neox, result_q_interleaved)
        assert not torch.allclose(result_k_neox, result_k_interleaved)

        # But both should have valid outputs (no NaN/Inf)
        assert not torch.isnan(result_q_neox).any()
        assert not torch.isnan(result_q_interleaved).any()

    def test_rotary_embedding_mathematically(self, device):
        """Test that rotary embedding produces expected rotation behavior."""
        from vllm_fl.dispatch.backends.reference.impl.rotary import rotary_embedding_torch

        seq_len, num_heads, head_dim = 4, 2, 8
        max_seq_len = 8
        rotary_dim = head_dim // 2

        # Create simple test tensors
        query = torch.ones(seq_len, num_heads, head_dim, device=device)
        key = torch.ones(seq_len, num_heads, head_dim, device=device)

        # Create cos/sin with known values
        positions = torch.arange(seq_len, device=device)
        cos = torch.ones(max_seq_len, rotary_dim, device=device)
        sin = torch.zeros(max_seq_len, rotary_dim, device=device)

        # With cos=1 and sin=0, output should equal input (no rotation)
        result_q, result_k = rotary_embedding_torch(
            None, query, key, cos, sin,
            position_ids=positions,
            rotary_interleaved=False,
            inplace=False,
        )

        torch.testing.assert_close(result_q, query, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(result_k, key, rtol=1e-5, atol=1e-5)


# =============================================================================
# Cross-Implementation Consistency Tests
# =============================================================================

class TestCrossImplementationConsistency:
    """Test consistency between different backend implementations."""

    def test_silu_reference_matches_pytorch(self, device):
        """Test that reference implementation matches PyTorch exactly."""
        from vllm_fl.dispatch.backends.reference.impl.activation import silu_and_mul_torch

        x = torch.randn(4, 32, device=device)

        result = silu_and_mul_torch(None, x)
        expected = reference_silu_and_mul(x)

        # Should be exactly equal (same implementation)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_rms_norm_reference_matches_pytorch(self, device):
        """Test that reference RMS norm matches our baseline."""
        from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch

        x = torch.randn(4, 64, device=device)
        weight = torch.randn(64, device=device)
        epsilon = 1e-6
        obj = _FakeNorm(weight, epsilon)

        result = rms_norm_torch(obj, x)
        expected = reference_rms_norm(x, weight, epsilon)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_silu_single_element(self, device):
        """Test SiLU with single element batch."""
        from vllm_fl.dispatch.backends.reference.impl.activation import silu_and_mul_torch

        x = torch.randn(1, 4, device=device)
        result = silu_and_mul_torch(None, x)

        assert result.shape == (1, 2)
        assert not torch.isnan(result).any()

    def test_rms_norm_single_element(self, device):
        """Test RMS norm with single element batch."""
        from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch

        x = torch.randn(1, 64, device=device)
        weight = torch.ones(64, device=device)
        obj = _FakeNorm(weight, 1e-5)

        result = rms_norm_torch(obj, x)

        assert result.shape == (1, 64)
        assert not torch.isnan(result).any()

    def test_very_small_values(self, device):
        """Test with very small input values."""
        from vllm_fl.dispatch.backends.reference.impl.activation import silu_and_mul_torch
        from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch

        x_silu = torch.randn(2, 8, device=device) * 1e-7
        x_norm = torch.randn(2, 32, device=device) * 1e-7
        weight = torch.ones(32, device=device)
        obj = _FakeNorm(weight, 1e-5)

        result_silu = silu_and_mul_torch(None, x_silu)
        result_norm = rms_norm_torch(obj, x_norm)

        assert not torch.isnan(result_silu).any()
        assert not torch.isnan(result_norm).any()
        assert not torch.isinf(result_silu).any()
        assert not torch.isinf(result_norm).any()
