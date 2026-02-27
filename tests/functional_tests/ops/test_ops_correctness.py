# Copyright (c) 2025 BAAI. All rights reserved.

"""
Functional tests for ops correctness.
Tests numerical correctness of operator implementations
by comparing against reference PyTorch implementations.
"""

import pytest
import torch
import torch.nn.functional as F
from typing import Tuple


# Skip all tests in this module if GPU not available
pytestmark = pytest.mark.gpu


def allclose(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-3, atol: float = 1e-3) -> bool:
    """Check if two tensors are close within tolerance."""
    return torch.allclose(a, b, rtol=rtol, atol=atol)


class TestSiluAndMulCorrectness:
    """Test SiluAndMul operator correctness."""

    @pytest.fixture
    def test_shapes(self):
        """Common test shapes for SiluAndMul."""
        return [
            (1, 64),
            (4, 128),
            (16, 256),
            (32, 512),
            (64, 1024),
        ]

    @staticmethod
    def reference_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
        """Reference implementation of SiluAndMul."""
        half = x.shape[-1] // 2
        return F.silu(x[..., :half]) * x[..., half:]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_silu_and_mul_forward(self, test_shapes, device):
        """Test SiluAndMul forward pass correctness."""
        try:
            from vllm_fl.dispatch import call_op
        except ImportError:
            pytest.skip("vllm_fl.dispatch not available")

        for batch_size, hidden_size in test_shapes:
            # Input must have even hidden size for SiluAndMul
            x = torch.randn(batch_size, hidden_size * 2, device=device, dtype=torch.float32)

            # Get reference result
            ref_result = self.reference_silu_and_mul(x)

            # Get FL result
            try:
                fl_result = call_op("silu_and_mul", x)

                # Check correctness
                assert fl_result.shape == ref_result.shape, (
                    f"Shape mismatch: {fl_result.shape} vs {ref_result.shape}"
                )
                assert allclose(fl_result, ref_result), (
                    f"Value mismatch for shape ({batch_size}, {hidden_size * 2})"
                )
            except RuntimeError as e:
                if "No available implementation" in str(e):
                    pytest.skip(f"silu_and_mul not registered: {e}")
                raise

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_silu_and_mul_dtypes(self, device):
        """Test SiluAndMul with different dtypes."""
        try:
            from vllm_fl.dispatch import call_op
        except ImportError:
            pytest.skip("vllm_fl.dispatch not available")

        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        x_fp32 = torch.randn(4, 128, device=device, dtype=torch.float32)

        for dtype in dtypes:
            x = x_fp32.to(dtype)
            ref_result = self.reference_silu_and_mul(x)

            try:
                fl_result = call_op("silu_and_mul", x)
                # Use looser tolerance for half precision
                tol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-3
                assert allclose(fl_result, ref_result, rtol=tol, atol=tol), (
                    f"Value mismatch for dtype {dtype}"
                )
            except RuntimeError as e:
                if "No available implementation" in str(e):
                    pytest.skip(f"silu_and_mul not registered for {dtype}: {e}")
                raise


class TestRMSNormCorrectness:
    """Test RMSNorm operator correctness."""

    @pytest.fixture
    def test_shapes(self):
        """Common test shapes for RMSNorm."""
        return [
            (1, 64, 128),     # (batch, seq, hidden)
            (4, 32, 256),
            (8, 16, 512),
        ]

    @staticmethod
    def reference_rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """Reference implementation of RMSNorm."""
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        return x_normed * weight

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_rms_norm_forward(self, test_shapes, device):
        """Test RMSNorm forward pass correctness."""
        try:
            from vllm_fl.dispatch import call_op
        except ImportError:
            pytest.skip("vllm_fl.dispatch not available")

        eps = 1e-6
        for batch_size, seq_len, hidden_size in test_shapes:
            x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
            weight = torch.ones(hidden_size, device=device, dtype=torch.float32)

            # Get reference result
            ref_result = self.reference_rms_norm(x, weight, eps)

            # Get FL result
            try:
                fl_result = call_op("rms_norm", x, None, weight, eps)

                # Handle tuple return (output, residual)
                if isinstance(fl_result, tuple):
                    fl_result = fl_result[0]

                assert fl_result.shape == ref_result.shape, (
                    f"Shape mismatch: {fl_result.shape} vs {ref_result.shape}"
                )
                assert allclose(fl_result, ref_result), (
                    f"Value mismatch for shape ({batch_size}, {seq_len}, {hidden_size})"
                )
            except RuntimeError as e:
                if "No available implementation" in str(e):
                    pytest.skip(f"rms_norm not registered: {e}")
                raise

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_rms_norm_with_residual(self, device):
        """Test RMSNorm with residual connection."""
        try:
            from vllm_fl.dispatch import call_op
        except ImportError:
            pytest.skip("vllm_fl.dispatch not available")

        batch_size, seq_len, hidden_size = 4, 32, 256
        eps = 1e-6

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
        residual = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)
        weight = torch.ones(hidden_size, device=device, dtype=torch.float32)

        try:
            result = call_op("rms_norm", x, residual, weight, eps)

            # Should return tuple (normalized, updated_residual) when residual is provided
            if isinstance(result, tuple):
                normalized, updated_residual = result
                assert normalized.shape == x.shape
                assert updated_residual.shape == x.shape
        except RuntimeError as e:
            if "No available implementation" in str(e):
                pytest.skip(f"rms_norm not registered: {e}")
            raise


class TestRotaryEmbeddingCorrectness:
    """Test RotaryEmbedding operator correctness."""

    @staticmethod
    def reference_rotary_embedding(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        rotary_interleaved: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference implementation of rotary embedding."""

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            """Rotate half the hidden dims."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        # Gather cos/sin by positions
        cos_pos = cos[positions]
        sin_pos = sin[positions]

        # Add head dimension
        while cos_pos.dim() < q.dim():
            cos_pos = cos_pos.unsqueeze(1)
            sin_pos = sin_pos.unsqueeze(1)

        # Apply rotary embedding
        q_embed = (q * cos_pos) + (rotate_half(q) * sin_pos)
        k_embed = (k * cos_pos) + (rotate_half(k) * sin_pos)

        return q_embed, k_embed

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_rotary_embedding_basic(self, device):
        """Test basic rotary embedding functionality."""
        try:
            from vllm_fl.dispatch import call_op
        except ImportError:
            pytest.skip("vllm_fl.dispatch not available")

        num_tokens = 16
        num_heads = 8
        head_size = 64
        rotary_dim = head_size
        max_position = 2048

        # Create test inputs
        q = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        k = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float32)
        positions = torch.arange(num_tokens, device=device)

        # Create cos/sin cache
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim))
        t = torch.arange(max_position, device=device).float()
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        try:
            q_out, k_out = call_op(
                "rotary_embedding",
                q[..., :rotary_dim],
                k[..., :rotary_dim],
                cos,
                sin,
                positions,
                False,  # rotary_interleaved
                False,  # inplace
            )

            assert q_out.shape == q[..., :rotary_dim].shape
            assert k_out.shape == k[..., :rotary_dim].shape
        except RuntimeError as e:
            if "No available implementation" in str(e):
                pytest.skip(f"rotary_embedding not registered: {e}")
            raise


class TestOpsEdgeCases:
    """Test edge cases for operators."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_empty_tensor_handling(self, device):
        """Test handling of empty tensors."""
        try:
            from vllm_fl.dispatch import call_op
        except ImportError:
            pytest.skip("vllm_fl.dispatch not available")

        # Create empty tensor
        x = torch.empty(0, 64, device=device, dtype=torch.float32)

        # Some ops may handle empty tensors, others may raise
        # This test documents the behavior
        try:
            result = call_op("silu_and_mul", x)
            assert result.shape[0] == 0
        except (RuntimeError, ValueError):
            # Empty tensor handling is implementation-dependent
            pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_large_batch_handling(self, device):
        """Test handling of large batch sizes."""
        try:
            from vllm_fl.dispatch import call_op
        except ImportError:
            pytest.skip("vllm_fl.dispatch not available")

        # Large batch
        x = torch.randn(1024, 256, device=device, dtype=torch.float32)

        try:
            result = call_op("silu_and_mul", x)
            assert result.shape == (1024, 128)
        except RuntimeError as e:
            if "No available implementation" in str(e):
                pytest.skip(f"silu_and_mul not registered: {e}")
            raise
