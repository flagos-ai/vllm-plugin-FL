# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for the IluvatarBackend class surface.

Covers the parts of ``vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py``
that the core unit tests do not reach: ``is_available`` gating (vendor-name
fast path, CUDA-device-name fallback, cache stickiness) and the operator
methods delegated to the shared reference implementations (silu_and_mul,
rms_norm, rotary_embedding) plus ``attention_backend`` enum routing.

All numeric expectations are computed with independent torch oracles
(``torch.sigmoid`` / ``torch.rsqrt`` / manual rotate-half formulas), never
by re-calling the implementation under test. CPU-only: no vendor runtime,
no cuda device. Vendor-gated to the iluvatar platform.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("iluvatar")]


@pytest.fixture(autouse=True)
def _reset_available_cache():
    """Reset the class-level availability cache before and after each test.

    ``is_available`` memoizes into ``IluvatarBackend._available``; a stale
    True/False from one test would leak into every later one, so the cache is
    cleared on both entry and exit (order independence contract).
    """
    from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
        IluvatarBackend,
    )

    saved = IluvatarBackend._available
    IluvatarBackend._available = None
    try:
        yield
    finally:
        IluvatarBackend._available = saved


class TestIsAvailable:
    """``is_available`` vendor gate and CUDA fallback branches."""

    def test_vendor_name_gate_true(self, monkeypatch):
        import vllm.platforms

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        # is_available() re-imports current_platform from vllm.platforms
        # inside the method body, so patch the attribute on vllm.platforms.
        fake = SimpleNamespace(vendor_name="iluvatar")
        monkeypatch.setattr(vllm.platforms, "current_platform", fake)
        assert IluvatarBackend().is_available() is True
        assert IluvatarBackend._available is True

    def test_other_vendor_falls_through_to_cuda(self, monkeypatch):
        import vllm.platforms

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        monkeypatch.setattr(
            vllm.platforms, "current_platform", SimpleNamespace(vendor_name="nvidia")
        )
        # cuda unavailable -> False without calling get_device_name
        monkeypatch.setattr(
            torch.cuda, "is_available", lambda: False, raising=True
        )
        assert IluvatarBackend().is_available() is False

    def test_missing_vendor_name_uses_cuda_fallback(self, monkeypatch):
        import vllm.platforms

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        # no vendor_name attribute at all -> falls to the cuda branch
        monkeypatch.setattr(
            vllm.platforms, "current_platform", SimpleNamespace()
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "get_device_name", lambda idx: "Iluvatar BI-V150"
        )
        assert IluvatarBackend().is_available() is True

    @pytest.mark.parametrize(
        "device_name,expected",
        [
            ("Iluvatar BI-V150", True),
            ("ILUVATAR CoreX BI-V150", True),  # match is case-insensitive
            ("NVIDIA A800", False),
            ("AMD Instinct MI300X", False),
        ],
    )
    def test_cuda_device_name_fallback(self, monkeypatch, device_name, expected):
        import vllm.platforms

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        monkeypatch.setattr(
            vllm.platforms, "current_platform", SimpleNamespace(vendor_name="other")
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: device_name)
        assert IluvatarBackend().is_available() is expected

    def test_cuda_unavailable_is_false(self, monkeypatch):
        import vllm.platforms

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        monkeypatch.setattr(
            vllm.platforms, "current_platform", SimpleNamespace(vendor_name="other")
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert IluvatarBackend().is_available() is False

    def test_class_cache_is_sticky(self, monkeypatch):
        """Second call must not re-probe the platform (memoized True)."""
        import vllm.platforms

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        monkeypatch.setattr(
            vllm.platforms, "current_platform", SimpleNamespace(vendor_name="iluvatar")
        )
        backend = IluvatarBackend()
        assert backend.is_available() is True
        # Change the platform; the cached value must win.
        monkeypatch.setattr(
            vllm.platforms, "current_platform", SimpleNamespace(vendor_name="other")
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert backend.is_available() is True

    def test_platform_exception_is_swallowed(self, monkeypatch):
        import vllm.platforms

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        class ExplodingPlatform:
            @property
            def vendor_name(self):
                raise RuntimeError("platform probe failed")

        monkeypatch.setattr(
            vllm.platforms, "current_platform", ExplodingPlatform()
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert IluvatarBackend().is_available() is False

    def test_name_and_vendor_properties(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        backend = IluvatarBackend()
        assert backend.name == "iluvatar"
        assert backend.vendor == "iluvatar"


class TestSiluAndMul:
    """silu_and_mul delegates to the pure-torch reference implementation."""

    @pytest.mark.parametrize(
        "shape", [(4, 8), (3, 5, 12), (6,), (2, 0, 4)]
    )
    def test_matches_sigmoid_oracle(self, shape):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        x = torch.randn(*shape)
        d = x.shape[-1] // 2
        out = IluvatarBackend().silu_and_mul(None, x)
        assert out.shape == x.shape[:-1] + (d,)
        x1, x2 = x[..., :d], x[..., d:]
        # independent oracle: silu(y) = y * sigmoid(y)
        oracle = x1 * torch.sigmoid(x1) * x2
        assert torch.allclose(out, oracle)

    def test_odd_trailing_dim_errors(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        # A trailing dim of 7 splits into 3 + 4 halves whose product
        # broadcasts wrongly; the pure-torch impl surfaces that error.
        x = torch.randn(7)
        with pytest.raises(RuntimeError):
            IluvatarBackend().silu_and_mul(None, x)

    def test_extreme_values(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        x = torch.tensor(
            [[-100.0, -100.0, 0.0, 100.0, 100.0, 100.0]]
        )
        out = IluvatarBackend().silu_and_mul(None, x)
        x1, x2 = x[..., :3], x[..., 3:]
        oracle = x1 * torch.sigmoid(x1) * x2
        assert torch.allclose(out, oracle)
        assert torch.all(torch.isfinite(out))


class TestRmsNorm:
    """rms_norm delegates to the shared reference rms_norm_torch."""

    @pytest.fixture
    def obj(self):
        torch.manual_seed(11)
        return SimpleNamespace(
            variance_epsilon=1e-6, weight=torch.randn(8, dtype=torch.float16)
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_matches_fp32_oracle(self, obj, dtype):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        x = torch.randn(3, 8).to(dtype)
        out = IluvatarBackend().rms_norm(obj, x)
        assert out.dtype == dtype
        x_float = x.float()
        var = x_float.square().mean(dim=-1, keepdim=True)
        oracle = x_float * torch.rsqrt(var + obj.variance_epsilon)
        oracle = (oracle.to(obj.weight.dtype) * obj.weight).to(dtype)
        # cast-back noise only; compare in fp32
        assert torch.allclose(out.float(), oracle.float(), atol=2e-2, rtol=2e-2)

    def test_residual_returns_pair(self, obj):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        x = torch.randn(3, 8, dtype=torch.float16)
        residual = torch.randn(3, 8, dtype=torch.float16)
        out, new_residual = IluvatarBackend().rms_norm(obj, x, residual)

        x_float = x.float() + residual.float()
        assert new_residual.dtype == torch.float16
        assert torch.allclose(new_residual.float(), x_float.to(torch.float16).float())
        var = x_float.square().mean(dim=-1, keepdim=True)
        oracle = x_float * torch.rsqrt(var + obj.variance_epsilon)
        oracle = (oracle.to(obj.weight.dtype) * obj.weight).to(torch.float16)
        assert torch.allclose(out.float(), oracle.float(), atol=2e-2, rtol=2e-2)

    def test_variance_size_override_slices_variance_window(self, obj):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        obj.variance_size_override = 5
        x = torch.randn(3, 8, dtype=torch.float16)
        out = IluvatarBackend().rms_norm(obj, x)
        x_float = x.float()
        var = x_float[..., :5].square().mean(dim=-1, keepdim=True)
        oracle = x_float * torch.rsqrt(var + obj.variance_epsilon)
        oracle = (oracle.to(obj.weight.dtype) * obj.weight).to(torch.float16)
        assert torch.allclose(out.float(), oracle.float(), atol=2e-2, rtol=2e-2)

    def test_pass_weight_false_skips_weight(self, obj):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        obj.pass_weight = False
        x = torch.randn(3, 8, dtype=torch.float16)
        out = IluvatarBackend().rms_norm(obj, x)
        x_float = x.float()
        var = x_float.square().mean(dim=-1, keepdim=True)
        oracle = (x_float * torch.rsqrt(var + obj.variance_epsilon)).to(
            torch.float16
        )
        assert torch.allclose(out.float(), oracle.float(), atol=2e-2, rtol=2e-2)

    def test_pass_weight_add_false_skips_weight_on_residual_path(self, obj):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        obj.pass_weight_add = False
        x = torch.randn(3, 8, dtype=torch.float16)
        residual = torch.randn(3, 8, dtype=torch.float16)
        out, _ = IluvatarBackend().rms_norm(obj, x, residual)
        x_float = x.float() + residual.float()
        var = x_float.square().mean(dim=-1, keepdim=True)
        oracle = (x_float * torch.rsqrt(var + obj.variance_epsilon)).to(
            torch.float16
        )
        assert torch.allclose(out.float(), oracle.float(), atol=2e-2, rtol=2e-2)

    def test_dtype_round_trip_float32_exact(self):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        obj = SimpleNamespace(variance_epsilon=1e-6, weight=torch.randn(8))
        x = torch.randn(3, 8, dtype=torch.float32)
        out = IluvatarBackend().rms_norm(obj, x)
        var = x.square().mean(dim=-1, keepdim=True)
        oracle = x * torch.rsqrt(var + 1e-6) * obj.weight
        # fp32 in / fp32 weight: every intermediate stays fp32, exact match
        assert out.dtype == torch.float32
        assert torch.equal(out, oracle)


def _reference_rope(q, k, cos, sin, position_ids, interleaved):
    """Independent rotary oracle built from first principles.

    ``cos``/``sin`` are 2D caches whose last dim is either the full head
    dim or its half; half-width caches expand per rotary style (adjacent
    pairs for interleaved, contiguous halves for NeoX).
    """
    head_dim = q.shape[-1]
    cache_width = cos.shape[-1]

    def expand(vals):
        if cache_width == head_dim:
            return vals
        if interleaved:
            return vals.repeat_interleave(2, dim=-1)
        return torch.cat([vals, vals], dim=-1)

    def rotate(x):
        if interleaved:
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).flatten(-2)
        x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    outs = []
    for tensor in (q, k):
        c = expand(cos[position_ids.long()])
        s = expand(sin[position_ids.long()])
        if tensor.dim() == 4 and position_ids.dim() == 1:
            c = c.unsqueeze(0).unsqueeze(0)
            s = s.unsqueeze(0).unsqueeze(0)
        else:
            c = c.unsqueeze(1)
            s = s.unsqueeze(1)
        outs.append(tensor * c.to(tensor.dtype) + rotate(tensor) * s.to(tensor.dtype))
    return outs


class TestRotaryEmbedding:
    """rotary_embedding delegates to the shared reference implementation."""

    HEAD_DIM = 8

    @pytest.fixture
    def caches(self):
        angles = (
            torch.arange(40)[:, None]
            * torch.arange(self.HEAD_DIM)[None, :]
            / self.HEAD_DIM
        )
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return cos, sin

    @pytest.mark.parametrize("interleaved", [False, True])
    @pytest.mark.parametrize("half_width", [False, True])
    def test_3d_shared_positions(self, caches, interleaved, half_width):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        torch.manual_seed(5)
        cos, sin = caches
        if half_width:
            cos, sin = cos[:, : self.HEAD_DIM // 2], sin[:, : self.HEAD_DIM // 2]
        q = torch.randn(6, 2, self.HEAD_DIM)
        k = torch.randn(6, 2, self.HEAD_DIM)
        pos = torch.arange(6)
        q_out, k_out = IluvatarBackend().rotary_embedding(
            None, q, k, cos, sin, pos, rotary_interleaved=interleaved
        )
        ref_q, ref_k = _reference_rope(q, k, cos, sin, pos, interleaved)
        assert torch.allclose(q_out, ref_q)
        assert torch.allclose(k_out, ref_k)

    @pytest.mark.parametrize("interleaved", [False, True])
    @pytest.mark.parametrize("shared", [False, True])
    def test_4d_batched_and_shared_positions(self, caches, interleaved, shared):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        torch.manual_seed(6)
        cos, sin = caches
        q = torch.randn(2, 3, 5, self.HEAD_DIM)
        k = torch.randn(2, 3, 5, self.HEAD_DIM)
        pos = (
            torch.arange(5)
            if shared
            else torch.randint(0, 30, (2, 5))
        )
        q_out, k_out = IluvatarBackend().rotary_embedding(
            None, q, k, cos, sin, pos, rotary_interleaved=interleaved
        )
        ref_q, ref_k = _reference_rope(q, k, cos, sin, pos, interleaved)
        assert torch.allclose(q_out, ref_q)
        assert torch.allclose(k_out, ref_k)

    def test_non_default_inplace_flag_matches(self, caches):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        cos, sin = caches
        q = torch.randn(6, 2, self.HEAD_DIM)
        k = torch.randn(6, 2, self.HEAD_DIM)
        pos = torch.arange(6)
        q_out, k_out = IluvatarBackend().rotary_embedding(
            None, q, k, cos, sin, pos, inplace=False
        )
        ref_q, ref_k = _reference_rope(q, k, cos, sin, pos, False)
        assert torch.allclose(q_out, ref_q)
        assert torch.allclose(k_out, ref_k)

    def test_invalid_position_dtype_raises(self, caches):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        cos, sin = caches
        q = torch.randn(4, 2, self.HEAD_DIM)
        k = torch.randn(4, 2, self.HEAD_DIM)
        with pytest.raises(ValueError, match="dtype"):
            IluvatarBackend().rotary_embedding(
                None, q, k, cos, sin, torch.arange(4, dtype=torch.float32)
            )

    def test_token_count_mismatch_raises(self, caches):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        cos, sin = caches
        q = torch.randn(4, 2, self.HEAD_DIM)
        k = torch.randn(4, 2, self.HEAD_DIM)
        with pytest.raises(ValueError, match="token counts"):
            IluvatarBackend().rotary_embedding(
                None, q, k, cos, sin, torch.arange(3)
            )

    def test_invalid_cache_width_raises(self, caches):
        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        cos, sin = caches
        q = torch.randn(4, 2, self.HEAD_DIM)
        k = torch.randn(4, 2, self.HEAD_DIM)
        with pytest.raises(ValueError, match="width"):
            IluvatarBackend().rotary_embedding(
                None, q, k, cos[:, :3], sin[:, :3], torch.arange(4)
            )


class TestAttentionBackend:
    """attention_backend routes to TRITON_ATTN / FLASHMLA / FLASHMLA_SPARSE."""

    def test_default_is_triton_attn(self):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        path = IluvatarBackend().attention_backend()
        assert path == AttentionBackendEnum.TRITON_ATTN.get_path()
        # flash_attn must not be selected on iluvatar
        assert "TritonAttentionBackend" in path

    def test_mla_selects_flashmla(self):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        path = IluvatarBackend().attention_backend(use_mla=True)
        assert path == AttentionBackendEnum.FLASHMLA.get_path()
        assert "FlashMLABackend" in path

    def test_mla_sparse_selects_flashmla_sparse(self):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        path = IluvatarBackend().attention_backend(use_mla=True, use_sparse=True)
        assert path == AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
        assert "FlashMLASparseBackend" in path

    def test_sparse_without_mla_stays_triton(self):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
            IluvatarBackend,
        )

        # sparse is only consulted inside the mla branch
        path = IluvatarBackend().attention_backend(use_mla=False, use_sparse=True)
        assert path == AttentionBackendEnum.TRITON_ATTN.get_path()
