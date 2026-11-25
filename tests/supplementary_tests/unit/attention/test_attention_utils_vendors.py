# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for the vendor arms of ``vllm_fl.attention.utils``.

``patch_mm_encoder_attention`` installs per-vendor multimodal encoder
attention adapters. The hygon arm (contiguous BHSD SDPA with D72 padding)
is already exercised by the hygon-platform suite; this file drives the
remaining arms against the real production source loaded under a stubbed
``vllm.*`` import graph (no vLLM install needed):

* **gcu/enflame** — ``forward_native``/``forward_oot`` replacement with the
  original saved once, the non-FLASH_ATTN delegation branch, the varlen
  flash path with cu_seqlens (per-batch lengths, scale/reshape contract)
  and the both-None assertion;
* **kunlunxin** — the ``_forward_sdpa`` query-chunked install (its numeric
  identity vs an independent explicit-softmax oracle, both the plain and
  the cu_seqlens varlen form, the GQA flag, chunk-boundary behavior beyond
  the 512 chunk, and the 3D-input reshape);
* **nvidia / other** — the ``forward_oot -> forward_cuda`` binding and the
  ``maybe_get_vit_flash_attn_backend`` override for every vendor arm,
  including the ImportError paths and the aiter branch.

All ``sys.modules`` entries are installed via ``monkeypatch.setitem`` so the
stubs never leak; numeric assertions use oracles written independently from
the source's formulas.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.supplementary]

_SOURCE = Path(__file__).resolve().parents[4] / "vllm_fl/attention/utils.py"
_MODULE_NAME = "supp_attention_utils_vendors_under_test"

_VIT_LEAVES = ("vllm.v1.attention.ops.vit_attn_wrappers",)
_MM_LEAVES = ("vllm.model_executor.layers.attention.mm_encoder_attention",)


def _install_vllm_stubs(monkeypatch, vendor_name, is_cuda=False):
    """Create the stubbed vllm module tree the adapters patch into.

    Returns the modules whose attributes the production code mutates.
    """
    modules = {}

    def leaf(name):
        parts = name.split(".")
        for length in range(1, len(parts) + 1):
            partial = ".".join(parts[:length])
            if partial not in modules:
                module = ModuleType(partial)
                module.__path__ = []
                modules[partial] = module
        for length in range(1, len(parts)):
            parent, attribute = ".".join(parts[:length]), parts[length]
            setattr(
                modules[parent], attribute, modules[".".join(parts[: length + 1])]
            )
        return modules[name]

    for name in _VIT_LEAVES + _MM_LEAVES + ("vllm.platforms", "vllm.logger"):
        leaf(name)

    # --- vllm.logger (the module's only top-level import) ---
    logger_module = modules["vllm.logger"]
    logged = []

    class _Logger:
        def info_once(self, msg, *args):
            logged.append(msg % args if args else msg)

        def info(self, msg, *args):
            logged.append(msg % args if args else msg)

        def warning(self, msg, *args):
            logged.append(msg % args if args else msg)

    logger_module.init_logger = lambda name: _Logger()

    # --- vllm.v1.attention.ops.vit_attn_wrappers ---
    vit = modules[_VIT_LEAVES[0]]
    vit.einops = SimpleNamespace(rearrange=_rearrange_bshd_bhsd)
    vit.F = SimpleNamespace(
        pad=F.pad, scaled_dot_product_attention=F.scaled_dot_product_attention
    )

    def original_apply_sdpa(q, k, v, scale=None, enable_gqa=False):
        return F.scaled_dot_product_attention(
            *(t.transpose(1, 2) for t in (q, k, v)),
            dropout_p=0.0,
            scale=scale,
            enable_gqa=enable_gqa,
        ).transpose(1, 2)

    vit.apply_sdpa = original_apply_sdpa

    # --- vllm.model_executor.layers.attention.mm_encoder_attention ---
    mm = modules[_MM_LEAVES[0]]

    class MMEncoderAttention:
        def forward_native(self, query, key, value, cu_seqlens=None, max_seqlen=None,
                           sequence_lengths=None):
            raise AssertionError("original forward_native must not run")

        def forward_oot(self, *args, **kwargs):
            raise AssertionError("original forward_oot must not run")

        def forward_cuda(self, *args, **kwargs):
            raise AssertionError("original forward_cuda must not run")

    mm.MMEncoderAttention = MMEncoderAttention

    def original_maybe_get(attn_backend):
        raise AssertionError("original maybe_get must be replaced")

    mm.maybe_get_vit_flash_attn_backend = original_maybe_get

    # --- vllm.platforms with a vendor-switchable current_platform ---
    platforms = modules["vllm.platforms"]
    platform = SimpleNamespace(vendor_name=vendor_name, is_cuda=lambda: is_cuda)
    platforms.current_platform = platform

    # --- misc imports used by the gcu path ---
    gpu_sync = leaf("vllm.utils.gpu_sync_debug")
    gpu_sync.gpu_sync_allowed = nullcontext
    registry = leaf("vllm.v1.attention.backends.registry")

    class AttentionBackendEnum:
        FLASH_ATTN = "FLASH_ATTN"
        ROCM_AITER_FA = "ROCM_AITER_FA"

    registry.AttentionBackendEnum = AttentionBackendEnum

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return SimpleNamespace(vit=vit, mm=mm, platform=platform, registry=registry)


def _rearrange_bshd_bhsd(x, pattern):
    """einops.rearrange stand-in for the two patterns the adapter uses."""
    if pattern.replace(" ", "") == "bshd->bhsd":
        return x.transpose(1, 2)
    if pattern.replace(" ", "") == "bhsd->bshd":
        return x.transpose(1, 2)
    raise AssertionError(f"unexpected rearrange pattern {pattern!r}")


@pytest.fixture
def load_adapter(monkeypatch):
    """Load a fresh copy of vllm_fl/attention/utils.py per test."""

    def load(vendor_name, is_cuda=False):
        stubs = _install_vllm_stubs(monkeypatch, vendor_name, is_cuda)
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, _MODULE_NAME, module)
        spec.loader.exec_module(module)
        module.patch_mm_encoder_attention()
        return module, stubs

    return load


def _oracle_bshd(q, k, v, scale=None, enable_gqa=False):
    """Independent SDPA oracle in BSHD layout, explicit softmax in float64."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    qf = q.transpose(1, 2).double()  # B,H,S,D
    kf = k.transpose(1, 2).double()
    vf = v.transpose(1, 2).double()
    if enable_gqa:
        repeats = qf.shape[1] // kf.shape[1]
        kf = kf.repeat_interleave(repeats, dim=1)
        vf = vf.repeat_interleave(repeats, dim=1)
    scores = torch.einsum("bhsd,bhtd->bhst", qf, kf) * scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhst,bhtd->bhsd", weights, vf)
    return out.transpose(1, 2).to(q.dtype)


class TestGcuEnflameArm:
    """The gcu/enflame branch of patch_mm_encoder_attention."""

    def test_patch_binds_forward_native_and_oot(self, load_adapter):
        """On gcu, MMEncoderAttention.forward_native and forward_oot are replaced."""
        module, stubs = load_adapter("gcu")
        cls = stubs.mm.MMEncoderAttention

        assert cls.forward_native is module._gcu_forward_mm_attention
        assert cls.forward_oot is module._gcu_forward_mm_attention
        assert hasattr(cls, "_fl_gcu_original_forward_native")

    def test_patch_is_idempotent(self, load_adapter):
        """Applying the gcu patch twice keeps exactly one original forward saved."""
        module, stubs = load_adapter("enflame")
        cls = stubs.mm.MMEncoderAttention
        first_original = cls._fl_gcu_original_forward_native

        module.patch_mm_encoder_attention()

        assert cls._fl_gcu_original_forward_native is first_original
        assert cls.forward_native is module._gcu_forward_mm_attention

    def test_early_return_skips_maybe_get_replacement(self, load_adapter):
        """The gcu/enflame early return skips the vit backend selector override."""
        _, stubs = load_adapter("gcu")
        selector = stubs.mm.maybe_get_vit_flash_attn_backend
        assert selector is not None
        # The original stub raises; the override would have a different name.
        assert selector.__name__ != "_patched_maybe_get_vit_flash_attn_backend"

    def test_delegates_when_backend_not_flash(self, load_adapter, monkeypatch):
        """With a non-FLASH_ATTN backend the original native forward runs."""
        module, stubs = load_adapter("gcu")
        cls = stubs.mm.MMEncoderAttention

        calls = []

        def original_forward(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "original-result"

        monkeypatch.setattr(cls, "_fl_gcu_original_forward_native", original_forward)

        attention = object.__new__(cls)
        attention.attn_backend = stubs.registry.AttentionBackendEnum.ROCM_AITER_FA

        result = module._gcu_forward_mm_attention(
            attention, "q", "k", "v", None, None, None
        )
        assert result == "original-result"
        assert len(calls) == 1
        # All six forward_native arguments are forwarded positionally.
        assert calls[0][0] == ("q", "k", "v", None, None, None)

    def test_cu_seqlens_max_seqlen_both_none_asserts(self, load_adapter, monkeypatch):
        """cu_seqlens None with a non-None max_seqlen fails the paired assertion."""
        module, stubs = load_adapter("gcu")
        cls = stubs.mm.MMEncoderAttention

        # The FLASH path imports flash_attn before the assert; stub it out.
        flash_pkg = ModuleType("flash_attn")
        flash_pkg.__path__ = []
        vllm_flash = ModuleType("flash_attn.vllm_flash_attn")
        vllm_flash.flash_attn_varlen_func = lambda **kwargs: None
        monkeypatch.setitem(sys.modules, "flash_attn", flash_pkg)
        monkeypatch.setitem(sys.modules, "flash_attn.vllm_flash_attn", vllm_flash)

        attention = object.__new__(cls)
        attention.attn_backend = stubs.registry.AttentionBackendEnum.FLASH_ATTN
        with pytest.raises(AssertionError, match="both set or both None"):
            module._gcu_forward_mm_attention(
                attention, torch.zeros(1), torch.zeros(1), torch.zeros(1),
                None, torch.tensor(5), None,
            )


class TestGcuVarlenForward:
    """The FLASH_ATTN varlen path of _gcu_forward_mm_attention."""

    @pytest.fixture
    def gcu_env(self, load_adapter, monkeypatch):
        """A gcu-patched module whose flash_attn kernel is a CPU oracle."""
        module, stubs = load_adapter("gcu")
        cls = stubs.mm.MMEncoderAttention
        kernel_calls = []

        def flash_attn_varlen_func(*, q, k, v, cu_seqlens_q, cu_seqlens_k,
                                   max_seqlen_q, max_seqlen_k, dropout_p,
                                   causal, softmax_scale, fa_version, **kwargs):
            kernel_calls.append(
                {
                    "q": q, "k": k, "v": v,
                    "cu_seqlens_q": cu_seqlens_q,
                    "cu_seqlens_k": cu_seqlens_k,
                    "max_seqlen_q": max_seqlen_q,
                    "max_seqlen_k": max_seqlen_k,
                    "softmax_scale": softmax_scale,
                    "causal": causal,
                    "fa_version": fa_version,
                }
            )
            # CPU oracle: attend over each varlen segment independently.
            # The kernel receives flattened (B*S, H, D) tensors; each segment
            # spans the rows of its batch, with q rows of length H*D.
            out = torch.empty_like(q)
            seg_q = cu_seqlens_q.tolist()
            seg_k = cu_seqlens_k.tolist()
            for idx in range(len(seg_q) - 1):
                # Per-batch slices of the flattened token dimension.
                q_seg = q[seg_q[idx]:seg_q[idx + 1]]  # (S_q, H, D)
                k_seg = k[seg_k[idx]:seg_k[idx + 1]]  # (S_k, H_kv, D)
                v_seg = v[seg_k[idx]:seg_k[idx + 1]]
                heads = q_seg.shape[1]
                kv_heads = k_seg.shape[1]
                repeats = heads // kv_heads
                k_exp = k_seg.repeat_interleave(repeats, dim=1)
                v_exp = v_seg.repeat_interleave(repeats, dim=1)
                # (S_q, H, S_k) via per-head dot products.
                scores = torch.einsum("shd,thd->hst", q_seg, k_exp) * softmax_scale
                weights = torch.softmax(scores, dim=-1)
                out_seg = torch.einsum("hst,thd->shd", weights, v_exp)
                out[seg_q[idx]:seg_q[idx + 1]] = out_seg
            return out

        flash_pkg = ModuleType("flash_attn")
        flash_pkg.__path__ = []
        vllm_flash = ModuleType("flash_attn.vllm_flash_attn")
        vllm_flash.flash_attn_varlen_func = flash_attn_varlen_func
        monkeypatch.setitem(sys.modules, "flash_attn", flash_pkg)
        monkeypatch.setitem(sys.modules, "flash_attn.vllm_flash_attn", vllm_flash)

        attention = object.__new__(cls)
        attention.attn_backend = stubs.registry.AttentionBackendEnum.FLASH_ATTN
        attention.scale = 0.25
        attention.num_heads = 2
        attention.num_kv_heads = 1
        attention.head_size = 4

        def view_qkv_to_4d(query, key, value, bsz, q_len, kv_len):
            # vLLM's real helper: [B, S, H*D] -> [B, S, H, D] (no transpose).
            return (
                query.view(bsz, q_len, attention.num_heads, attention.head_size),
                key.view(bsz, kv_len, attention.num_kv_heads, attention.head_size),
                value.view(bsz, kv_len, attention.num_kv_heads, attention.head_size),
            )

        attention.view_qkv_to_4d = view_qkv_to_4d
        return SimpleNamespace(
            module=module,
            attention=attention,
            calls=kernel_calls,
            stubs=stubs,
        )

    def test_without_cu_seqlens_synthesizes_uniform_segments(self, gcu_env):
        """cu_seqlens=None builds (B+1) uniform segment boundaries per batch."""
        generator = torch.Generator().manual_seed(41)
        bsz, q_len, kv_len, heads, dim = 2, 3, 5, 2, 4
        query = torch.randn(bsz, q_len, heads * dim, generator=generator)
        key = torch.randn(bsz, kv_len, 1 * dim, generator=generator)
        value = torch.randn(bsz, kv_len, 1 * dim, generator=generator)

        output = gcu_env.module._gcu_forward_mm_attention(
            gcu_env.attention, query, key, value, None, None, None
        )

        call = gcu_env.calls[-1]
        # Two uniform segments: [0, 3) and [3, 6) for q, [0, 5)/[5,10) for k.
        assert call["cu_seqlens_q"].tolist() == [0, 3, 6]
        assert call["cu_seqlens_k"].tolist() == [0, 5, 10]
        assert call["max_seqlen_q"] == 3
        assert call["max_seqlen_k"] == 5
        assert call["softmax_scale"] == 0.25
        assert call["causal"] is False
        assert call["fa_version"] == 2
        # The flattened kernel tensors are the transposed 4D views.
        assert call["q"].shape == (bsz * q_len, heads, dim)
        # Output restores the (B, S, H*D) input layout.
        assert output.shape == (bsz, q_len, heads * dim)
        # Numeric check against the independent BSHD oracle.
        q4 = query.view(bsz, q_len, heads, dim)
        k4 = key.view(bsz, kv_len, 1, dim)
        v4 = value.view(bsz, kv_len, 1, dim)
        expected = _oracle_bshd(q4, k4, v4, scale=0.25, enable_gqa=True)
        torch.testing.assert_close(
            output, expected.reshape(bsz, q_len, -1), rtol=1e-4, atol=1e-4
        )

    def test_with_cu_seqlens_uses_them_for_both_sides(self, gcu_env):
        """cu_seqlens/max_seqlen are forwarded for both q and k boundaries.

        The production path flattens the 4D views to (B*S, H, D) before the
        kernel, and the kernel oracle attends each cu_seqlens segment of the
        flattened token dimension.
        """
        generator = torch.Generator().manual_seed(43)
        # The fixture attention is configured with num_heads=2,
        # num_kv_heads=1, head_size=4: q carries 2 heads, k/v carry 1.
        bsz, q_len = 1, 6
        query = torch.randn(bsz, q_len, 2, 4, generator=generator)
        key = torch.randn(bsz, q_len, 1, 4, generator=generator)
        value = torch.randn(bsz, q_len, 1, 4, generator=generator)
        cu_seqlens = torch.tensor([0, 4, 6], dtype=torch.int32)
        max_seqlen = torch.tensor(4)

        output = gcu_env.module._gcu_forward_mm_attention(
            gcu_env.attention, query, key, value, cu_seqlens, max_seqlen, None
        )

        call = gcu_env.calls[-1]
        assert torch.equal(call["cu_seqlens_q"], cu_seqlens)
        assert torch.equal(call["cu_seqlens_k"], cu_seqlens)
        assert call["max_seqlen_q"] == 4
        assert call["max_seqlen_k"] == 4
        assert output.shape == (bsz, q_len, 2, 4)
        # Oracle: the kernel sees the flattened per-head layout (S, H, D);
        # attend over each cu_seqlens segment with both heads sharing the
        # single kv head (GQA), per-head dot products.
        flat_q = query.reshape(q_len, 2, 4)  # (S, H=2, D)
        flat_k = key.reshape(q_len, 1, 4)  # (S, H_kv=1, D)
        flat_v = value.reshape(q_len, 1, 4)
        bounds = cu_seqlens.tolist()
        expected = torch.empty_like(flat_q)
        for idx in range(len(bounds) - 1):
            q_seg = flat_q[bounds[idx]:bounds[idx + 1]]  # (s, H, D)
            k_seg = flat_k[bounds[idx]:bounds[idx + 1]]  # (s, 1, D)
            v_seg = flat_v[bounds[idx]:bounds[idx + 1]]
            scores = torch.einsum("shd,thd->hst", q_seg, k_seg) * 0.25
            weights = torch.softmax(scores, dim=-1)
            out_seg = torch.einsum("hst,thd->shd", weights, v_seg)
            expected[bounds[idx]:bounds[idx + 1]] = out_seg
        torch.testing.assert_close(output, expected.reshape(bsz, q_len, 2, 4))


class TestKunlunxinArm:
    """The kunlunxin branch of patch_mm_encoder_attention."""

    def test_forward_sdpa_installed(self, load_adapter):
        """On kunlunxin, MMEncoderAttention._forward_sdpa is the chunked version."""
        module, stubs = load_adapter("kunlunxin")
        cls = stubs.mm.MMEncoderAttention

        assert hasattr(cls, "_forward_sdpa")
        source = _SOURCE.read_text(encoding="utf-8")
        assert "_KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE = 512" in source
        assert source.count("_forward_sdpa = _kunlunxin_forward_sdpa") == 1

    def _make_attention(self, load_adapter, heads=2, kv_heads=2, dim=8):
        module, stubs = load_adapter("kunlunxin")
        attention = object.__new__(stubs.mm.MMEncoderAttention)
        attention.scale = dim**-0.5
        attention.num_heads = heads
        attention.num_kv_heads = kv_heads
        attention.head_size = dim

        def view_qkv_to_4d(query, key, value, bsz, q_len, kv_len):
            # vLLM's real helper: [B, S, H*D] -> [B, S, H, D] (no transpose).
            return (
                query.view(bsz, q_len, heads, dim),
                key.view(bsz, kv_len, kv_heads, dim),
                value.view(bsz, kv_len, kv_heads, dim),
            )

        attention.view_qkv_to_4d = view_qkv_to_4d
        return module, attention

    def test_plain_sdpa_matches_oracle(self, load_adapter):
        """Without cu_seqlens the chunked SDPA is numerically exact."""
        module, attention = self._make_attention(load_adapter)
        generator = torch.Generator().manual_seed(61)
        bsz, s_q, s_k, heads, dim = 2, 7, 9, 2, 8
        query = torch.randn(bsz, s_q, heads, dim, generator=generator)
        key = torch.randn(bsz, s_k, 2, dim, generator=generator)
        value = torch.randn(bsz, s_k, 2, dim, generator=generator)

        out = attention._forward_sdpa(query, key, value, cu_seqlens=None)
        expected = _oracle_bshd(query, key, value, scale=attention.scale)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_cu_seqlens_segments_attended_independently(self, load_adapter):
        """cu_seqlens splits the flat tensors into independently attended parts.

        With cu_seqlens the production code splits along dim=1 of the 4D
        views (one segment per cu_seqlens span), so the caller's flat token
        layout must have every segment contiguous: build (1, S, H, D) inputs.
        """
        module, attention = self._make_attention(load_adapter)
        generator = torch.Generator().manual_seed(63)
        heads, dim = 2, 8
        s_q = 10
        query = torch.randn(1, s_q, heads, dim, generator=generator)
        key = torch.randn(1, s_q, heads, dim, generator=generator)
        value = torch.randn(1, s_q, heads, dim, generator=generator)
        cu_seqlens = torch.tensor([0, 4, 10], dtype=torch.int32)

        out = attention._forward_sdpa(query, key, value, cu_seqlens)

        bounds = cu_seqlens.tolist()
        expected = torch.empty_like(query)
        for idx in range(len(bounds) - 1):
            q_seg = query[0, bounds[idx]:bounds[idx + 1]]
            k_seg = key[0, bounds[idx]:bounds[idx + 1]]
            v_seg = value[0, bounds[idx]:bounds[idx + 1]]
            expected[0, bounds[idx]:bounds[idx + 1]] = _oracle_bshd(
                q_seg.unsqueeze(0), k_seg.unsqueeze(0), v_seg.unsqueeze(0),
                scale=attention.scale,
            )[0]
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_gqa_flag_passed_when_heads_differ(self, load_adapter, monkeypatch):
        """num_heads > num_kv_heads passes enable_gqa=True to SDPA."""
        module, attention = self._make_attention(
            load_adapter, heads=4, kv_heads=2
        )
        seen = {}

        def recording_sdpa(q, k, v, dropout_p=0.0, scale=None, enable_gqa=False):
            seen["enable_gqa"] = enable_gqa
            return torch.zeros_like(q)

        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention",
                            recording_sdpa)
        generator = torch.Generator().manual_seed(65)
        bsz, s_q, s_k, heads, dim = 1, 3, 3, 4, 8
        query = torch.randn(bsz, s_q, heads * dim, generator=generator)
        key = torch.randn(bsz, s_k, 2 * dim, generator=generator)
        value = torch.randn(bsz, s_k, 2 * dim, generator=generator)
        attention._forward_sdpa(query, key, value, cu_seqlens=None)
        assert seen["enable_gqa"] is True

    def test_chunk_boundary_exactness_beyond_chunk_size(self, load_adapter,
                                                        monkeypatch):
        """Inputs longer than the 512 query chunk still produce exact output."""
        heads, dim = 2, 8
        module, attention = self._make_attention(
            load_adapter, heads=heads, kv_heads=2, dim=dim
        )
        monkeypatch.setattr(
            module, "_KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE", 3, raising=False
        )
        generator = torch.Generator().manual_seed(67)
        bsz, s_q, s_k = 2, 8, 6
        query = torch.randn(bsz, s_q, heads * dim, generator=generator)
        key = torch.randn(bsz, s_k, 2 * dim, generator=generator)
        value = torch.randn(bsz, s_k, 2 * dim, generator=generator)

        out = attention._forward_sdpa(query, key, value, cu_seqlens=None)

        q4 = query.view(bsz, s_q, heads, dim)
        k4 = key.view(bsz, s_k, 2, dim)
        v4 = value.view(bsz, s_k, 2, dim)
        expected = _oracle_bshd(q4, k4, v4, scale=attention.scale)
        torch.testing.assert_close(
            out, expected.reshape(bsz, s_q, -1), rtol=1e-4, atol=1e-4
        )


class TestNvidiaAndSelectorArms:
    """The nvidia forward_oot binding and the vit backend selector override."""

    def test_nvidia_binds_forward_oot_to_forward_cuda(self, load_adapter):
        """On CUDA platforms, forward_oot must alias vLLM's forward_cuda."""
        module, stubs = load_adapter("nvidia", is_cuda=True)
        cls = stubs.mm.MMEncoderAttention

        sentinel = object()
        cls.forward_cuda = sentinel
        module.patch_mm_encoder_attention()
        assert cls.forward_oot is sentinel

    @pytest.mark.parametrize("vendor", ["hygon", "kunlunxin", "nvidia"])
    def test_selector_replaced_for_non_gcu_vendors(self, load_adapter, vendor):
        """maybe_get_vit_flash_attn_backend is replaced on non-GCU vendor paths."""
        module, stubs = load_adapter(vendor)
        replaced = stubs.mm.maybe_get_vit_flash_attn_backend
        assert replaced is not None
        assert callable(replaced)
        assert replaced.__name__ == "_patched_maybe_get_vit_flash_attn_backend"

    def test_selector_flash_backend_returns_varlen_func(self, load_adapter,
                                                         monkeypatch):
        """The replaced backend selector maps FLASH_ATTN to a varlen callable."""
        module, stubs = load_adapter("hygon")
        enum = stubs.registry.AttentionBackendEnum

        fake_varlen = object()
        fake_pkg = ModuleType("vllm.vllm_flash_attn")
        fake_pkg.flash_attn_varlen_func = fake_varlen
        monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", fake_pkg)
        monkeypatch.setitem(sys.modules, "vllm", sys.modules["vllm"])

        selector = stubs.mm.maybe_get_vit_flash_attn_backend
        assert selector(enum.FLASH_ATTN) is fake_varlen
        # The ROCM_AITER_FA branch imports from aiter (absent here): ImportError,
        # matching upstream behaviour, not a silent None.
        with pytest.raises(ImportError):
            selector(enum.ROCM_AITER_FA)
        # Unknown backends map to None.
        assert selector("UNKNOWN") is None

    def test_selector_falls_back_to_flash_attn_package(self, load_adapter,
                                                       monkeypatch):
        """Without vllm.vllm_flash_attn the flash_attn package is used."""
        module, stubs = load_adapter("hygon")
        enum = stubs.registry.AttentionBackendEnum

        fake_varlen = object()
        fake_pkg = ModuleType("flash_attn")
        fake_pkg.flash_attn_varlen_func = fake_varlen
        # Make the vllm.vllm_flash_attn import fail while flash_attn works.
        monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", None)
        monkeypatch.setitem(sys.modules, "flash_attn", fake_pkg)

        selector = stubs.mm.maybe_get_vit_flash_attn_backend
        assert selector(enum.FLASH_ATTN) is fake_varlen
