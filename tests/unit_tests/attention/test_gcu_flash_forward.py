# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU tests for the remaining ``vllm_fl/attention/utils.py`` lines.

``test_mm_encoder_attention_adapters.py`` (which must not be modified) covers
the adapter wiring: the hygon SDPA patch numerics, the gcu forward delegation
when the backend is not FLASH_ATTN, the kunlunxin chunked SDPA, the nvidia
``forward_oot`` binding and the vit selector's primary
``vllm.vllm_flash_attn`` import path.  Still uncovered after that file runs:

* the FLASH_ATTN main path of ``_gcu_forward_mm_attention`` — the
  ``flash_attn.vllm_flash_attn`` import, the cu_seqlens/max_seqlen pairing
  assert, the automatic cu_seqlens range construction (``cu_seqlens=None``)
  and the explicit-cu_seqlens branch, the flash-attention call itself and
  both output reshape forms (4D passthrough vs flattened 3D);
* the vit selector's ``flash_attn`` fallback import when
  ``vllm.vllm_flash_attn`` is unimportable, and the ROCM_AITER_FA branch
  returning the aiter provider's function.

The fake ``flash_attn_varlen_func`` computes exact per-sequence attention
with an explicit float64 softmax oracle, so the gcu forward's reshape
behaviour is verified against real numbers, not just call recording.
"""

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/attention/utils.py"
_MODULE_NAME = "supp_gcu_flash_forward_under_test"



def _leaf(modules, name):
    """Create (or fetch) a module and its parents inside *modules*."""
    parts = name.split(".")
    for length in range(1, len(parts) + 1):
        partial = ".".join(parts[:length])
        if partial not in modules:
            module = ModuleType(partial)
            module.__path__ = []
            modules[partial] = module
    for length in range(1, len(parts)):
        parent, attribute = ".".join(parts[:length]), parts[length]
        setattr(modules[parent], attribute, modules[".".join(parts[: length + 1])])
    return modules[name]


def _install_stubs(monkeypatch, vendor_name):
    """Build the stubbed vllm tree plus the flash_attn provider double.

    Returns a SimpleNamespace with the pieces the tests steer: the fake
    ``flash_attn_varlen_func``, its captured calls, the recording logger and
    the registry enum.
    """
    modules = {}
    for name in (
        "vllm.logger",
        "vllm.platforms",
        "vllm.utils.gpu_sync_debug",
        "vllm.v1.attention.backends.registry",
        "vllm.v1.attention.ops.vit_attn_wrappers",
        "vllm.model_executor.layers.attention.mm_encoder_attention",
    ):
        _leaf(modules, name)

    logged = []

    class _Logger:
        def info_once(self, msg, *args):
            logged.append(msg % args if args else msg)

        def info(self, msg, *args):
            logged.append(msg % args if args else msg)

    modules["vllm.logger"].init_logger = lambda name: _Logger()

    gpu_sync = modules["vllm.utils.gpu_sync_debug"]
    gpu_sync.gpu_sync_allowed = nullcontext

    class AttentionBackendEnum:
        FLASH_ATTN = "FLASH_ATTN"
        ROCM_AITER_FA = "ROCM_AITER_FA"

    modules[
        "vllm.v1.attention.backends.registry"
    ].AttentionBackendEnum = AttentionBackendEnum

    class MMEncoderAttention:
        def forward_native(self, *args, **kwargs):
            raise AssertionError("placeholder must be replaced")

        def forward_oot(self, *args, **kwargs):
            raise AssertionError("placeholder must be replaced")

        def forward_cuda(self, *args, **kwargs):
            raise AssertionError("placeholder must be replaced")

    mm = modules["vllm.model_executor.layers.attention.mm_encoder_attention"]
    mm.MMEncoderAttention = MMEncoderAttention
    mm.maybe_get_vit_flash_attn_backend = lambda backend: None

    platforms = modules["vllm.platforms"]
    platforms.current_platform = SimpleNamespace(
        vendor_name=vendor_name, is_cuda=lambda: False
    )

    # The Enflame provider import inside _gcu_forward_mm_attention.
    captured = []

    def fake_flash_varlen(
        q,
        k,
        v,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        dropout_p=None,
        causal=None,
        softmax_scale=None,
        fa_version=None,
        **kwargs,
    ):
        captured.append(
            {
                "cu_q": cu_seqlens_q.tolist(),
                "cu_k": cu_seqlens_k.tolist(),
                "max_q": max_seqlen_q,
                "max_k": max_seqlen_k,
                "dropout_p": dropout_p,
                "causal": causal,
                "scale": softmax_scale,
                "fa_version": fa_version,
                "q_shape": tuple(q.shape),
                "k_shape": tuple(k.shape),
            }
        )
        return _varlen_sdpa_oracle(
            q, k, v, cu_seqlens_q.tolist(), cu_seqlens_k.tolist(), softmax_scale
        )

    flash_pkg = _leaf(modules, "flash_attn.vllm_flash_attn")
    flash_pkg.flash_attn_varlen_func = fake_flash_varlen

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        captured=captured,
        logged=logged,
        enum=AttentionBackendEnum,
        mm=mm,
    )


def _varlen_sdpa_oracle(q, k, v, cu_q, cu_k, scale):
    """Exact per-sequence attention: explicit float64 softmax per head."""
    outs = []
    for i in range(len(cu_q) - 1):
        qs = q[cu_q[i] : cu_q[i + 1]].double()
        ks = k[cu_k[i] : cu_k[i + 1]].double()
        vs = v[cu_k[i] : cu_k[i + 1]].double()
        scores = torch.einsum("shd,thd->hst", qs, ks) * scale
        weights = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hst,thd->shd", weights, vs))
    return torch.cat(outs, dim=0).to(q.dtype)


@pytest.fixture
def load_module(monkeypatch):
    """Load a fresh copy of vllm_fl/attention/utils.py and apply the patch."""

    def load(vendor_name):
        stubs = _install_stubs(monkeypatch, vendor_name)
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, _MODULE_NAME, module)
        spec.loader.exec_module(module)
        module.patch_mm_encoder_attention()
        return module, stubs

    return load


class _FakeMmAttention:
    """Self double exposing exactly the attributes the gcu forward reads."""

    attn_backend = "FLASH_ATTN"
    scale = 0.25
    num_heads = 2
    num_kv_heads = 1
    head_size = 8

    def view_qkv_to_4d(self, q, k, v, batch_size, q_len, kv_len):
        def to4(t, heads):
            if t.dim() == 4:
                return t
            return t.reshape(*t.shape[:2], heads, t.shape[-1] // heads)

        return (
            to4(q, self.num_heads),
            to4(k, self.num_kv_heads),
            to4(v, self.num_kv_heads),
        )


class TestGcuFlashForwardPath:
    """``_gcu_forward_mm_attention`` with a FLASH_ATTN backend."""

    def test_without_cu_seqlens_builds_token_ranges(self, load_module):
        """cu_seqlens=None builds q/k ranges from the batch shape and passes
        q_len/kv_len as the max sequence lengths; the output matches the
        per-sequence oracle in 4D layout."""
        module, stubs = load_module("gcu")
        generator = torch.Generator().manual_seed(11)
        b, s, h, hkv, d, kv = 2, 2, 2, 1, 8, 3
        query = torch.randn(b, s, h, d, generator=generator)
        key = torch.randn(b, kv, hkv, d, generator=generator)
        value = torch.randn(b, kv, hkv, d, generator=generator)

        out = module._gcu_forward_mm_attention(_FakeMmAttention(), query, key, value)

        call = stubs.captured[0]
        assert call["cu_q"] == [0, 2, 4]  # arange(0, (b+1)*s, s)
        assert call["cu_k"] == [0, 3, 6]  # arange(0, (b+1)*kv, kv)
        assert call["max_q"] == s and call["max_k"] == kv
        assert call["dropout_p"] == 0.0 and call["causal"] is False
        assert call["scale"] == 0.25 and call["fa_version"] == 2
        assert call["q_shape"] == (4, h, d) and call["k_shape"] == (6, hkv, d)

        expected = _varlen_sdpa_oracle(
            query.flatten(0, 1),
            key.flatten(0, 1),
            value.flatten(0, 1),
            [0, 2, 4],
            [0, 3, 6],
            0.25,
        ).reshape(b, s, h, d)
        assert out.shape == (b, s, h, d)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_with_explicit_cu_seqlens_uses_them_for_q_and_k(self, load_module):
        """cu_seqlens given: both cu_seqlens_q and _k alias it and the max
        lengths come from max_seqlen.item() under gpu_sync_allowed."""
        module, stubs = load_module("gcu")
        generator = torch.Generator().manual_seed(23)
        query = torch.randn(1, 3, 2, 8, generator=generator)
        key = torch.randn(1, 3, 1, 8, generator=generator)
        value = torch.randn(1, 3, 1, 8, generator=generator)
        cu = torch.tensor([0, 2, 3], dtype=torch.int32)

        out = module._gcu_forward_mm_attention(
            _FakeMmAttention(),
            query,
            key,
            value,
            cu_seqlens=cu,
            max_seqlen=torch.tensor(2),
        )

        call = stubs.captured[0]
        assert call["cu_q"] == [0, 2, 3] and call["cu_k"] == [0, 2, 3]
        assert call["max_q"] == 2 and call["max_k"] == 2

        expected = _varlen_sdpa_oracle(
            query.flatten(0, 1),
            key.flatten(0, 1),
            value.flatten(0, 1),
            [0, 2, 3],
            [0, 2, 3],
            0.25,
        ).reshape(1, 3, 2, 8)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_reshaped_input_returns_flattened_output(self, load_module):
        """A 3D [batch, seq, heads*dim] input is viewed to 4D, run through
        flash attention and returned flattened again."""
        module, stubs = load_module("gcu")
        generator = torch.Generator().manual_seed(37)
        b, s, h, hkv, d, kv = 2, 2, 2, 1, 8, 3
        query = torch.randn(b, s, h * d, generator=generator)
        key = torch.randn(b, kv, hkv * d, generator=generator)
        value = torch.randn(b, kv, hkv * d, generator=generator)

        out = module._gcu_forward_mm_attention(_FakeMmAttention(), query, key, value)

        expected = _varlen_sdpa_oracle(
            query.reshape(b, s, h, d).flatten(0, 1),
            key.reshape(b, kv, hkv, d).flatten(0, 1),
            value.reshape(b, kv, hkv, d).flatten(0, 1),
            [0, 2, 4],
            [0, 3, 6],
            0.25,
        ).reshape(b, s, h * d)
        assert out.shape == (b, s, h * d)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        assert len(stubs.captured) == 1

    def test_mismatched_cu_seqlens_pairing_rejected(self, load_module):
        """cu_seqlens and max_seqlen must be both set or both None."""
        module, _ = load_module("gcu")
        query = torch.randn(1, 3, 2, 8)
        with pytest.raises(AssertionError, match="both set or both None"):
            module._gcu_forward_mm_attention(
                _FakeMmAttention(),
                query,
                torch.randn(1, 3, 1, 8),
                torch.randn(1, 3, 1, 8),
                cu_seqlens=None,
                max_seqlen=torch.tensor(2),
            )


class TestVitSelectorFallbacks:
    """The patched vit backend selector's fallback branches."""

    def test_flash_attn_fallback_when_vllm_flash_attn_missing(
        self, load_module, monkeypatch
    ):
        """An unimportable vllm.vllm_flash_attn falls back to the flash_attn
        package and logs the fallback message."""
        module, stubs = load_module("nvidia")
        # A None entry makes ``from vllm.vllm_flash_attn import ...`` raise.
        monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", None)

        sentinel = object()
        flash_pkg = ModuleType("flash_attn")
        flash_pkg.flash_attn_varlen_func = sentinel
        monkeypatch.setitem(sys.modules, "flash_attn", flash_pkg)

        selector = stubs.mm.maybe_get_vit_flash_attn_backend
        assert selector(stubs.enum.FLASH_ATTN) is sentinel
        assert "Using flash_attn for vit attention" in stubs.logged

    def test_rocm_aiter_branch_returns_aiter_function(self, load_module, monkeypatch):
        """The ROCM_AITER_FA branch returns the aiter provider's function."""
        module, stubs = load_module("nvidia")

        sentinel = object()
        aiter_pkg = ModuleType("aiter")
        aiter_pkg.flash_attn_varlen_func = sentinel
        monkeypatch.setitem(sys.modules, "aiter", aiter_pkg)

        selector = stubs.mm.maybe_get_vit_flash_attn_backend
        assert selector(stubs.enum.ROCM_AITER_FA) is sentinel
