# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for vendor multimodal-encoder attention patches.

Only torch CPU tensors and stub vLLM modules are used; no accelerator or
weight is touched.  Each test installs the exact modules the patched code
imports, then validates the resulting numerics/layout against independent
Torch SDPA references.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as torch_f


def _find_source(rel: str) -> pathlib.Path:
    override = os.environ.get("VLLM_FL_SRC")
    candidates = []
    if override:
        candidates.append(pathlib.Path(override) / rel)
    candidates.append(pathlib.Path("/tmp/flsrc") / rel)
    for parent in pathlib.Path(__file__).resolve().parents:
        candidates.append(parent / rel)
    candidates.append(pathlib.Path.cwd() / rel)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not locate {rel}")


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def info_once(self, *args, **kwargs):
        self.messages.append(("info_once", args))

    def info(self, *args, **kwargs):
        self.messages.append(("info", args))

    def debug(self, *args, **kwargs):
        self.messages.append(("debug", args))

    def warning(self, *args, **kwargs):
        self.messages.append(("warning", args))


@pytest.fixture(scope="module")
def attn_utils():
    saved = {name: sys.modules.get(name) for name in ("vllm", "vllm.logger")}
    if "vllm" not in sys.modules or saved["vllm"] is None:
        pkg = types.ModuleType("vllm")
        pkg.__path__ = []
        sys.modules["vllm"] = pkg
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda name: _FakeLogger()
    sys.modules["vllm.logger"] = logger_mod
    try:
        path = _find_source("vllm_fl/attention/utils.py")
        spec = importlib.util.spec_from_file_location("w8b_attention_utils", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _ensure(monkeypatch, name):
    existing = sys.modules.get(name)
    if isinstance(existing, types.ModuleType):
        return existing
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    return module


def _put(monkeypatch, name, module):
    if "." in name:
        parent_name, leaf = name.rsplit(".", 1)
        parent = _ensure(monkeypatch, parent_name)
        monkeypatch.setattr(parent, leaf, module, raising=False)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _stub(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return _put(monkeypatch, name, module)


def _enum():
    return SimpleNamespace(FLASH_ATTN=object(), ROCM_AITER_FA=object())


def _platform(monkeypatch, vendor, is_cuda=False):
    return _stub(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(
            vendor_name=vendor, is_cuda=lambda: is_cuda
        ),
    )


def _fake_f():
    def sdpa(q, k, v, dropout_p=0.0, scale=None, enable_gqa=False):
        if enable_gqa and k.shape[1] != q.shape[1]:
            repeats = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        return torch_f.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, scale=scale
        )

    return SimpleNamespace(pad=torch_f.pad, scaled_dot_product_attention=sdpa)


def _rearrange(x, pattern):
    pattern = pattern.strip()
    if pattern == "b s h d -> b h s d":
        return x.permute(0, 2, 1, 3)
    if pattern == "b h s d -> b s h d":
        return x.permute(0, 2, 1, 3)
    raise AssertionError(f"unexpected pattern {pattern!r}")


def _vit_module(fn):
    return SimpleNamespace(
        apply_sdpa=fn,
        einops=SimpleNamespace(rearrange=_rearrange),
        F=_fake_f(),
    )


def test_hygon_patch_pads_d72_and_restores_layout(attn_utils, monkeypatch):
    captured = {}

    def original(q, k, v, scale=None, enable_gqa=False):
        raise AssertionError("original apply_sdpa must be replaced")

    vit = _vit_module(original)
    _put(monkeypatch, "vllm.v1.attention.ops.vit_attn_wrappers", vit)
    attn_utils._patch_hygon_sdpa_contiguous()
    patched = vit.apply_sdpa
    assert patched is not original
    assert patched._fl_hygon_sdpa_layout == "bhsd-pad-d72"

    q = torch.randn(2, 3, 2, 72)
    k = torch.randn(2, 3, 2, 72)
    v = torch.randn(2, 3, 2, 72)
    out = patched(q, k, v, scale=None, enable_gqa=True)
    assert out.shape == (2, 3, 2, 72)
    # scale=None defaulted to 72**-0.5 with D72 padding.
    ref = torch_f.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
    ).permute(0, 2, 1, 3)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    # Non-72 head dims skip padding but still use contiguous BHSD layout.
    q2 = torch.randn(1, 2, 2, 64)
    out2 = patched(q2, q2, q2, scale=0.5, enable_gqa=False)
    assert out2.shape == (1, 2, 2, 64)


def test_hygon_patch_is_idempotent(attn_utils, monkeypatch):
    def original(q, k, v, scale=None, enable_gqa=False):
        return None

    original._fl_hygon_sdpa_layout = "bhsd-pad-d72"
    vit = _vit_module(original)
    _put(monkeypatch, "vllm.v1.attention.ops.vit_attn_wrappers", vit)
    attn_utils._patch_hygon_sdpa_contiguous()
    assert vit.apply_sdpa is original


def _mm_module():
    class MMEncoderAttention:
        def forward_native(self, *args, **kwargs):
            return "native"

        def forward_cuda(self, *args, **kwargs):
            return "cuda"

        def forward_oot(self, *args, **kwargs):
            return "oot"

        def view_qkv_to_4d(self, query, key, value, batch_size, q_len, kv_len):
            scope = getattr(self, "_heads", None)
            assert scope is not None
            num_heads, head_size = scope
            return (
                query.reshape(batch_size, q_len, num_heads, head_size),
                key.reshape(batch_size, kv_len, num_heads, head_size),
                value.reshape(batch_size, kv_len, num_heads, head_size),
            )

    return SimpleNamespace(MMEncoderAttention=MMEncoderAttention)


def test_gcu_patch_installs_both_entrypoints(attn_utils, monkeypatch):
    _platform(monkeypatch, "gcu")
    _stub(monkeypatch, "vllm.v1.attention.backends.registry",
          AttentionBackendEnum=_enum())
    mm = _mm_module()
    _put(
        monkeypatch,
        "vllm.model_executor.layers.attention.mm_encoder_attention",
        mm,
    )
    attn_utils.patch_mm_encoder_attention()
    cls = mm.MMEncoderAttention
    assert cls._fl_gcu_original_forward_native is not None
    assert cls.forward_native is attn_utils._gcu_forward_mm_attention
    assert cls.forward_oot is attn_utils._gcu_forward_mm_attention


def _gcu_self(attn_utils, backend, heads=(2, 4), fallback=None):
    return SimpleNamespace(
        attn_backend=backend,
        _fl_gcu_original_forward_native=fallback,
        _heads=heads,
        scale=0.5,
        num_heads=heads[0],
        head_size=heads[1],
        view_qkv_to_4d=lambda q, k, v, b, ql, kl: (
            q.reshape(b, ql, heads[0], heads[1]),
            k.reshape(b, kl, heads[0], heads[1]),
            v.reshape(b, kl, heads[0], heads[1]),
        ),
    )


def _install_gcu_deps(monkeypatch, fa):
    enum = _enum()
    _stub(monkeypatch, "vllm.v1.attention.backends.registry",
          AttentionBackendEnum=enum)
    _stub(
        monkeypatch,
        "vllm.utils.gpu_sync_debug",
        gpu_sync_allowed=lambda: contextlib.nullcontext(),
    )
    _stub(monkeypatch, "flash_attn.vllm_flash_attn", flash_attn_varlen_func=fa)
    return enum


def test_gcu_attention_falls_back_for_other_backends(attn_utils, monkeypatch):
    enum = _install_gcu_deps(monkeypatch, lambda **kwargs: None)
    self = _gcu_self(
        attn_utils,
        SimpleNamespace(),
        fallback=lambda *a: ("fallback", a[:3]),
    )
    result = attn_utils._gcu_forward_mm_attention(self, "q", "k", "v")
    assert result[0] == "fallback"


def test_gcu_attention_uses_varlen_without_cu_seqlens(attn_utils, monkeypatch):
    calls = {}

    def fa(**kwargs):
        calls.update(kwargs)
        q = kwargs["q"]
        return torch.zeros(q.shape[0], 2, 4)

    enum = _install_gcu_deps(monkeypatch, fa)
    self = _gcu_self(attn_utils, enum.FLASH_ATTN)
    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 5, 8)
    out = attn_utils._gcu_forward_mm_attention(self, query, key, key)
    # Reshaped inputs (3-D) return flattened 2-D per-token features.
    assert out.shape == (2, 3, 8)
    assert calls["max_seqlen_q"] == 3 and calls["max_seqlen_k"] == 5
    assert calls["cu_seqlens_q"].tolist() == [0, 3, 6]
    assert calls["cu_seqlens_k"].tolist() == [0, 5, 10]
    assert calls["fa_version"] == 2 and calls["causal"] is False


def test_gcu_attention_uses_supplied_cu_seqlens(attn_utils, monkeypatch):
    def fa(**kwargs):
        q = kwargs["q"]
        return torch.zeros(q.shape[0], 2, 4)

    enum = _install_gcu_deps(monkeypatch, fa)
    self = _gcu_self(attn_utils, enum.FLASH_ATTN)
    query = torch.randn(1, 4, 2, 4)
    key = torch.randn(1, 4, 2, 4)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    out = attn_utils._gcu_forward_mm_attention(
        self, query, key, key, cu_seqlens=cu, max_seqlen=torch.tensor(2)
    )
    # 4-D input keeps the 4-D output.
    assert out.shape == (1, 4, 2, 4)


def test_gcu_attention_requires_matching_sequence_kwargs(attn_utils, monkeypatch):
    enum = _install_gcu_deps(monkeypatch, lambda **kwargs: None)
    self = _gcu_self(attn_utils, enum.FLASH_ATTN)
    query = torch.randn(1, 2, 2, 4)
    with pytest.raises(AssertionError):
        attn_utils._gcu_forward_mm_attention(
            self, query, query, query, cu_seqlens=torch.tensor([0, 2])
        )


def test_kunlunxin_patch_chunks_queries(attn_utils, monkeypatch):
    _platform(monkeypatch, "kunlunxin")
    _stub(monkeypatch, "vllm.v1.attention.backends.registry",
          AttentionBackendEnum=_enum())
    mm = _mm_module()
    _put(
        monkeypatch,
        "vllm.model_executor.layers.attention.mm_encoder_attention",
        mm,
    )
    monkeypatch.setattr(attn_utils, "_KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE", 2)
    attn_utils.patch_mm_encoder_attention()
    forward = mm.MMEncoderAttention._forward_sdpa

    self = SimpleNamespace(
        scale=0.25,
        num_heads=2,
        num_kv_heads=2,
        _heads=(2, 4),
        view_qkv_to_4d=lambda q, k, v, b, ql, kl: (
            q.reshape(b, ql, 2, 4),
            k.reshape(b, kl, 2, 4),
            v.reshape(b, kl, 2, 4),
        ),
    )
    q = torch.randn(1, 5, 2, 4)
    k = torch.randn(1, 5, 2, 4)
    v = torch.randn(1, 5, 2, 4)
    out = forward(self, q, k, v)
    ref = torch_f.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        scale=0.25,
    ).permute(0, 2, 1, 3)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    # Grouped-query attention enables GQA and reshapes 3-D inputs.
    self_gqa = SimpleNamespace(
        scale=0.5,
        num_heads=2,
        num_kv_heads=1,
        _heads=(2, 4),
        view_qkv_to_4d=lambda q, k, v, b, ql, kl: (
            q.reshape(b, ql, 2, 4),
            k.reshape(b, kl, 1, 4),
            v.reshape(b, kl, 1, 4),
        ),
    )
    q3 = torch.randn(1, 3, 8)
    k3 = torch.randn(1, 3, 4)
    out3 = forward(self_gqa, q3, k3, k3)
    assert out3.shape == (1, 3, 8)


def test_kunlunxin_patch_handles_cu_seqlens(attn_utils, monkeypatch):
    _platform(monkeypatch, "kunlunxin")
    _stub(monkeypatch, "vllm.v1.attention.backends.registry",
          AttentionBackendEnum=_enum())
    mm = _mm_module()
    _put(
        monkeypatch,
        "vllm.model_executor.layers.attention.mm_encoder_attention",
        mm,
    )
    monkeypatch.setattr(attn_utils, "_KUNLUNXIN_SDPA_QUERY_CHUNK_SIZE", 2)
    attn_utils.patch_mm_encoder_attention()
    forward = mm.MMEncoderAttention._forward_sdpa
    self = SimpleNamespace(
        scale=0.3,
        num_heads=2,
        num_kv_heads=2,
        _heads=(2, 4),
        view_qkv_to_4d=lambda q, k, v, b, ql, kl: (
            q.reshape(b, ql, 2, 4),
            k.reshape(b, kl, 2, 4),
            v.reshape(b, kl, 2, 4),
        ),
    )
    q = torch.randn(1, 5, 2, 4)
    k = torch.randn(1, 5, 2, 4)
    v = torch.randn(1, 5, 2, 4)
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    out = forward(self, q, k, v, cu_seqlens=cu)
    assert out.shape == (1, 5, 2, 4)
    # Each segment matches an independent per-segment SDPA reference.
    for start, end in ((0, 2), (2, 5)):
        ref = torch_f.scaled_dot_product_attention(
            q[:, start:end].permute(0, 2, 1, 3),
            k[:, start:end].permute(0, 2, 1, 3),
            v[:, start:end].permute(0, 2, 1, 3),
            scale=0.3,
        ).permute(0, 2, 1, 3)
        torch.testing.assert_close(
            out[:, start:end], ref, atol=1e-5, rtol=1e-5
        )


def test_cuda_platform_binds_forward_oot_to_cuda(attn_utils, monkeypatch):
    _platform(monkeypatch, "nvidia", is_cuda=True)
    _stub(monkeypatch, "vllm.v1.attention.backends.registry",
          AttentionBackendEnum=_enum())
    mm = _mm_module()
    _put(
        monkeypatch,
        "vllm.model_executor.layers.attention.mm_encoder_attention",
        mm,
    )
    attn_utils.patch_mm_encoder_attention()
    assert mm.MMEncoderAttention.forward_oot is mm.MMEncoderAttention.forward_cuda


def _patch_vit(attn_utils, monkeypatch, enum):
    _platform(monkeypatch, "npu")
    _stub(monkeypatch, "vllm.v1.attention.backends.registry",
          AttentionBackendEnum=enum)
    mm = _mm_module()
    _put(
        monkeypatch,
        "vllm.model_executor.layers.attention.mm_encoder_attention",
        mm,
    )
    attn_utils.patch_mm_encoder_attention()
    return mm


def test_vit_backend_prefers_vllm_flash_attn(attn_utils, monkeypatch):
    enum = _enum()
    mm = _patch_vit(attn_utils, monkeypatch, enum)
    sentinel = object()
    _stub(monkeypatch, "vllm.vllm_flash_attn", flash_attn_varlen_func=sentinel)
    assert mm.maybe_get_vit_flash_attn_backend(enum.FLASH_ATTN) is sentinel


def test_vit_backend_falls_back_to_flash_attn(attn_utils, monkeypatch):
    enum = _enum()
    mm = _patch_vit(attn_utils, monkeypatch, enum)
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", None)
    fallback = object()
    _stub(monkeypatch, "flash_attn", flash_attn_varlen_func=fallback)
    assert mm.maybe_get_vit_flash_attn_backend(enum.FLASH_ATTN) is fallback


def test_vit_backend_rocm_and_default(attn_utils, monkeypatch):
    enum = _enum()
    mm = _patch_vit(attn_utils, monkeypatch, enum)
    aiter_fn = object()
    _stub(monkeypatch, "aiter", flash_attn_varlen_func=aiter_fn)
    assert mm.maybe_get_vit_flash_attn_backend(enum.ROCM_AITER_FA) is aiter_fn
    assert mm.maybe_get_vit_flash_attn_backend(object()) is None
