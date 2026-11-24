# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""G3 coverage: vllm_fl.attention.utils vendor attention adapters (814b).

Exercises the Enflame (GCU) varlen adapter and the Kunlunxin query-chunked
Torch SDPA adapter. All external modules (flash_attn, vllm sync helper, the
vLLM mm-encoder module) and the current platform are replaced with fakes
through ``sys.modules`` / attribute injection; tensor work is CPU-only.
"""

from __future__ import annotations

import contextlib
import sys
import types
from types import SimpleNamespace as NS

import pytest
import torch

from vllm.v1.attention.backends.registry import AttentionBackendEnum

import vllm_fl.attention.utils as attn_utils


# ---------------------------------------------------------------------------
# Fake wiring helpers
# ---------------------------------------------------------------------------


def _fake_mm_module():
    module = types.ModuleType(
        "vllm.model_executor.layers.attention.mm_encoder_attention"
    )

    class MMEncoderAttention:
        def forward_native(self, *args, **kwargs):  # pragma: no cover - stub
            return "native"

        def forward_cuda(self, *args, **kwargs):  # pragma: no cover - stub
            return "cuda"

    module.MMEncoderAttention = MMEncoderAttention
    module.maybe_get_vit_flash_attn_backend = lambda backend: None
    return module


def _install_patch(monkeypatch, vendor, is_cuda):
    mm_mod = _fake_mm_module()
    monkeypatch.setitem(sys.modules, mm_mod.__name__, mm_mod)

    import vllm.platforms as vllm_platforms

    monkeypatch.setattr(
        vllm_platforms,
        "current_platform",
        NS(vendor_name=vendor, is_cuda=lambda: is_cuda),
        raising=False,
    )
    attn_utils.patch_mm_encoder_attention()
    return mm_mod


def _install_flash_attn(monkeypatch):
    calls = []
    parent = types.ModuleType("flash_attn")
    child = types.ModuleType("flash_attn.vllm_flash_attn")

    def fake_flash_attn_varlen_func(**kwargs):
        calls.append(kwargs)
        return torch.zeros_like(kwargs["q"])

    child.flash_attn_varlen_func = fake_flash_attn_varlen_func
    parent.vllm_flash_attn = child
    monkeypatch.setitem(sys.modules, "flash_attn", parent)
    monkeypatch.setitem(sys.modules, "flash_attn.vllm_flash_attn", child)
    return calls


def _stub_sdpa(monkeypatch):
    """Keep the chunked-SDPA math CPU/version agnostic."""

    def fake_sdpa(q, k, v, dropout_p=0.0, scale=None, enable_gqa=False):
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)


def _install_gpu_sync(monkeypatch):
    module = types.ModuleType("vllm.utils.gpu_sync_debug")

    @contextlib.contextmanager
    def gpu_sync_allowed():
        yield

    module.gpu_sync_allowed = gpu_sync_allowed
    monkeypatch.setitem(sys.modules, "vllm.utils.gpu_sync_debug", module)
    return module


class _GcuSelf:
    scale = 0.5
    num_heads = 2
    head_size = 4

    @staticmethod
    def view_qkv_to_4d(query, key, value, batch_size, q_len, kv_len):
        return (
            query.view(batch_size, q_len, 2, 4),
            key.view(batch_size, kv_len, 2, 4),
            value.view(batch_size, kv_len, 2, 4),
        )


class _KunlunxinSelf:
    scale = 0.5
    num_heads = 2
    num_kv_heads = 1

    @staticmethod
    def view_qkv_to_4d(query, key, value, batch_size, q_len, kv_len):
        return (
            query.view(batch_size, q_len, 2, 4),
            key.view(batch_size, kv_len, 2, 4),
            value.view(batch_size, kv_len, 2, 4),
        )


# ---------------------------------------------------------------------------
# Enflame (GCU) varlen adapter
# ---------------------------------------------------------------------------


def test_gcu_forward_delegates_for_non_flash_backend(monkeypatch):
    _install_gpu_sync(monkeypatch)
    calls = []

    class Self(_GcuSelf):
        attn_backend = AttentionBackendEnum.TORCH_SDPA

        def _fl_gcu_original_forward_native(self, *args):
            calls.append(args)
            return "original"

    query = torch.zeros(1, 2, 8)
    out = attn_utils._gcu_forward_mm_attention(
        Self(), query, query, query, None, None, None
    )
    assert out == "original"
    assert calls == [(query, query, query, None, None, None)]


def test_gcu_forward_flash_builds_dense_seqlens(monkeypatch):
    calls = _install_flash_attn(monkeypatch)
    _install_gpu_sync(monkeypatch)

    class Self(_GcuSelf):
        attn_backend = AttentionBackendEnum.FLASH_ATTN

    query = torch.zeros(2, 3, 8)
    key = torch.zeros(2, 5, 8)
    value = torch.zeros(2, 5, 8)

    out = attn_utils._gcu_forward_mm_attention(Self(), query, key, value)

    assert out.shape == (2, 3, 8)
    assert len(calls) == 1
    kwargs = calls[0]
    # Dense path infers per-batch seqlens from the tensor shapes.
    assert kwargs["max_seqlen_q"] == 3
    assert kwargs["max_seqlen_k"] == 5
    assert kwargs["cu_seqlens_q"].tolist() == [0, 3, 6]
    assert kwargs["cu_seqlens_k"].tolist() == [0, 5, 10]


def test_gcu_forward_flash_uses_cu_seqlens_and_sync_guard(monkeypatch):
    calls = _install_flash_attn(monkeypatch)
    _install_gpu_sync(monkeypatch)

    class Self(_GcuSelf):
        attn_backend = AttentionBackendEnum.FLASH_ATTN

    query = torch.zeros(2, 3, 8)
    key = torch.zeros(2, 3, 8)
    value = torch.zeros(2, 3, 8)
    cu_seqlens = torch.tensor([0, 2, 6])
    max_seqlen = torch.tensor(3)

    out = attn_utils._gcu_forward_mm_attention(
        Self(), query, key, value, cu_seqlens, max_seqlen
    )

    assert out.shape == (2, 3, 8)
    kwargs = calls[0]
    assert kwargs["cu_seqlens_q"] is cu_seqlens
    assert kwargs["cu_seqlens_k"] is cu_seqlens
    assert kwargs["max_seqlen_q"] == 3
    assert kwargs["max_seqlen_k"] == 3


# ---------------------------------------------------------------------------
# Kunlunxin query-chunked SDPA adapter
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# maybe_get_vit_flash_attn_backend override
# ---------------------------------------------------------------------------






