# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA remains an eager boundary and never retries a state writer.

Public library operators are explicitly stubbed for these dispatch contracts.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import AttentionCGSupport

from vllm_fl.dispatch.backends.flaggems.impl import mla_sparse


def _make_impl(topk_tokens: int = 16) -> "mla_sparse.FlagGemsSparseMLAImpl":
    return mla_sparse.FlagGemsSparseMLAImpl(
        num_heads=2,
        head_size=8,
        scale=0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        topk_indices_buffer=torch.zeros(topk_tokens, dtype=torch.int32),
        indexer=None,
        kv_lora_rank=8,
    )


def test_sparse_mla_is_an_eager_boundary():
    spec = SimpleNamespace(head_size=576)
    for cls in (
        mla_sparse.FlagGemsSparseMLABackend,
        mla_sparse.FlagGemsSparseMLAMetadataBuilder,
    ):
        assert cls._cudagraph_support is AttentionCGSupport.NEVER
        assert cls.get_cudagraph_support(None, spec) is AttentionCGSupport.NEVER


def test_missing_library_fails_before_execution(monkeypatch):
    monkeypatch.setattr(mla_sparse, "_flag_op", lambda *a: None)
    with pytest.raises(RuntimeError, match="requires FlagGems-vllm"):
        _make_impl()


def test_failure_after_cache_write_propagates_without_second_impl(monkeypatch):
    cache_writes: list[int] = []

    def writer(kv_c, k_pe, kv_cache, slots, *, kv_cache_dtype, scale):
        cache_writes.append(1)
        kv_cache.view(-1, kv_cache.shape[-1])[slots] = 1.0
        raise RuntimeError("kernel failed after writing")

    def fake_flag_op(module, name):
        if module == "concat_and_cache_mla":
            return writer
        return lambda *a, **k: None

    monkeypatch.setattr(mla_sparse, "_flag_op", fake_flag_op)
    impl = _make_impl()

    kv_cache = torch.zeros(4, 2, 1, 12, dtype=torch.bfloat16)
    kv_c = torch.zeros(3, 8, dtype=torch.bfloat16)
    k_pe = torch.zeros(3, 1, 4, dtype=torch.bfloat16)
    slots = torch.tensor([0, 2, 5], dtype=torch.int64)

    with pytest.raises(RuntimeError, match="after writing"):
        impl.do_kv_cache_update(
            kv_c, k_pe, kv_cache, slots, "auto", torch.ones(1, dtype=torch.float32)
        )
    assert cache_writes == [1]
