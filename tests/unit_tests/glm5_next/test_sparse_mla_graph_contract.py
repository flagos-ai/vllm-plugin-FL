# SPDX-License-Identifier: Apache-2.0
"""C5: the portable sparse MLA backend must not advertise graph support while a
non-capturable Torch fallback is reachable, and must select its implementation
once instead of retrying after a kernel may have written the KV cache."""

from types import SimpleNamespace

import torch
import pytest

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


def test_both_declarations_are_never():
    assert (
        mla_sparse.FlagGemsSparseMLABackend._cudagraph_support
        is AttentionCGSupport.NEVER
    )
    assert (
        mla_sparse.FlagGemsSparseMLAMetadataBuilder._cudagraph_support
        is AttentionCGSupport.NEVER
    )


def test_capability_never_without_kernels(monkeypatch):
    monkeypatch.setattr(mla_sparse, "_flag_op", lambda *a, **k: None)
    support, reason = mla_sparse.sparse_mla_cudagraph_support(576)
    assert support is AttentionCGSupport.NEVER
    assert "fallback" in reason


def test_capability_never_when_opt_in_disabled(monkeypatch):
    monkeypatch.setattr(
        mla_sparse, "_resolve_sparse_mla_kernels", lambda: (object(), object())
    )
    monkeypatch.delenv("VLLM_FL_GLM5_MLA_SPARSE_GRAPH", raising=False)
    support, reason = mla_sparse.sparse_mla_cudagraph_support(576)
    assert support is AttentionCGSupport.NEVER
    assert "not enabled" in reason


def test_capability_only_after_successful_probe(monkeypatch):
    monkeypatch.setattr(
        mla_sparse, "_resolve_sparse_mla_kernels", lambda: (object(), object())
    )
    monkeypatch.setenv("VLLM_FL_GLM5_MLA_SPARSE_GRAPH", "1")

    monkeypatch.setattr(mla_sparse, "_probe_graph_capture", lambda hs: (True, None))
    support, reason = mla_sparse.sparse_mla_cudagraph_support(576)
    assert support is AttentionCGSupport.UNIFORM_BATCH
    assert reason is None

    monkeypatch.setattr(
        mla_sparse, "_probe_graph_capture", lambda hs: (False, "boom")
    )
    support, reason = mla_sparse.sparse_mla_cudagraph_support(576)
    assert support is AttentionCGSupport.NEVER
    assert "boom" in reason


def test_builder_and_backend_delegate_to_capability(monkeypatch):
    monkeypatch.setattr(
        mla_sparse,
        "sparse_mla_cudagraph_support",
        lambda hs: (AttentionCGSupport.UNIFORM_BATCH, None),
    )
    spec = SimpleNamespace(head_size=576)
    assert (
        mla_sparse.FlagGemsSparseMLAMetadataBuilder.get_cudagraph_support(None, spec)
        is AttentionCGSupport.UNIFORM_BATCH
    )
    assert (
        mla_sparse.FlagGemsSparseMLABackend.get_cudagraph_support(None, spec)
        is AttentionCGSupport.UNIFORM_BATCH
    )


def test_fallback_forbidden_under_capture(monkeypatch):
    monkeypatch.setattr(mla_sparse, "_flag_op", lambda *a, **k: None)
    impl = _make_impl()
    assert impl._capturable is False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="cannot run under CUDA graph"):
        impl._forbid_fallback_under_capture()


def test_eager_fallback_does_not_query_an_unavailable_cuda_runtime(monkeypatch):
    monkeypatch.setattr(mla_sparse, "_flag_op", lambda *a, **k: None)
    impl = _make_impl()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def unexpected_cuda_query():
        raise AssertionError("CUDA is unavailable on this platform")

    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", unexpected_cuda_query
    )
    cache = torch.zeros(1, 1, 1, 12, dtype=torch.bfloat16)
    impl.do_kv_cache_update(
        torch.ones(1, 8, dtype=torch.bfloat16),
        torch.ones(1, 1, 4, dtype=torch.bfloat16),
        cache,
        torch.tensor([0]),
        "auto",
        torch.ones(1),
    )
    assert torch.equal(cache, torch.ones_like(cache))


def test_implementation_selected_once_at_init(monkeypatch):
    cache_fn = object()
    sparse_fn = object()

    def fake_flag_op(module, name):
        return {
            "concat_and_cache_mla": cache_fn,
            "flashmla_sparse": sparse_fn,
        }.get(module)

    monkeypatch.setattr(mla_sparse, "_flag_op", fake_flag_op)
    impl = _make_impl()
    assert impl._cache_writer is cache_fn
    assert impl._sparse_attn is sparse_fn


def test_torch_fallback_used_only_when_kernel_absent(monkeypatch):
    monkeypatch.setattr(mla_sparse, "_flag_op", lambda *a, **k: None)
    impl = _make_impl()

    kv_cache = torch.zeros(4, 2, 1, 12, dtype=torch.bfloat16)
    kv_c = torch.arange(3 * 8, dtype=torch.bfloat16).reshape(3, 8)
    k_pe = torch.arange(3 * 1 * 4, dtype=torch.bfloat16).reshape(3, 1, 4)
    slots = torch.tensor([0, 2, 5], dtype=torch.int64)

    impl.do_kv_cache_update(
        kv_c, k_pe, kv_cache, slots, "auto", torch.ones(1, dtype=torch.float32)
    )
    written = kv_cache.view(-1, 12)[slots]
    assert torch.equal(written, torch.cat((kv_c, k_pe.squeeze(1)), dim=-1))


def test_failure_after_cache_write_propagates_without_second_impl(monkeypatch):
    cache_writes: list[int] = []

    def writer(kv_c, k_pe, kv_cache, slots, *, kv_cache_dtype, scale):
        cache_writes.append(1)
        kv_cache.view(-1, kv_cache.shape[-1])[slots] = 1.0
        raise RuntimeError("kernel failed after writing")

    def fake_flag_op(module, name):
        if module == "concat_and_cache_mla":
            return writer
        return object()

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


def test_unsupported_layout_raises_before_write(monkeypatch):
    monkeypatch.setattr(mla_sparse, "_flag_op", lambda *a, **k: None)
    impl = _make_impl()

    kv_cache = torch.zeros(4, 2, 1, 12, dtype=torch.bfloat16)
    kv_c = torch.zeros(3, 8, dtype=torch.bfloat16)
    k_pe = torch.zeros(3, 1, 4, dtype=torch.bfloat16)
    slots = torch.tensor([0, 2, 5], dtype=torch.int64)

    with pytest.raises(NotImplementedError):
        impl.do_kv_cache_update(
            kv_c, k_pe, kv_cache, slots, "fp8", torch.ones(1, dtype=torch.float32)
        )
    assert kv_cache.count_nonzero() == 0
