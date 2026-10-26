# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU numerics for the TXDA SDPA adapter; accelerator imports are stubbed."""

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "vllm_fl/dispatch/backends/vendor/txda/impl/attention.py"


def _load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def attention(monkeypatch):
    backend = ModuleType("vllm.v1.attention.backend")
    backend.AttentionType = SimpleNamespace(
        DECODER="decoder",
        ENCODER="encoder",
        ENCODER_ONLY="encoder_only",
        ENCODER_DECODER="encoder_decoder",
    )
    backend.AttentionCGSupport = SimpleNamespace(NEVER="never")
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    shared = ModuleType("vllm_fl.dispatch.backends.flaggems.impl.attention")
    shared.AttentionFLBackend = type("AttentionFLBackend", (), {})
    shared.AttentionFLImpl = type("AttentionFLImpl", (), {})

    class MetadataBuilder:
        def __init__(self, *args):
            pass

        def build(self, prefix, metadata, fast_build=False):
            return SimpleNamespace(prefix=prefix, metadata=metadata, fast=fast_build)

    shared.AttentionFLMetadataBuilder = MetadataBuilder
    monkeypatch.setitem(sys.modules, shared.__name__, shared)
    return _load(monkeypatch, "txda_attention_under_test", SOURCE)


def _impl(module, *, kv_heads=2, window=-1, attn_type="decoder"):
    impl = module.TxdaSDPAAttentionImpl.__new__(module.TxdaSDPAAttentionImpl)
    impl.num_heads = 4
    impl.num_kv_heads = kv_heads
    impl.head_size = 8
    impl.num_queries_per_kv = 4 // kv_heads
    impl.scale = 0.37
    impl.sliding_window = (window, window)
    impl.attn_type = attn_type
    impl.alibi_slopes = None
    impl.logits_soft_cap = 0
    return impl


def _oracle(q, k, v, scale, *, causal=True, window=-1, window_right=None):
    # Explicit softmax/matmul oracle independent of the production SDPA call.
    repeats = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeats, dim=1).double()
    v = v.repeat_interleave(repeats, dim=1).double()
    scores = torch.einsum("qhd,khd->hqk", q.double(), k) * scale
    query_positions = torch.arange(q.shape[0]) + k.shape[0] - q.shape[0]
    key_positions = torch.arange(k.shape[0])
    mask = torch.ones(q.shape[0], k.shape[0], dtype=torch.bool)
    if causal:
        mask &= key_positions[None] <= query_positions[:, None]
    if window >= 0:
        mask &= key_positions[None] >= query_positions[:, None] - window
        if not causal:
            right = window if window_right is None else window_right
            mask &= key_positions[None] <= query_positions[:, None] + right
    probs = scores.masked_fill(~mask, -torch.inf).softmax(-1)
    return torch.einsum("hqk,khd->qhd", probs, v).to(q.dtype)


@pytest.mark.parametrize("query_len,seq_len", [(5, 5), (3, 9), (1, 9)])
@pytest.mark.parametrize("window", [-1, 0, 2])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_sdpa_matches_explicit_attention(
    attention, query_len, seq_len, window, causal, kv_heads
):
    impl = _impl(attention, kv_heads=kv_heads, window=window)
    gen = torch.Generator().manual_seed(220)
    q = torch.randn(query_len, 4, 8, generator=gen)
    k, v = (torch.randn(seq_len, kv_heads, 8, generator=gen) for _ in range(2))
    actual = impl._sdpa(q, k, v, seq_len, window, causal=causal)
    expected = _oracle(q, k, v, impl.scale, causal=causal, window=window)
    assert actual.shape == q.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("window", [-1, 2])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_paged_mixed_batch_ignores_invalid_padding(attention, causal, window, layout):
    impl = _impl(attention, window=window)
    gen = torch.Generator().manual_seed(21)
    block_size = 4
    cache = torch.randn(2, 8, block_size, 2, 8, generator=gen)
    if layout == "HND":
        cache = cache.transpose(2, 3).contiguous().transpose(2, 3)
    query = torch.randn(7, 4, 8, generator=gen)
    # Last two queries and two trailing block entries per request are padding.
    output = torch.full_like(query, 991)
    starts = [0, 3, 4, 5]
    lengths = [7, 9, 1]
    block_table = torch.tensor(
        [[3, 1, 999, 999, 999], [5, 0, 7, 999, 999], [2, 999, 999, 999, 999]]
    )
    metadata = SimpleNamespace(
        num_actual_tokens=5,
        query_start_loc=torch.tensor(starts),
        seq_lens=torch.tensor(lengths),
        block_table=block_table,
        use_cascade=False,
        causal=causal,
    )
    result = impl.forward(None, query, None, None, cache, metadata, output)
    assert result is output
    for request, seq_len in enumerate(lengths):
        pages = block_table[request, : math.ceil(seq_len / block_size)]
        k = cache[0, pages].flatten(0, 1)[:seq_len]
        v = cache[1, pages].flatten(0, 1)[:seq_len]
        lo, hi = starts[request : request + 2]
        expected = _oracle(query[lo:hi], k, v, impl.scale, causal=causal, window=window)
        torch.testing.assert_close(result[lo:hi], expected, rtol=1e-5, atol=1e-6)
    assert torch.all(output[5:] == 991)


def test_cache_update_skips_negative_slots_and_padded_kv(attention):
    impl = _impl(attention)
    cache = torch.full((2, 3, 4, 2, 8), -7.0)
    key = torch.arange(5 * 2 * 8).reshape(5, 2, 8).float()
    value = key + 1000
    slots = torch.tensor([0, -1, 6])
    impl.do_kv_cache_update(None, key, value, cache, slots)
    expected = torch.full_like(cache, -7)
    expected[0, 0, 0] = key[0]
    expected[1, 0, 0] = value[0]
    expected[0, 1, 2] = key[2]
    expected[1, 1, 2] = value[2]
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


@pytest.mark.parametrize("window", [-1, 2])
@pytest.mark.parametrize("attn_type", ["encoder", "encoder_only"])
def test_encoder_segments_are_bidirectional(attention, window, attn_type):
    impl = _impl(attention, window=window, attn_type=attn_type)
    gen = torch.Generator().manual_seed(87)
    q = torch.randn(6, 4, 8, generator=gen)
    k, v = (torch.randn(6, 2, 8, generator=gen) for _ in range(2))
    output = torch.empty_like(q)
    metadata = SimpleNamespace(
        num_actual_tokens=6, query_start_loc=torch.tensor([0, 2, 6]), use_cascade=False
    )
    impl.forward(None, q, k, v, None, metadata, output)
    for lo, hi in [(0, 2), (2, 6)]:
        expected = _oracle(
            q[lo:hi], k[lo:hi], v[lo:hi], impl.scale, causal=False, window=window
        )
        torch.testing.assert_close(output[lo:hi], expected, rtol=1e-5, atol=1e-6)


def test_metadata_disables_unimplemented_paths(attention):
    backend = attention.TxdaSDPAAttentionBackend
    assert not backend.forward_includes_kv_cache_update
    assert not backend.supports_attn_type("encoder_decoder")
    assert not attention.TxdaSDPAAttentionImpl.can_return_lse_for_decode
    builder_cls = backend.get_builder_cls()
    assert builder_cls._cudagraph_support == "never"
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1)
    )
    builder = builder_cls(None, [], config, "cpu")
    assert not builder.use_cascade_attention()
    assert builder.build(32, "common", True).prefix == 0
    config.parallel_config.decode_context_parallel_size = 2
    with pytest.raises(NotImplementedError, match="context parallelism"):
        builder_cls(None, [], config, "cpu")


def test_profile_and_quantized_output(attention):
    impl = _impl(attention)
    output = torch.ones(2, 4, 8)
    assert impl.forward(None, output, None, None, None, None, output) is output
    assert not output.any()
    with pytest.raises(NotImplementedError, match="output quantization"):
        impl.forward(None, output, None, None, None, None, output, output_scale=1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_sdpa(attention, dtype):
    impl = _impl(attention)
    generator = torch.Generator().manual_seed(912)
    q = torch.randn(2, 4, 8, generator=generator).to(dtype)
    k, v = (torch.randn(11, 2, 8, generator=generator).to(dtype) for _ in range(2))
    output = impl._sdpa(q, k, v, 11, -1)
    torch.testing.assert_close(
        output, _oracle(q, k, v, impl.scale), rtol=0.02, atol=0.01
    )


def test_noncausal_respects_asymmetric_window(attention):
    impl = _impl(attention, window=2)
    impl.sliding_window = (2, 0)
    generator = torch.Generator().manual_seed(601)
    q = torch.randn(4, 4, 8, generator=generator)
    k, v = (torch.randn(4, 2, 8, generator=generator) for _ in range(2))
    output = impl._sdpa(q, k, v, 4, 2, causal=False)
    expected = _oracle(q, k, v, impl.scale, causal=False, window=2, window_right=0)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
