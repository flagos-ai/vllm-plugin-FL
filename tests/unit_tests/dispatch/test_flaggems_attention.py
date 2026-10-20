# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagGems attention cache-layout and paged GQA regression coverage."""

import importlib
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA FlagGems attention test"
)


@pytest.fixture(scope="module")
def attention():
    # An import failure on the target platform is an adaptation failure.
    return importlib.import_module("vllm_fl.dispatch.backends.flaggems.impl.attention")


def _cache(layout, blocks, heads, block_size, head_size, *, device="cpu"):
    if layout == "legacy":
        return torch.zeros(2, blocks, block_size, heads, head_size, device=device)
    if layout == "LBNHC":
        return torch.zeros(
            blocks, block_size, heads, 2 * head_size, device=device
        ).transpose(1, 2)
    return torch.zeros(blocks, heads, block_size, 2 * head_size, device=device)


@pytest.mark.parametrize("layout", ["LBHNC", "LBNHC", "legacy"])
def test_kv_cache_views_preserve_upstream_storage(attention, layout):
    cache = _cache(layout, 3, 2, 16, 8)
    keys, values = attention._split_kv_cache(cache, 8)
    assert keys.shape == values.shape == (3, 16, 2, 8)
    keys[1, 5, 1, 3] = 11
    values[1, 5, 1, 3] = 29
    if layout == "legacy":
        assert cache[0, 1, 5, 1, 3].item() == 11
        assert cache[1, 1, 5, 1, 3].item() == 29
    else:
        assert cache[1, 1, 5, 3].item() == 11
        assert cache[1, 1, 5, 11].item() == 29
    assert cache.count_nonzero().item() == 2


def test_cache_view_rejects_unequal_kv_heads(attention):
    with pytest.raises(ValueError, match="equal K/V head sizes"):
        attention._split_kv_cache(torch.zeros(3, 2, 16, 24), 8)


def test_attention_block_size_accepts_current_cache_spec(attention, monkeypatch):
    monkeypatch.setattr(
        attention,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None, cache_config=None),
    )
    expected = attention.AttentionFLBackend.get_supported_kernel_block_sizes()
    actual = attention.AttentionFLBackend.get_supported_kernel_block_sizes(object())
    assert [(type(size), getattr(size, "base", size)) for size in actual] == [
        (type(size), getattr(size, "base", size)) for size in expected
    ]


@pytest.mark.parametrize("mm_prefix", [False, True])
def test_attention_selector_accepts_mm_prefix_argument(attention, mm_prefix):
    from vllm.platforms.interface import DeviceCapability

    reason = attention.AttentionFLBackend.supports_combination(
        64,
        torch.bfloat16,
        "auto",
        16,
        False,
        False,
        False,
        mm_prefix,
        DeviceCapability(9, 0),
    )
    assert (reason is not None) is mm_prefix


def _impl(attention, head_size=64):
    return attention.AttentionFLImpl(
        num_heads=4,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_batch_invariant_mode_uses_upstream_environment(attention, monkeypatch, enabled):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", str(int(enabled)))
    assert _impl(attention).batch_invariant_enabled is enabled


def _scales():
    return SimpleNamespace(
        _q_scale=torch.ones((), device="cuda"),
        _k_scale=torch.ones((), device="cuda"),
        _v_scale=torch.ones((), device="cuda"),
    )


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["LBHNC", "LBNHC", "legacy"])
def test_cache_write_handles_packed_strides_and_padding(attention, layout):
    impl = _impl(attention)
    cache = _cache(layout, 3, 2, 16, 64, device="cuda").to(torch.bfloat16)
    key = torch.arange(4 * 2 * 64, device="cuda").reshape(4, 2, 64).to(torch.bfloat16)
    value = -key
    slots = torch.tensor([0, 17, -1, 35], dtype=torch.int64, device="cuda")
    impl.do_kv_cache_update(_scales(), key, value, cache, slots)
    actual_k, actual_v = attention._split_kv_cache(cache, 64)
    expected_k = torch.zeros(3, 16, 2, 64, dtype=torch.bfloat16)
    expected_v = torch.zeros_like(expected_k)
    for token, slot in enumerate([0, 17, -1, 35]):
        if slot >= 0:
            expected_k[slot // 16, slot % 16] = key[token].cpu()
            expected_v[slot // 16, slot % 16] = value[token].cpu()
    torch.testing.assert_close(actual_k.cpu(), expected_k, rtol=0, atol=0)
    torch.testing.assert_close(actual_v.cpu(), expected_v, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["LBHNC", "LBNHC", "legacy"])
@pytest.mark.parametrize("query_lens", [(3, 2), (1, 1)], ids=["prefill", "decode"])
def test_paged_gqa_matches_sdpa_with_packed_cache(attention, layout, query_lens):
    head_size, block_size, heads_kv, heads_q = 64, 16, 2, 4
    seq_lens = [7, 19]
    generator = torch.Generator().manual_seed(20261019)
    keys = torch.randn(
        sum(seq_lens), heads_kv, head_size, generator=generator
    ).bfloat16()
    values = torch.randn(
        sum(seq_lens), heads_kv, head_size, generator=generator
    ).bfloat16()
    queries = torch.randn(
        sum(query_lens), heads_q, head_size, generator=generator
    ).bfloat16()
    pages = [[2, 0], [3, 1]]
    slots = [
        pages[request][token // block_size] * block_size + token % block_size
        for request, length in enumerate(seq_lens)
        for token in range(length)
    ]
    query_starts = [0, query_lens[0], sum(query_lens)]
    cache = _cache(layout, 4, heads_kv, block_size, head_size, device="cuda").bfloat16()
    impl = _impl(attention, head_size)
    scales = _scales()
    impl.do_kv_cache_update(
        scales,
        keys.cuda(),
        values.cuda(),
        cache,
        torch.tensor(slots, dtype=torch.int64, device="cuda"),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=sum(query_lens),
        use_cascade=False,
        query_start_loc=torch.tensor(query_starts, dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table=torch.tensor(pages, dtype=torch.int32, device="cuda"),
        scheduler_metadata=None,
        causal=True,
    )
    output = torch.empty_like(queries, device="cuda")
    impl.forward(
        scales, queries.cuda(), keys.cuda(), values.cuda(), cache, metadata, output
    )
    expected, key_start, query_start = [], 0, 0
    for query_len, seq_len in zip(query_lens, seq_lens, strict=True):
        query = queries[query_start : query_start + query_len].float().transpose(0, 1)
        key = keys[key_start : key_start + seq_len].float().transpose(0, 1)
        value = values[key_start : key_start + seq_len].float().transpose(0, 1)
        key = key.repeat_interleave(heads_q // heads_kv, dim=0)
        value = value.repeat_interleave(heads_q // heads_kv, dim=0)
        q_positions = torch.arange(query_len) + seq_len - query_len
        mask = torch.arange(seq_len)[None, :] <= q_positions[:, None]
        result = F.scaled_dot_product_attention(
            query[None],
            key[None],
            value[None],
            attn_mask=mask,
            scale=1 / math.sqrt(head_size),
        )
        expected.append(result[0].transpose(0, 1))
        key_start += seq_len
        query_start += query_len
    torch.testing.assert_close(
        output.float().cpu(), torch.cat(expected), atol=0.025, rtol=0.025
    )
