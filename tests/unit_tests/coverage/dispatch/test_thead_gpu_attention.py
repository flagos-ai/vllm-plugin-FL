# Copyright 2025 BAAI. All rights reserved.

"""Real-PPU FA3 numerics for the T-Head varlen bridge.

The CPU contract test (``test_thead_attention_wrapper_cpu.py``) pins the
35-argument call layout with sentinels; the core device suite
(``tests/unit_tests/dispatch/test_thead_attention.py``) covers paged GQA and
graph replay. What remains uncovered on the real PPU are the *non-paged*
branches of ``_thead_flash_attn_varlen_func``:

* ``cu_seqlens_k`` provided (prefill against unpacked K/V — the branch the
  paged tests never exercise),
* ``causal=False``,
* a sliding ``window_size``,
* ``return_softmax_lse=True`` against a float64 logsumexp oracle.

Importing ``impl/attention.py`` pulls in the vLLM flash_attn module, whose
import chain resolves ``vllm.platforms.current_platform`` lazily. The
container's ``empty`` wheel lacks ``vllm._C_stable_libtorch``, so the lazy
resolution raises; the fixture pre-publishes an inert platform (same trick
as the CPU contract test) and restores it afterwards. On wheels where the
real platform resolves, the pre-publish is a harmless no-op equivalent.
"""

from __future__ import annotations

import importlib
import math

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("thead")]

_MODULE = "vllm_fl.dispatch.backends.vendor.thead.impl.attention"
_INJECTED_NAMES = (
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "get_flash_attn_version",
    "reshape_and_cache_flash",
)
_FLASH_ATTN_MOD = "vllm.v1.attention.backends.flash_attn"


@pytest.fixture(scope="module")
def ppu_attention():
    """Import the bridge with the FA3 wheel and an importable platform."""
    if not torch.cuda.is_available():
        pytest.skip("THead PPU accelerator required")
    pytest.importorskip("flash_attn_3._C", reason="PPU flash_attn_3 wheel required")
    pytest.importorskip("vllm", reason="vLLM required for the PPU bridge")

    # Publish an inert current_platform so the lazy ``vllm.platforms``
    # attribute never imports ``vllm.platforms.cuda`` (whose extension is
    # missing in the empty wheel). Where the platform resolves normally this
    # overwrite is equivalent to what resolution would produce.
    import vllm.platforms as platforms_mod

    try:
        from vllm.platforms.interface import UnspecifiedPlatform
    except Exception:  # pragma: no cover - very old vLLM layout
        UnspecifiedPlatform = object

    class _InertPlatform(UnspecifiedPlatform):
        device_type = "cuda"
        vendor_name = "thead"
        dist_backend = "nccl"

    inert = _InertPlatform()
    marker = {
        "current": platforms_mod.__dict__.get("current_platform"),
        "private": getattr(platforms_mod, "_current_platform", None),
        "installed": platforms_mod.__dict__.get("_inert_platform_installed", False),
    }
    platforms_mod.__dict__["current_platform"] = inert
    platforms_mod._current_platform = inert
    platforms_mod.__dict__["_inert_platform_installed"] = True
    platforms_mod.__dict__["_inert_platform_state"] = marker
    try:
        return importlib.import_module(_MODULE)
    except ImportError:
        pytest.skip(f"PPU bridge module failed to import: {_MODULE}")


def _varlen_inputs(num_seqs=2, seq_len=64, head_dim=128, dtype=torch.bfloat16):
    """Non-paged varlen inputs: packed Q/K/V with cu_seqlens for both."""
    generator = torch.Generator().manual_seed(2026)
    query = torch.randn(
        num_seqs * seq_len, 4, head_dim, generator=generator
    ).to(dtype).cuda()
    key = torch.randn(
        num_seqs * seq_len, 2, head_dim, generator=generator
    ).to(dtype).cuda()
    value = torch.randn(
        num_seqs * seq_len, 2, head_dim, generator=generator
    ).to(dtype).cuda()
    cu = torch.tensor(
        [i * seq_len for i in range(num_seqs + 1)], device="cuda", dtype=torch.int32
    )
    return query, key, value, cu, num_seqs, seq_len, head_dim


def _gqa_bhld(x, num_seqs, seq_len, num_target_heads):
    """[N, h, d] -> [B, H, L, d] with GQA head expansion."""
    h = x.shape[1]
    t = (
        x.detach().cpu().float()
        .reshape(num_seqs, seq_len, h, -1)
        .permute(0, 2, 1, 3)
    )
    return t.repeat_interleave(num_target_heads // h, dim=1)


def _sdpa_reference(query, key, value, num_seqs, seq_len, head_dim, *, causal):
    qq = _gqa_bhld(query, num_seqs, seq_len, query.shape[1])
    kk = _gqa_bhld(key, num_seqs, seq_len, query.shape[1])
    vv = _gqa_bhld(value, num_seqs, seq_len, query.shape[1])
    out = F.scaled_dot_product_attention(
        qq, kk, vv, is_causal=causal, scale=head_dim**-0.5
    )
    return out.permute(0, 2, 1, 3).reshape(num_seqs * seq_len, query.shape[1], head_dim)


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
def test_non_paged_prefill_matches_sdpa(ppu_attention, head_dim, causal):
    """``cu_seqlens_k`` provided: the never-paged-tested branch of the bridge."""
    query, key, value, cu, num_seqs, seq_len, head_dim = _varlen_inputs(
        num_seqs=2, seq_len=48, head_dim=head_dim
    )
    out = ppu_attention._thead_flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        max_seqlen_q=seq_len,
        cu_seqlens_q=cu,
        max_seqlen_k=seq_len,
        cu_seqlens_k=cu,
        causal=causal,
        softmax_scale=head_dim**-0.5,
    )
    torch.cuda.synchronize()
    expected = _sdpa_reference(
        query, key, value, num_seqs, seq_len, head_dim, causal=causal
    )
    torch.testing.assert_close(
        out.cpu().float(),
        expected,
        rtol=0.02,
        atol=0.02,
        msg=lambda message: f"PPU FA3 non-paged prefill: {message}",
    )


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim", [64, 128])
def test_sliding_window_matches_reference(ppu_attention, head_dim):
    """``window_size=[W, 0]`` (vLLM sliding-window convention) on the PPU."""
    window = 16
    query, key, value, cu, num_seqs, seq_len, head_dim = _varlen_inputs(
        num_seqs=2, seq_len=48, head_dim=head_dim
    )
    out = ppu_attention._thead_flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        max_seqlen_q=seq_len,
        cu_seqlens_q=cu,
        max_seqlen_k=seq_len,
        cu_seqlens_k=cu,
        causal=True,
        window_size=[window, 0],
        softmax_scale=head_dim**-0.5,
    )
    torch.cuda.synchronize()

    qq = _gqa_bhld(query, num_seqs, seq_len, query.shape[1]).double()
    kk = _gqa_bhld(key, num_seqs, seq_len, query.shape[1]).double()
    vv = _gqa_bhld(value, num_seqs, seq_len, query.shape[1]).double()
    scores = torch.einsum("bhqd,bhkd->bhqk", qq, kk) * (head_dim**-0.5)
    positions = torch.arange(seq_len)
    mask = (positions[None, :] <= positions[:, None]) & (
        positions[None, :] >= positions[:, None] - window
    )
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    expected = (
        torch.einsum("bhqk,bhkd->bhqd", probs, vv)
        .permute(0, 2, 1, 3)
        .reshape(num_seqs * seq_len, query.shape[1], head_dim)
        .float()
    )
    torch.testing.assert_close(
        out.cpu().float(),
        expected,
        rtol=0.02,
        atol=0.02,
        msg=lambda message: f"PPU FA3 sliding window: {message}",
    )


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim", [64, 128])
def test_return_softmax_lse_matches_logsumexp_oracle(ppu_attention, head_dim):
    """``return_softmax_lse=True`` returns LSE shaped [heads, total_tokens]."""
    query, key, value, cu, num_seqs, seq_len, head_dim = _varlen_inputs(
        num_seqs=2, seq_len=48, head_dim=head_dim
    )
    out, lse = ppu_attention._thead_flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        max_seqlen_q=seq_len,
        cu_seqlens_q=cu,
        max_seqlen_k=seq_len,
        cu_seqlens_k=cu,
        causal=True,
        softmax_scale=head_dim**-0.5,
        return_softmax_lse=True,
    )
    torch.cuda.synchronize()
    assert lse is not None and lse.ndim == 2
    assert lse.shape[0] == query.shape[1]
    assert lse.shape[1] == num_seqs * seq_len

    # float64 logsumexp oracle: [B, H, L, L] causal scores, reduced over keys,
    # then repacked to the FA [heads, B * L] layout.
    qq = _gqa_bhld(query, num_seqs, seq_len, query.shape[1]).double()
    kk = _gqa_bhld(key, num_seqs, seq_len, query.shape[1]).double()
    scores = torch.einsum("bhqd,bhkd->bhqk", qq, kk) * (head_dim**-0.5)
    positions = torch.arange(seq_len)
    mask = positions[None, :] <= positions[:, None]
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    per_batch = torch.logsumexp(scores, dim=-1)  # [B, H, L]
    expected_lse = (
        per_batch.permute(1, 0, 2).reshape(query.shape[1], num_seqs * seq_len)
    )
    torch.testing.assert_close(
        lse.cpu().double(),
        expected_lse,
        rtol=0.0,
        atol=1e-3,
        msg=lambda message: f"PPU FA3 softmax LSE: {message}",
    )


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim", [64, 128])
def test_paged_decode_cu_seqlens_k_none_forces_tile_hint(ppu_attention, head_dim):
    """``cu_seqlens_k=None`` forces ``max_seqlen_k = 1`` inside the bridge.

    The paged path is the one the core suite exercises; this check pins the
    forced tile hint against the packed HND cache with shuffled pages, using
    small sequences so the tile-size hint (not the KV length) is what the
    kernel sees.
    """
    block_size, num_kv, num_q = 16, 2, 8
    query_lens, seq_lens = (1, 3), (33, 49)
    pages_per_request = math.ceil(max(seq_lens) / block_size)
    num_blocks = len(seq_lens) * pages_per_request
    generator = torch.Generator().manual_seed(1729 + head_dim)
    block_table = (
        torch.randperm(num_blocks, generator=generator)
        .reshape(len(seq_lens), pages_per_request)
        .to(torch.int32)
    )
    packed_cpu = torch.randn(
        num_blocks, num_kv, block_size, 2 * head_dim, generator=generator
    ).to(torch.bfloat16)
    packed = packed_cpu.cuda()
    key_cache, value_cache = packed.transpose(1, 2).split(head_dim, dim=-1)
    query = torch.randn(
        sum(query_lens), num_q, head_dim, generator=generator
    ).to(torch.bfloat16).cuda()
    cu_q = torch.tensor(
        [0, query_lens[0], sum(query_lens)], device="cuda", dtype=torch.int32
    )

    out = ppu_attention._thead_flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        max_seqlen_q=max(query_lens),
        cu_seqlens_q=cu_q,
        max_seqlen_k=max(seq_lens),
        seqused_k=torch.tensor(seq_lens, device="cuda", dtype=torch.int32),
        block_table=block_table.cuda(),
        causal=True,
        softmax_scale=head_dim**-0.5,
    )
    torch.cuda.synchronize()
    assert out.shape == (sum(query_lens), num_q, head_dim)

    # SDPA oracle over the gathered pages (bottom-right causal alignment).
    key_cpu = key_cache.cpu()
    value_cpu = value_cache.cpu()
    expected, query_start = [], 0
    for request, (query_len, sequence_len) in enumerate(
        zip(query_lens, seq_lens, strict=True)
    ):
        pages = block_table[request, : math.ceil(sequence_len / block_size)].long()
        keys = (
            key_cpu.index_select(0, pages).flatten(0, 1)[:sequence_len].float()
        )
        values = (
            value_cpu.index_select(0, pages).flatten(0, 1)[:sequence_len].float()
        )
        keys = keys.transpose(0, 1).repeat_interleave(num_q // num_kv, dim=0)
        values = values.transpose(0, 1).repeat_interleave(num_q // num_kv, dim=0)
        queries = query.cpu().float()[query_start : query_start + query_len].transpose(0, 1)
        query_positions = torch.arange(query_len) + sequence_len - query_len
        mask = torch.arange(sequence_len)[None, :] <= query_positions[:, None]
        result = F.scaled_dot_product_attention(
            queries[None], keys[None], values[None],
            attn_mask=mask, scale=head_dim**-0.5,
        )
        expected.append(result[0].transpose(0, 1))
        query_start += query_len
    torch.testing.assert_close(
        out.cpu().float(),
        torch.cat(expected),
        rtol=0.02,
        atol=0.02,
        msg=lambda message: f"PPU FA3 paged decode: {message}",
    )


@pytest.mark.gpu
def test_default_softmax_scale_used_when_none(ppu_attention):
    """``softmax_scale=None`` falls back to ``q.shape[-1] ** -0.5``."""
    query, key, value, cu, num_seqs, seq_len, head_dim = _varlen_inputs(
        num_seqs=1, seq_len=32, head_dim=64
    )
    out_default = ppu_attention._thead_flash_attn_varlen_func(
        q=query, k=key, v=value,
        max_seqlen_q=seq_len, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_k=seq_len, causal=True, softmax_scale=None,
    )
    out_explicit = ppu_attention._thead_flash_attn_varlen_func(
        q=query, k=key, v=value,
        max_seqlen_q=seq_len, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_k=seq_len, causal=True, softmax_scale=head_dim**-0.5,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out_default.cpu().float(), out_explicit.cpu().float(), rtol=0, atol=0
    )
