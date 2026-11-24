# Copyright 2025 BAAI. All rights reserved.

"""Real-PPU coverage for the FlagGems TRITON_ATTN dispatch path.

On the thead platform ``FlagGemsBackend.attention_backend`` returns the
upstream ``TRITON_ATTN`` registry entry by default, and only the
``VLLM_FL_USE_FLAGGEMS_ATTN=1`` opt-in re-routes to the FlagGems
``AttentionFLBackend`` (which needs the ``flaggems_vllm`` package — absent
in this container, so that branch is asserted structurally). The remaining
real-device surface is the small-shape end-to-end run of the Triton
attention kernel on the PPU against the SDPA oracle, which the flaggems
dispatch selection feeds in production.

The ``empty`` vLLM wheel cannot resolve ``vllm.platforms.current_platform``
lazily (missing ``vllm._C_stable_libtorch``), so the module-level fixtures
pre-publish an inert thead platform — the same approach the CPU contract
suite uses — before importing the backend selection code.
"""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("thead")]


@pytest.fixture(scope="module")
def _inert_platform():
    """Publish an importable current_platform for the session; keep it.

    The thead container's empty vLLM wheel raises on lazy platform
    resolution; a pre-published inert platform with thead identity lets the
    registry / backend modules import. Where the platform resolves normally
    this overwrite is equivalent to the natural resolution result.
    """
    if not torch.cuda.is_available():
        pytest.skip("THead PPU accelerator required")
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
    platforms_mod.__dict__["current_platform"] = inert
    platforms_mod._current_platform = inert


@pytest.fixture(scope="module")
def flaggems_backend(_inert_platform):
    """The FlagGems dispatch backend class (import needs the platform)."""
    module = importlib.import_module(
        "vllm_fl.dispatch.backends.flaggems.flaggems"
    )
    return module.FlagGemsBackend


@pytest.fixture(scope="module")
def triton_attn_backend(_inert_platform):
    """The upstream TritonAttentionBackend that the default path selects."""
    return importlib.import_module("vllm.v1.attention.backends.triton_attn")


class TestAttentionBackendSelection:
    """The thead default resolves to TRITON_ATTN; the opt-in is structural."""

    def test_default_returns_upstream_triton_attn(self, flaggems_backend, monkeypatch):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        monkeypatch.delenv("VLLM_FL_USE_FLAGGEMS_ATTN", raising=False)
        expected = AttentionBackendEnum.TRITON_ATTN.get_path()
        assert expected.startswith("vllm."), "registry entry must be upstream vLLM"
        assert flaggems_backend().attention_backend() == expected

    def test_opt_in_returns_flaggems_attentionfl(self, flaggems_backend, monkeypatch):
        monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", "1")
        result = flaggems_backend().attention_backend()
        assert result == (
            "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"
        )

    def test_opt_in_disabled_variants_do_not_count(
        self, flaggems_backend, monkeypatch
    ):
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        for raw in ("0", "false", "no", "off", ""):
            monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", raw)
            assert flaggems_backend().attention_backend() == (
                AttentionBackendEnum.TRITON_ATTN.get_path()
            ), f"VLLM_FL_USE_FLAGGEMS_ATTN={raw!r} must not opt in"

    def test_mla_raises_not_implemented(self, flaggems_backend):
        with pytest.raises(NotImplementedError, match="mla"):
            flaggems_backend().attention_backend(use_mla=True)


def _paged_case():
    """Small paged GQA case: shared with the core flaggems attention test."""
    head_size, block_size, heads_kv, heads_q = 64, 16, 2, 4
    seq_lens, query_lens = [7, 19], [3, 2]
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
    return keys, values, queries, pages, slots, seq_lens, query_lens, head_size


class _ScaleLayer(torch.nn.Module):
    """Minimal attention layer carrying the descale scales."""

    def __init__(self, device):
        super().__init__()
        self._q_scale = torch.ones((), device=device)
        self._k_scale = torch.ones((), device=device)
        self._v_scale = torch.ones((), device=device)


@pytest.mark.gpu
def test_triton_attn_small_shape_matches_sdpa_on_ppu(
    _inert_platform, triton_attn_backend
):
    """End-to-end small-shape run of the selected TRITON_ATTN kernel on PPU."""
    from vllm.v1.attention.backend import AttentionType

    (
        keys, values, queries, pages, slots, seq_lens, query_lens, head_size
    ) = _paged_case()
    block_size = 16
    heads_q = queries.shape[1]
    num_blocks = 4

    impl = triton_attn_backend.TritonAttentionImpl(
        num_heads=heads_q,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=keys.shape[1],
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        attn_type=AttentionType.DECODER,
    )
    assert impl.use_td is False, "tensor descriptors must stay off on the PPU"

    # Logical [B, H, N, 2*D] cache; fill through the impl's own split view so
    # the write and the read agree on the packed layout.
    kv_cache = torch.zeros(
        num_blocks, keys.shape[1], block_size, 2 * head_size,
        device="cuda", dtype=torch.bfloat16,
    )
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    slot_tensor = torch.tensor(slots, dtype=torch.int64, device="cuda")
    for token, slot in enumerate(slots):
        if slot >= 0:
            key_cache[slot // block_size, slot % block_size].copy_(keys[token].cuda())
            value_cache[slot // block_size, slot % block_size].copy_(
                values[token].cuda()
            )

    metadata = SimpleNamespace(
        num_actual_tokens=sum(query_lens),
        query_start_loc=torch.tensor(
            [0, query_lens[0], sum(query_lens)], dtype=torch.int32, device="cuda"
        ),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table=torch.tensor(pages, dtype=torch.int32, device="cuda"),
        causal=True,
        seq_threshold_3D=100000,
        num_par_softmax_segments=0,
        softmax_segm_output=None,
        softmax_segm_max=None,
        softmax_segm_expsum=None,
        mm_prefix_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
    )
    layer = _ScaleLayer("cuda")
    output = torch.empty(
        sum(query_lens), heads_q, head_size, device="cuda", dtype=torch.bfloat16
    )
    impl.forward(
        layer,
        queries.cuda(),
        keys.cuda(),
        values.cuda(),
        kv_cache,
        metadata,
        output,
    )
    torch.cuda.synchronize()

    expected, key_start, query_start = [], 0, 0
    for query_len, seq_len in zip(query_lens, seq_lens, strict=True):
        query = queries[query_start : query_start + query_len].float().transpose(0, 1)
        key = (
            keys[key_start : key_start + seq_len]
            .float()
            .transpose(0, 1)
            .repeat_interleave(heads_q // keys.shape[1], dim=0)
        )
        value = (
            values[key_start : key_start + seq_len]
            .float()
            .transpose(0, 1)
            .repeat_interleave(heads_q // keys.shape[1], dim=0)
        )
        q_positions = torch.arange(query_len) + seq_len - query_len
        mask = torch.arange(seq_len)[None, :] <= q_positions[:, None]
        result = F.scaled_dot_product_attention(
            query[None], key[None], value[None], attn_mask=mask,
            scale=1 / math.sqrt(head_size),
        )
        expected.append(result[0].transpose(0, 1))
        key_start += seq_len
        query_start += query_len
    torch.testing.assert_close(
        output.float().cpu(),
        torch.cat(expected),
        atol=0.03,
        rtol=0.03,
        msg=lambda message: f"PPU TRITON_ATTN small shape: {message}",
    )
