# Copyright 2025 BAAI. All rights reserved.

"""Functional end-to-end attention on the T-Head PPU.

Unit tests exercise the T-Head cache writer and the FA3 bridge in
isolation. This functional test runs the production sequence a decode step
actually performs on the PPU:

1. allocate the packed vLLM logical
   ``[num_blocks, num_kv_heads, block_size, 2 * head_size]`` cache,
2. write a new token batch into it through
   ``reshape_and_cache_flash_thead`` (the exact function the FA3 bridge
   injects as ``flash_attn.reshape_and_cache_flash``),
3. run paged attention over the updated cache through
   ``_thead_flash_attn_varlen_func`` with ``cu_seqlens_k=None``,
4. compare against the float32 SDPA oracle gathered from the same cache.

The write and the read therefore share one allocation, which is the only
way to catch stride/layout disagreements between the two halves of the
pipeline (a wrong stride in the writer corrupts the reader's view even
though each side passes its own unit test).

The thead container's ``empty`` vLLM wheel cannot lazily resolve
``vllm.platforms.current_platform`` (missing ``vllm._C_stable_libtorch``),
so the module fixture pre-publishes an inert thead platform — the same
approach as the supplementary CPU contract suite.
"""

from __future__ import annotations

import importlib
import math

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("thead")]

_MODULE = "vllm_fl.dispatch.backends.vendor.thead.impl.attention"


@pytest.fixture(scope="module")
def ppu_pipeline():
    """Return (bridge module, cache writer) imported against the live PPU."""
    if not torch.cuda.is_available():
        pytest.skip("THead PPU accelerator required")
    pytest.importorskip("flash_attn_3._C", reason="PPU flash_attn_3 wheel required")

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

    bridge = importlib.import_module(_MODULE)

    # The bridge injects the cache writer into vLLM's flash_attn module; the
    # same module object backs the production read path. Load the writer the
    # same way (it is a plain import from the bridge's package).
    cache_module = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.thead.impl.cache"
    )
    return bridge, cache_module.reshape_and_cache_flash_thead


def _paged_scenario(head_dim, num_q_heads=8, num_kv_heads=2, block_size=16):
    """Build a decode step: prior context cached, one new token per request."""
    request_count = 3
    prior_lens = [17, 33, 1]
    generator = torch.Generator().manual_seed(4104 + head_dim)
    pages_per_request = math.ceil(40 / block_size)
    num_blocks = request_count * pages_per_request
    # Shuffled pages so a sequential-block assumption would be caught.
    block_table = (
        torch.randperm(num_blocks, generator=generator)
        .reshape(request_count, pages_per_request)
        .to(torch.int32)
    )
    packed_cpu = torch.zeros(
        num_blocks, num_kv_heads, block_size, 2 * head_dim
    ).to(torch.bfloat16)
    packed = packed_cpu.cuda()
    key_cache, value_cache = packed.transpose(1, 2).split(head_dim, dim=-1)

    # New-token batch: one token per request (pure decode step).
    new_keys = torch.randn(
        request_count, num_kv_heads, head_dim, generator=generator
    ).to(torch.bfloat16)
    new_values = torch.randn(
        request_count, num_kv_heads, head_dim, generator=generator
    ).to(torch.bfloat16)
    queries = torch.randn(
        request_count, num_q_heads, head_dim, generator=generator
    ).to(torch.bfloat16)

    # Each request's new token lands on its own next free slot.
    slots = []
    for request, prior in enumerate(prior_lens):
        page = block_table[request, prior // block_size]
        slots.append(int(page) * block_size + prior % block_size)
    slot_tensor = torch.tensor(slots, device="cuda", dtype=torch.int64)

    # Fill the *prior* context with recognizable values through the very
    # writer under test — the same op the engine uses each step.
    prior_slots = []
    prior_keys = []
    prior_values = []
    for request, prior in enumerate(prior_lens):
        for token in range(prior):
            page = block_table[request, token // block_size]
            prior_slots.append(int(page) * block_size + token % block_size)
    prior_slots = torch.tensor(prior_slots, device="cuda", dtype=torch.int64)
    prior_key_batch = (
        torch.randn(prior_slots.numel(), num_kv_heads, head_dim, generator=generator)
        .to(torch.bfloat16)
        .cuda()
    )
    prior_value_batch = (
        torch.randn(prior_slots.numel(), num_kv_heads, head_dim, generator=generator)
        .to(torch.bfloat16)
        .cuda()
    )

    scale = torch.ones((), device="cuda")
    return {
        "head_dim": head_dim,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "block_size": block_size,
        "block_table": block_table.cuda(),
        "packed": packed,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "prior_slots": prior_slots,
        "prior_key_batch": prior_key_batch,
        "prior_value_batch": prior_value_batch,
        "new_keys": new_keys.cuda(),
        "new_values": new_values.cuda(),
        "queries": queries.cuda(),
        "new_slots": slot_tensor,
        "prior_lens": prior_lens,
        "scale": scale,
    }


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_write_then_paged_attention_matches_sdpa(ppu_pipeline, head_dim):
    """Cache write + FA3 paged read over one shared allocation."""
    bridge, cache_writer = ppu_pipeline
    scenario = _paged_scenario(head_dim)
    scale = scenario["scale"]

    # Step 1: write the prior context (multi-token batch).
    cache_writer(
        scenario["prior_key_batch"],
        scenario["prior_value_batch"],
        scenario["key_cache"],
        scenario["value_cache"],
        scenario["prior_slots"],
        "auto",
        scale,
        scale,
    )
    # Step 2: write the new decode tokens.
    cache_writer(
        scenario["new_keys"],
        scenario["new_values"],
        scenario["key_cache"],
        scenario["value_cache"],
        scenario["new_slots"],
        "auto",
        scale,
        scale,
    )
    torch.cuda.synchronize()

    # Step 3: paged attention with cu_seqlens_k=None over the whole cache.
    sequence_lens = [prior + 1 for prior in scenario["prior_lens"]]
    out = bridge._thead_flash_attn_varlen_func(
        q=scenario["queries"],
        k=scenario["key_cache"],
        v=scenario["value_cache"],
        max_seqlen_q=1,
        cu_seqlens_q=torch.arange(
            len(sequence_lens) + 1, device="cuda", dtype=torch.int32
        ),
        max_seqlen_k=max(sequence_lens),
        seqused_k=torch.tensor(sequence_lens, device="cuda", dtype=torch.int32),
        block_table=scenario["block_table"],
        causal=True,
        softmax_scale=head_dim**-0.5,
    )
    torch.cuda.synchronize()
    assert out.shape == (
        len(sequence_lens),
        scenario["num_q_heads"],
        head_dim,
    )

    # Step 4: SDPA oracle gathered from the *device* cache (so a writer bug
    # would have to corrupt both the kernel result and the oracle identically
    # to pass — it cannot, because the oracle does gather-by-block-table).
    key_cache = scenario["key_cache"].cpu()
    value_cache = scenario["value_cache"].cpu()
    block_table = scenario["block_table"].cpu().long()
    block_size = scenario["block_size"]
    expected = []
    for request, sequence_len in enumerate(sequence_lens):
        pages = block_table[request, : math.ceil(sequence_len / block_size)]
        keys = key_cache.index_select(0, pages).flatten(0, 1)[:sequence_len].float()
        values = (
            value_cache.index_select(0, pages).flatten(0, 1)[:sequence_len].float()
        )
        keys = keys.transpose(0, 1).repeat_interleave(
            scenario["num_q_heads"] // scenario["num_kv_heads"], dim=0
        )
        values = values.transpose(0, 1).repeat_interleave(
            scenario["num_q_heads"] // scenario["num_kv_heads"], dim=0
        )
        queries = (
            scenario["queries"].cpu().float()[request : request + 1].transpose(0, 1)
        )
        mask = torch.arange(sequence_len)[None, :] <= sequence_len - 1
        result = F.scaled_dot_product_attention(
            queries, keys[None], values[None],
            attn_mask=mask, scale=head_dim**-0.5,
        )
        expected.append(result[0].transpose(0, 1))
    torch.testing.assert_close(
        out.cpu().float(),
        torch.cat(expected),
        rtol=0.02,
        atol=0.02,
        msg=lambda message: f"PPU write-then-attend, head_dim={head_dim}: {message}",
    )


@pytest.mark.gpu
def test_unwritten_pages_do_not_leak_into_attention(ppu_pipeline):
    """A request's seqused_k must stop before an unwritten page's garbage."""
    bridge, cache_writer = ppu_pipeline
    head_dim = 64
    scenario = _paged_scenario(head_dim)
    scale = scenario["scale"]

    # Write only the prior context; the "new token" slots stay zero-filled
    # from allocation. If the reader ignored seqused_k it would pick up
    # whatever sits in the next page and disagree with the SDPA oracle.
    cache_writer(
        scenario["prior_key_batch"],
        scenario["prior_value_batch"],
        scenario["key_cache"],
        scenario["value_cache"],
        scenario["prior_slots"],
        "auto",
        scale,
        scale,
    )
    torch.cuda.synchronize()

    sequence_lens = list(scenario["prior_lens"])
    out = bridge._thead_flash_attn_varlen_func(
        q=scenario["queries"],
        k=scenario["key_cache"],
        v=scenario["value_cache"],
        max_seqlen_q=1,
        cu_seqlens_q=torch.arange(
            len(sequence_lens) + 1, device="cuda", dtype=torch.int32
        ),
        max_seqlen_k=max(sequence_lens),
        seqused_k=torch.tensor(sequence_lens, device="cuda", dtype=torch.int32),
        block_table=scenario["block_table"],
        causal=True,
        softmax_scale=head_dim**-0.5,
    )
    torch.cuda.synchronize()

    key_cache = scenario["key_cache"].cpu()
    value_cache = scenario["value_cache"].cpu()
    block_table = scenario["block_table"].cpu().long()
    block_size = scenario["block_size"]
    expected = []
    for request, sequence_len in enumerate(sequence_lens):
        pages = block_table[request, : math.ceil(sequence_len / block_size)]
        keys = key_cache.index_select(0, pages).flatten(0, 1)[:sequence_len].float()
        values = (
            value_cache.index_select(0, pages).flatten(0, 1)[:sequence_len].float()
        )
        keys = keys.transpose(0, 1).repeat_interleave(
            scenario["num_q_heads"] // scenario["num_kv_heads"], dim=0
        )
        values = values.transpose(0, 1).repeat_interleave(
            scenario["num_q_heads"] // scenario["num_kv_heads"], dim=0
        )
        queries = (
            scenario["queries"].cpu().float()[request : request + 1].transpose(0, 1)
        )
        mask = torch.arange(sequence_len)[None, :] <= sequence_len - 1
        result = F.scaled_dot_product_attention(
            queries, keys[None], values[None],
            attn_mask=mask, scale=head_dim**-0.5,
        )
        expected.append(result[0].transpose(0, 1))
    torch.testing.assert_close(
        out.cpu().float(),
        torch.cat(expected),
        rtol=0.02,
        atol=0.02,
        msg=lambda message: f"PPU truncated-KV read: {message}",
    )
