# Copyright 2025 BAAI. All rights reserved.
"""Functional end-to-end FlagGems attention on the Iluvatar BI-V150.

``vllm_fl/dispatch/backends/flaggems/impl/attention.py`` is 81/297 on the
CPU baseline — the ``AttentionFLImpl`` forward paths execute
``flaggems_vllm.flash_attn_varlen_func``, a Triton kernel that needs a real
device. This test drives the production decoder sequence end-to-end on the
BI-V150 (mirroring ``tests/unit_tests/dispatch/test_flaggems_attention.py``
which is gated to nvidia):

1. allocate the packed vLLM logical
   ``[num_blocks, num_kv_heads, block_size, 2 * head_size]`` cache,
2. write K/V through ``AttentionFLImpl.do_kv_cache_update`` (vLLm's
   stride-aware Triton writer behind the unified_kv_cache_update op),
3. run paged GQA attention through ``AttentionFLImpl.forward`` with the
   real ``AttentionFLMetadata``,
4. compare against an fp32 SDPA oracle gathered from the same cache.

The write and the read share one allocation, so a stride/layout
disagreement between writer and reader corrupts both sides differently and
cannot pass. Encoder-only attention (no KV cache, bidirectional) is covered
through ``AttentionType.ENCODER_ONLY``. Tolerances follow
``tests/platforms/iluvatar.yaml``: bf16 1e-2 (attention inputs are bf16 on
BI-V150), default 1e-3 for fp32 encoder input.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.supplementary,
    pytest.mark.platforms("iluvatar"),
    pytest.mark.gpu,
]

# tests/platforms/iluvatar.yaml tolerance.inference
RTOL_BF16 = 1e-2
ATOL_BF16 = 1e-2
RTOL_DEFAULT = 1e-3
ATOL_DEFAULT = 1e-3


def _impl(head_size=64, attn_type="decoder"):
    """Build the real AttentionFLImpl on the iluvatar platform."""
    from vllm.v1.attention.backend import AttentionType

    from vllm_fl.dispatch.backends.flaggems.impl.attention import AttentionFLImpl

    type_map = {
        "decoder": AttentionType.DECODER,
        "encoder_only": AttentionType.ENCODER_ONLY,
    }
    return AttentionFLImpl(
        num_heads=4,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        attn_type=type_map[attn_type],
    )


def _scales(device):
    return SimpleNamespace(
        _q_scale=torch.ones((), device=device),
        _k_scale=torch.ones((), device=device),
        _v_scale=torch.ones((), device=device),
    )


def _cache(blocks, kv_heads, block_size, head_size, device):
    """Packed vLLM logical [B, H, N, 2*D] cache."""
    return torch.zeros(
        blocks, kv_heads, block_size, 2 * head_size, device=device
    ).to(torch.bfloat16)


def _decoder_scenario(head_size, block_size, query_lens, seq_lens, device):
    """Random paged decode/prefill batch with shuffled page tables."""
    heads_q, heads_kv = 4, 2
    generator = torch.Generator().manual_seed(20261019 + head_size)
    num_requests = len(seq_lens)
    pages_per_request = math.ceil(max(seq_lens) / block_size) + 1
    num_blocks = num_requests * pages_per_request
    block_table = (
        torch.randperm(num_blocks, generator=generator)
        .reshape(num_requests, pages_per_request)
        .to(torch.int32)
    )

    total_tokens = sum(seq_lens)
    keys = torch.randn(total_tokens, heads_kv, head_size, generator=generator)
    values = torch.randn(total_tokens, heads_kv, head_size, generator=generator)
    total_queries = sum(query_lens)
    queries = torch.randn(total_queries, heads_q, head_size, generator=generator)

    # every token's slot via the page table
    slots = [
        int(block_table[request][token // block_size]) * block_size
        + token % block_size
        for request, length in enumerate(seq_lens)
        for token in range(length)
    ]

    return {
        "head_size": head_size,
        "heads_q": heads_q,
        "heads_kv": heads_kv,
        "block_size": block_size,
        "block_table": block_table.to(device),
        "pages_per_request": pages_per_request,
        "num_blocks": num_blocks,
        "keys": keys.bfloat16().to(device),
        "values": values.bfloat16().to(device),
        "queries": queries.bfloat16().to(device),
        "query_lens": query_lens,
        "seq_lens": seq_lens,
        "slots": torch.tensor(slots, dtype=torch.int64, device=device),
        "query_start_loc": torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(query_lens), 0).tolist()),
            dtype=torch.int32,
            device=device,
        ),
        "generator": generator,
    }


def _run_decoder(scenario, impl, device):
    """do_kv_cache_update + forward over one shared cache allocation."""
    head_size = scenario["head_size"]
    block_size = scenario["block_size"]
    cache = _cache(
        scenario["num_blocks"], scenario["heads_kv"], block_size, head_size, device
    )
    scales = _scales(device)

    impl.do_kv_cache_update(
        scales, scenario["keys"], scenario["values"], cache, scenario["slots"]
    )
    torch.cuda.synchronize(device)

    metadata = SimpleNamespace(
        num_actual_tokens=sum(scenario["query_lens"]),
        use_cascade=False,
        query_start_loc=scenario["query_start_loc"],
        seq_lens=torch.tensor(scenario["seq_lens"], dtype=torch.int32, device=device),
        max_query_len=max(scenario["query_lens"]),
        max_seq_len=max(scenario["seq_lens"]),
        block_table=scenario["block_table"],
        scheduler_metadata=None,
        causal=True,
    )
    output = torch.empty_like(scenario["queries"])
    impl.forward(
        scales,
        scenario["queries"],
        scenario["keys"],
        scenario["values"],
        cache,
        metadata,
        output,
    )
    torch.cuda.synchronize(device)
    return output, cache


def _sdpa_oracle(scenario, cache):
    """fp32 SDPA oracle gathered from the device cache by page table."""
    head_size = scenario["head_size"]
    block_size = scenario["block_size"]
    # split the packed cache the same way _split_kv_cache does
    key_cache, value_cache = cache.transpose(1, 2).split(head_size, dim=-1)
    key_cache = key_cache.cpu()
    value_cache = value_cache.cpu()
    block_table = scenario["block_table"].cpu().long()

    expected = []
    key_start = 0
    query_start = 0
    for query_len, seq_len in zip(
        scenario["query_lens"], scenario["seq_lens"], strict=True
    ):
        request_index = len(expected)
        pages = block_table[request_index, : math.ceil(seq_len / block_size)]
        keys = key_cache.index_select(0, pages).flatten(0, 1)[:seq_len].float()
        values = value_cache.index_select(0, pages).flatten(0, 1)[:seq_len].float()
        keys = keys.transpose(0, 1).repeat_interleave(
            scenario["heads_q"] // scenario["heads_kv"], dim=0
        )
        values = values.transpose(0, 1).repeat_interleave(
            scenario["heads_q"] // scenario["heads_kv"], dim=0
        )
        queries = (
            scenario["queries"]
            .cpu()
            .float()[query_start : query_start + query_len]
            .transpose(0, 1)
        )
        q_positions = torch.arange(query_len) + seq_len - query_len
        mask = torch.arange(seq_len)[None, :] <= q_positions[:, None]
        result = F.scaled_dot_product_attention(
            queries[None],
            keys[None],
            values[None],
            attn_mask=mask,
            scale=head_size**-0.5,
        )
        expected.append(result[0].transpose(0, 1))
        key_start += seq_len
        query_start += query_len
    return torch.cat(expected)


@pytest.mark.parametrize("head_size", [64, 128], ids=["d64", "d128"])
@pytest.mark.parametrize(
    "query_lens,seq_lens",
    [((1, 1), (17, 33)), ((3, 2), (7, 19))],
    ids=["decode", "prefill"],
)
def test_paged_gqa_decoder_matches_sdpa(head_size, query_lens, seq_lens):
    """Cache write + paged GQA read vs the SDPA oracle."""
    if not torch.cuda.is_available():
        pytest.skip("Iluvatar BI-V150 accelerator required")
    device = torch.device("cuda:0")
    scenario = _decoder_scenario(head_size, 16, query_lens, seq_lens, device)
    impl = _impl(head_size=head_size)
    output, cache = _run_decoder(scenario, impl, device)

    assert output.shape == (
        sum(query_lens),
        scenario["heads_q"],
        head_size,
    )
    expected = _sdpa_oracle(scenario, cache)
    torch.testing.assert_close(
        output.float().cpu(),
        expected,
        rtol=RTOL_BF16,
        atol=ATOL_BF16,
    )


def test_encoder_only_attention_matches_sdpa():
    """ENCODER_ONLY forward: no KV cache, bidirectional, causal=False."""
    if not torch.cuda.is_available():
        pytest.skip("Iluvatar BI-V150 accelerator required")
    device = torch.device("cuda:0")
    head_size, heads_q, heads_kv = 64, 4, 2
    generator = torch.Generator().manual_seed(31)
    query_lens = [5, 3]
    total = sum(query_lens)
    queries = (
        torch.randn(total, heads_q, head_size, generator=generator)
        .bfloat16()
        .to(device)
    )
    keys = (
        torch.randn(total, heads_kv, head_size, generator=generator)
        .bfloat16()
        .to(device)
    )
    values = (
        torch.randn(total, heads_kv, head_size, generator=generator)
        .bfloat16()
        .to(device)
    )

    impl = _impl(head_size=head_size, attn_type="encoder_only")
    scales = _scales(device)
    metadata = SimpleNamespace(
        num_actual_tokens=total,
        use_cascade=False,
        query_start_loc=torch.tensor(
            [0, query_lens[0], total], dtype=torch.int32, device=device
        ),
        max_query_len=max(query_lens),
        max_seq_len=max(query_lens),
        seq_lens=None,
        block_table=None,
        slot_mapping=None,
        scheduler_metadata=None,
        causal=False,
    )
    output = torch.empty_like(queries)
    # encoder attention ignores the cache; pass the tensors themselves
    impl.forward(scales, queries, keys, values, torch.empty(0, device=device), metadata, output)
    torch.cuda.synchronize(device)

    # bidirectional SDPA oracle
    expected = []
    start = 0
    for length in query_lens:
        q = queries.float()[start : start + length].transpose(0, 1)
        k = keys.float()[start : start + length].transpose(0, 1).repeat_interleave(
            heads_q // heads_kv, dim=0
        )
        v = values.float()[start : start + length].transpose(0, 1).repeat_interleave(
            heads_q // heads_kv, dim=0
        )
        result = F.scaled_dot_product_attention(
            q[None], k[None], v[None], scale=head_size**-0.5
        )
        expected.append(result[0].transpose(0, 1))
        start += length
    torch.testing.assert_close(
        output.float().cpu(),
        torch.cat(expected),
        rtol=RTOL_BF16,
        atol=ATOL_BF16,
    )


def test_profiling_run_fills_output_with_zeros():
    """``attn_metadata is None`` short-circuits into an output zero-fill."""
    if not torch.cuda.is_available():
        pytest.skip("Iluvatar BI-V150 accelerator required")
    device = torch.device("cuda:0")
    impl = _impl(head_size=64)
    scales = _scales(device)
    output = torch.full((2, 4, 64), 3.0, dtype=torch.bfloat16, device=device)
    result = impl.forward(
        scales,
        torch.empty(2, 4, 64, dtype=torch.bfloat16, device=device),
        torch.empty(2, 2, 64, dtype=torch.bfloat16, device=device),
        torch.empty(2, 2, 64, dtype=torch.bfloat16, device=device),
        torch.empty(0, device=device),
        None,
        output,
    )
    assert result is output
    assert torch.count_nonzero(output).item() == 0


def test_output_scale_rejected():
    """Fused output quantization is not supported and must raise."""
    if not torch.cuda.is_available():
        pytest.skip("Iluvatar BI-V150 accelerator required")
    from vllm.v1.attention.backend import AttentionType

    from vllm_fl.dispatch.backends.flaggems.impl.attention import AttentionFLImpl

    device = torch.device("cuda:0")
    impl = AttentionFLImpl(
        num_heads=4,
        head_size=64,
        scale=64**-0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    scales = _scales(device)
    output = torch.empty(2, 4, 64, dtype=torch.bfloat16, device=device)
    with pytest.raises(NotImplementedError, match="fused output quantization"):
        impl.forward(
            scales,
            torch.empty(2, 4, 64, dtype=torch.bfloat16, device=device),
            torch.empty(2, 2, 64, dtype=torch.bfloat16, device=device),
            torch.empty(2, 2, 64, dtype=torch.bfloat16, device=device),
            torch.empty(0, device=device),
            SimpleNamespace(num_actual_tokens=2, causal=True),
            output,
            output_scale=torch.ones(1, device=device),
        )
