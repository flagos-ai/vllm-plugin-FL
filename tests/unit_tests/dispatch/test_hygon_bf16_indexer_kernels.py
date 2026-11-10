# Copyright (c) 2026 BAAI. All rights reserved.

"""GPU correctness checks for the Hygon BF16 Lightning Indexer leaves.

Run on a Hygon server with::

    pytest -q -s tests/unit_tests/dispatch/test_hygon_bf16_indexer_kernels.py

The tests intentionally use PyTorch references instead of importing vllm_hcu.
Each comparison prints finite/error statistics before asserting closeness.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def bf16_impl():
    if not torch.cuda.is_available():
        pytest.skip("Hygon/HIP GPU is required")
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention import bf16_indexer

    return bf16_indexer


def _randn(*shape: int, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype)


@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("page_padding", [0, 8192])
def test_decode_graph_replay_growing_lengths(bf16_impl, next_n, page_padding):
    """A graph captured on short rows must also compute later long rows."""
    torch.manual_seed(915)
    batch, capacity = 8, 32768
    q = _randn(batch, next_n, 64, 128) * 0.125
    cache = _randn(512, 64, 128) * 0.125
    if page_padding:
        backing = torch.full((512, 8192 + page_padding), float("nan"),
                             device="cuda", dtype=torch.bfloat16)
        padded = torch.as_strided(backing, cache.shape,
                                  (8192 + page_padding, 128, 1), storage_offset=17)
        padded.copy_(cache)
        cache = padded
    weights = torch.randn(batch * next_n, 64, device="cuda") * 0.02
    lens = torch.ones(batch * next_n, device="cuda", dtype=torch.int32)
    table = torch.randint(0, 512, (batch, 519), device="cuda", dtype=torch.int32)
    table = table[:, :512]
    bf16_impl.bf16_paged_mqa_logits(q, cache, weights, lens, table, capacity)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = bf16_impl.bf16_paged_mqa_logits(
            q, cache, weights, lens, table, capacity
        )
    for length in (0, 63, 65, 320, 4097, 32768):
        lens.fill_(length)
        q.normal_(std=0.125)
        out.fill_(float("nan"))
        graph.replay()
        assert torch.isnan(out[:, length:]).all()
        if length == 0:
            continue
        positions = torch.arange(length, device="cuda")
        # Sample both ends, including the last speculative query of a request.
        for row in (0, batch * next_n - 1):
            keys = cache[table[row // next_n, positions // 64].long(), positions % 64]
            query = q.reshape(-1, 64, 128)[row]
            expected = (torch.relu(keys.float() @ query.float().T) * weights[row]).sum(-1)
            torch.testing.assert_close(out[row, :length], expected, atol=1e-6, rtol=1e-5)


def _report_and_assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    print(
        f"{name}: shape={tuple(actual.shape)} dtype={actual.dtype} "
        f"nan={bool(torch.isnan(actual_f).any().item())} "
        f"inf={bool(torch.isinf(actual_f).any().item())} "
        f"max_abs={diff.max().item():.8g} "
        f"mean_abs={diff.mean().item():.8g}"
    )
    assert torch.isfinite(actual_f).all(), f"{name} produced NaN/Inf"
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _report_and_assert_same_index_set(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    """Compare top-k selections without assuming a particular output order."""
    actual_sorted = actual.sort(dim=-1).values
    expected_sorted = expected.sort(dim=-1).values
    mismatches = (actual_sorted != expected_sorted).sum().item()
    ordered_matches = (actual == expected).sum().item()
    print(
        f"{name}: shape={tuple(actual.shape)} dtype={actual.dtype} "
        f"set_mismatches={mismatches}/{actual.numel()} "
        f"ordered_matches={ordered_matches}/{actual.numel()}"
    )
    torch.testing.assert_close(actual_sorted, expected_sorted, atol=0, rtol=0)


def _cos_sin_cache(num_positions: int, rope_dim: int = 64) -> torch.Tensor:
    angles = torch.randn(
        num_positions,
        rope_dim // 2,
        device="cuda",
        dtype=torch.float32,
    )
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)


def _rope_bf16_reference(
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int = 64,
) -> torch.Tensor:
    result = value.float().clone()
    nope_dim = value.shape[-1] - rope_dim
    half = rope_dim // 2
    cos_sin = cos_sin_cache[positions.long()].float()
    cos, sin = cos_sin[..., :half], cos_sin[..., half:]
    while cos.ndim < result.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rotary = result[..., nope_dim:]
    # Clone both views because the reference writes the rotated result back
    # into ``result`` in-place.
    even = rotary[..., 0::2].clone()
    odd = rotary[..., 1::2].clone()
    result[..., nope_dim::2] = even * cos - odd * sin
    result[..., nope_dim + 1 :: 2] = odd * cos + even * sin
    return result.to(torch.bfloat16)


def test_fused_indexer_q_rope_bf16_matches_torch(bf16_impl):
    torch.manual_seed(101)
    num_tokens, num_heads, head_dim = 11, 64, 128
    positions = torch.tensor(
        [0, 1, 2, 3, 5, 8, 13, 17, 21, 29, 31],
        device="cuda",
        dtype=torch.int64,
    )
    q = _randn(num_tokens, num_heads, head_dim)
    weights = _randn(num_tokens, num_heads)
    cos_sin = _cos_sin_cache(32)
    softmax_scale = head_dim**-0.5
    head_scale = num_heads**-0.5

    actual_q, actual_weights = bf16_impl.fused_indexer_q_rope_bf16(
        positions,
        q,
        cos_sin,
        weights,
        softmax_scale,
        head_scale,
    )
    expected_q = _rope_bf16_reference(q, positions, cos_sin)
    expected_weights = weights.float() * softmax_scale * head_scale

    _report_and_assert_close(
        "fused_indexer_q_rope_bf16/q",
        actual_q,
        expected_q,
        atol=0.015625,
        rtol=0.005,
    )
    _report_and_assert_close(
        "fused_indexer_q_rope_bf16/weights",
        actual_weights,
        expected_weights,
        atol=1e-7,
        rtol=1e-6,
    )


def test_gather_bf16_indexer_cache_matches_torch(bf16_impl):
    torch.manual_seed(102)
    cache = _randn(6, 64, 128)
    block_table = torch.tensor(
        [[4, 1], [5, 2]],
        device="cuda",
        dtype=torch.int32,
    )
    seq_lens = (70, 33)
    cu_seq_lens = torch.tensor(
        [0, seq_lens[0], sum(seq_lens)],
        device="cuda",
        dtype=torch.int32,
    )
    actual = torch.empty(sum(seq_lens), 128, device="cuda", dtype=torch.bfloat16)

    bf16_impl.gather_bf16_indexer_cache(
        cache,
        actual,
        block_table,
        cu_seq_lens,
    )

    expected_rows = []
    for batch_idx, seq_len in enumerate(seq_lens):
        for position in range(seq_len):
            block = block_table[batch_idx, position // 64].item()
            expected_rows.append(cache[block, position % 64])
    expected = torch.stack(expected_rows)
    _report_and_assert_close(
        "gather_bf16_indexer_cache",
        actual,
        expected,
        atol=0.0,
        rtol=0.0,
    )


def _paged_mqa_reference(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, next_n = q.shape[:2]
    max_context_len = int(context_lens.max().item())
    reference = torch.zeros(
        batch_size * next_n,
        max_context_len,
        device=q.device,
        dtype=torch.float32,
    )
    valid = torch.zeros_like(reference, dtype=torch.bool)
    flat_lens = context_lens.reshape(-1)
    for row in range(batch_size * next_n):
        batch_idx = row // next_n
        context_len = int(flat_lens[row].item())
        keys = []
        for position in range(context_len):
            block = block_table[batch_idx, position // 64].item()
            keys.append(cache[block, position % 64])
        key = torch.stack(keys).float()
        query = q.reshape(-1, 64, 128)[row].float()
        scores = torch.relu(key @ query.transpose(0, 1))
        reference[row, :context_len] = (
            scores * weights.reshape(-1, 64)[row].float().unsqueeze(0)
        ).sum(dim=-1)
        valid[row, :context_len] = True
    return reference, valid


@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("page_padding", [0, 8192])
@pytest.mark.parametrize("length_driven", ["0", "1"])
def test_bf16_paged_mqa_logits_matches_torch(
    bf16_impl, next_n, page_padding, length_driven, monkeypatch
):
    monkeypatch.setenv("VLLM_FL_HYGON_INDEXER_LENGTH_DRIVEN", length_driven)
    torch.manual_seed(103 + next_n)
    batch_size = 2
    q = _randn(batch_size, next_n, 64, 128) * 0.125
    cache = _randn(6, 64, 128) * 0.125
    if page_padding:
        backing = torch.full((6, 8192 + page_padding), float("nan"),
                             device="cuda", dtype=torch.bfloat16)
        padded = torch.as_strided(backing, cache.shape,
                                  (8192 + page_padding, 128, 1), storage_offset=17)
        padded.copy_(cache)
        cache = padded
    weights = (
        torch.randn(
            batch_size * next_n,
            64,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    )
    if next_n == 1:
        context_lens = torch.tensor([[57], [79]], device="cuda", dtype=torch.int32)
    else:
        context_lens = torch.tensor(
            [[57, 58], [79, 80]],
            device="cuda",
            dtype=torch.int32,
        )
    block_table = torch.tensor(
        [[3, 1], [2, 5]],
        device="cuda",
        dtype=torch.int32,
    )
    max_context_len = 128

    actual = bf16_impl.bf16_paged_mqa_logits(
        q,
        cache,
        weights,
        context_lens,
        block_table,
        max_context_len,
    )
    expected, _ = _paged_mqa_reference(
        q,
        cache,
        weights,
        context_lens,
        block_table,
    )
    expected_padded = torch.zeros_like(actual)
    expected_padded[:, : expected.shape[1]] = expected
    valid_mask = torch.arange(actual.shape[1], device="cuda").unsqueeze(
        0
    ) < context_lens.reshape(-1, 1)
    _report_and_assert_close(
        f"bf16_paged_mqa_logits/next_n={next_n}",
        actual[valid_mask],
        expected_padded[valid_mask],
        atol=0.03,
        rtol=0.03,
    )


@pytest.mark.parametrize("batch", [16, 24, 32, 48, 64])
def test_decode_packed_pages_typical_batches(bf16_impl, batch, monkeypatch):
    """Packed pages and changing physical tables remain correct on graph replay."""
    monkeypatch.setenv("VLLM_FL_HYGON_INDEXER_LENGTH_DRIVEN", "1")
    torch.manual_seed(917 + batch)
    q = _randn(batch, 1, 64, 128) * 0.125
    weights = torch.randn(batch, 64, device="cuda") * 0.02
    keys = _randn(128, 64, 128) * 0.125
    backing = torch.full((128, 32768), float("nan"), device="cuda", dtype=torch.bfloat16)
    packed = torch.as_strided(backing, keys.shape, (32768, 128, 1), storage_offset=17)
    packed.copy_(keys)
    lengths = torch.full((batch, 1), 513, device="cuda", dtype=torch.int32)
    tables = torch.randint(0, 128, (batch, 128), device="cuda", dtype=torch.int32)
    def run():
        return bf16_impl.bf16_paged_mqa_logits(q, packed, weights, lengths, tables, 8192)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for length in (63, 513, 4097):
        lengths.fill_(length)
        tables.random_(0, 128)
        q.normal_(std=0.125)
        graph.replay()
        expected = bf16_impl.bf16_paged_mqa_logits(q, keys, weights, lengths, tables, 8192)
        assert torch.equal(actual[:, :length], expected[:, :length])


def _compress_reference(
    state_cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    cos_sin: torch.Tensor,
    compress_ratio: int,
) -> list[torch.Tensor]:
    head_dim = 128
    state_width = state_cache.shape[-1] // 2
    overlap = 1
    outputs = []
    for position_tensor in positions:
        position = int(position_tensor.item())
        if (position + 1) % compress_ratio:
            continue
        start = position - (1 + overlap) * compress_ratio + 1
        kv_rows = []
        score_rows = []
        for token_offset in range((1 + overlap) * compress_ratio):
            state_position = start + token_offset
            if state_position < 0:
                continue
            block = int(block_table[0, state_position // 256].item())
            row = state_cache[block, state_position % 256]
            head_offset = head_dim if token_offset >= compress_ratio else 0
            kv_rows.append(row[head_offset : head_offset + head_dim].float())
            score_rows.append(
                row[
                    state_width + head_offset : state_width + head_offset + head_dim
                ].float()
            )
        kv = torch.stack(kv_rows)
        scores = torch.stack(score_rows).softmax(dim=0)
        compressed = (kv * scores).sum(dim=0)
        variance = compressed.square().sum() / head_dim
        normed = compressed * torch.rsqrt(variance + rms_eps) * rms_weight.float()
        compressed_position = position // compress_ratio * compress_ratio
        outputs.append(
            _rope_bf16_reference(
                normed.view(1, head_dim),
                torch.tensor([compressed_position], device="cuda"),
                cos_sin,
            ).squeeze(0)
        )
    return outputs


def test_compress_norm_rope_store_bf16_matches_torch(bf16_impl):
    torch.manual_seed(106)
    num_actual = 8
    state_cache = _randn(1, 256, 512, dtype=torch.float32) * 0.25
    token_to_req_indices = torch.zeros(num_actual, device="cuda", dtype=torch.int32)
    positions = torch.arange(num_actual, device="cuda", dtype=torch.int64)
    state_slot_mapping = torch.arange(num_actual, device="cuda", dtype=torch.int64)
    block_table = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    rms_weight = _randn(128).abs() + 0.5
    rms_eps = 1e-6
    cos_sin = _cos_sin_cache(16)
    cache = torch.zeros(1, 64, 128, device="cuda", dtype=torch.bfloat16)
    kv_slot_mapping = torch.full(
        (num_actual,),
        -1,
        device="cuda",
        dtype=torch.int64,
    )
    kv_slot_mapping[3] = 0
    kv_slot_mapping[7] = 1
    metadata = SimpleNamespace(slot_mapping=kv_slot_mapping)

    bf16_impl.compress_norm_rope_store_bf16(
        state_cache=state_cache,
        num_actual=num_actual,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        slot_mapping=state_slot_mapping,
        block_table=block_table,
        block_size=256,
        state_width=256,
        cos_sin_cache=cos_sin,
        kv_cache=cache,
        k_cache_metadata=metadata,
        pdl_kwargs={},
        head_dim=128,
        rope_head_dim=64,
        compress_ratio=4,
        overlap=True,
        use_fp4_cache=False,
        rms_norm_weight=rms_weight,
        rms_norm_eps=rms_eps,
        quant_block=128,
        token_stride=128,
        scale_dim=4,
    )
    expected = _compress_reference(
        state_cache,
        positions,
        block_table,
        rms_weight,
        rms_eps,
        cos_sin,
        4,
    )
    _report_and_assert_close(
        "compress_norm_rope_store_bf16",
        cache[0, :2],
        torch.stack(expected),
        atol=0.02,
        rtol=0.02,
    )


def test_lightop_bf16_prefill_mqa_matches_torch(bf16_impl):
    del bf16_impl
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.sparse_attn_indexer import (
        _get_lightop_attention,
    )

    attention = _get_lightop_attention()
    if attention is None:
        pytest.fail("Hygon BF16 Indexer prefill requires LightOp attention")

    torch.manual_seed(107)
    num_queries = num_keys = 128
    q = _randn(num_queries, 64, 128) * 0.125
    k = _randn(num_keys, 128) * 0.125
    weights = (
        torch.randn(
            num_queries,
            64,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    )
    starts = torch.zeros(num_queries, device="cuda", dtype=torch.int32)
    ends = torch.full(
        (num_queries,),
        num_keys,
        device="cuda",
        dtype=torch.int32,
    )
    actual = torch.empty(
        num_queries,
        num_keys,
        device="cuda",
        dtype=torch.float32,
    )

    attention.mqa_logits(q, k, weights, starts, ends, None, True, actual)
    scores = torch.einsum("kD,qhD->qkh", k.float(), q.float()).relu()
    expected = (scores * weights[:, None, :]).sum(dim=-1)
    _report_and_assert_close(
        "lightop_bf16_prefill_mqa",
        actual,
        expected,
        atol=0.03,
        rtol=0.03,
    )


def test_lightop_indexer_topk_matches_torch(bf16_impl):
    del bf16_impl
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.sparse_attn_indexer import (
        top_k_per_row_decode_hygon_out,
        top_k_per_row_prefill_hygon_out,
    )

    torch.manual_seed(108)
    num_rows, num_columns, topk = 4, 4096, 2048
    logits = torch.randn(
        num_rows,
        num_columns,
        device="cuda",
        dtype=torch.float32,
    )

    starts = torch.tensor([0, 64, 128, 256], device="cuda", dtype=torch.int32)
    ends = torch.tensor([4096, 4000, 3900, 3800], device="cuda", dtype=torch.int32)
    prefill_actual = torch.empty(
        num_rows,
        topk,
        device="cuda",
        dtype=torch.int32,
    )
    top_k_per_row_prefill_hygon_out(
        logits,
        starts,
        ends,
        prefill_actual,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        topk,
    )
    prefill_expected = torch.stack(
        [
            torch.topk(
                logits[row, int(starts[row].item()) : int(ends[row].item())],
                k=topk,
            ).indices.to(torch.int32)
            for row in range(num_rows)
        ]
    )
    _report_and_assert_same_index_set(
        "lightop_indexer_topk/prefill",
        prefill_actual,
        prefill_expected,
    )

    next_n = 2
    seq_lens = torch.tensor(
        [[3071, 3072], [3583, 3584]],
        device="cuda",
        dtype=torch.int32,
    )
    decode_actual = torch.empty_like(prefill_actual)
    top_k_per_row_decode_hygon_out(
        logits,
        next_n,
        seq_lens,
        decode_actual,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        topk,
    )
    decode_expected = torch.stack(
        [
            torch.topk(
                logits[row, : int(seq_lens.reshape(-1)[row].item())],
                k=topk,
            ).indices.to(torch.int32)
            for row in range(num_rows)
        ]
    )
    _report_and_assert_same_index_set(
        "lightop_indexer_topk/decode",
        decode_actual,
        decode_expected,
    )
