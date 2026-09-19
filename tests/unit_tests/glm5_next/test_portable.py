# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

from vllm_fl.kernels.glm5_next import portable


def test_safe_gate_matches_fp32_definition() -> None:
    raw = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
    a_log = torch.tensor([math.log(0.5), math.log(2.0)])
    bias = torch.tensor([[0.25, -0.5], [0.0, 1.0]])

    actual = portable.safe_kda_gate(raw, a_log, 2, bias, lower_bound=-5.0)
    expected = -5.0 * torch.sigmoid(
        torch.exp(a_log).reshape(1, 2, 1) * (raw.reshape(1, 2, 2) + bias)
    )

    torch.testing.assert_close(actual, expected)


def test_safe_gate_chunk_cumsum_resets_at_sequence_and_chunk() -> None:
    raw = torch.zeros(1, 7, 1, 1)
    a_log = torch.zeros(1)
    boundaries = torch.tensor([0, 3, 7], dtype=torch.int32)

    actual = portable.safe_kda_gate_chunk_cumsum(
        raw,
        a_log,
        cu_seqlens=boundaries,
        chunk_size=2,
    ).flatten()
    gate = -2.5 / math.log(2.0)
    expected = torch.tensor([gate, 2 * gate, gate, gate, 2 * gate, gate, 2 * gate])

    torch.testing.assert_close(actual, expected)


def test_recurrent_kda_matches_independent_loop() -> None:
    torch.manual_seed(7)
    q = torch.randn(1, 4, 2, 3)
    k = torch.randn_like(q)
    v = torch.randn(1, 4, 2, 5)
    gate = -torch.rand_like(q)
    beta = torch.rand(1, 4, 2)
    initial = torch.randn(1, 2, 3, 5)
    scale = 0.25

    state = initial[0].float().clone()
    expected = torch.empty_like(v)
    for token in range(q.shape[1]):
        state *= torch.exp(gate[0, token].float()).unsqueeze(-1)
        residual = v[0, token].float() - torch.einsum(
            "hk,hkv->hv", k[0, token].float(), state
        )
        state += torch.einsum(
            "hk,hv->hkv",
            beta[0, token].float().unsqueeze(-1) * k[0, token].float(),
            residual,
        )
        expected[0, token] = torch.einsum(
            "hk,hkv->hv", q[0, token].float() * scale, state
        )

    actual, final_state = portable.recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        scale=scale,
        initial_state=initial,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(final_state[0], state)


def test_portable_causal_conv_prefill_and_decode_update_state() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    weight = torch.tensor([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]])
    state = torch.zeros(1, 2, 2)
    boundaries = torch.tensor([0, 3], dtype=torch.int32)
    state_ids = torch.tensor([0], dtype=torch.int32)

    actual = portable.causal_conv1d_fn(
        x,
        weight,
        None,
        state,
        boundaries,
        cache_indices=state_ids,
        has_initial_state=torch.tensor([False]),
        activation=None,
    )
    expected = torch.tensor([[100.0, 210.0, 321.0], [800.0, 1080.0, 1308.0]])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state[0], torch.tensor([[2.0, 3.0], [5.0, 6.0]]))

    decoded = portable.causal_conv1d_update(
        torch.tensor([[7.0, 8.0]]),
        state,
        weight,
        conv_state_indices=state_ids,
        activation=None,
    )
    torch.testing.assert_close(decoded, torch.tensor([[732.0, 1730.0]]))
    torch.testing.assert_close(state[0], torch.tensor([[3.0, 7.0], [6.0, 8.0]]))


def test_hadamard128_is_normalized_and_self_inverse() -> None:
    torch.manual_seed(11)
    value = torch.randn(3, 128)
    transformed = portable.hadamard128(value)
    restored = portable.hadamard128(transformed)

    torch.testing.assert_close(restored, value, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        transformed.square().sum(-1),
        value.square().sum(-1),
        atol=2e-5,
        rtol=2e-5,
    )


def test_fwht_quant_matches_materialized_reference() -> None:
    try:
        portable.get_fp8_dtype_and_max()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    torch.manual_seed(13)
    query = torch.randn(7, 128, dtype=torch.bfloat16)
    rotated = portable.hadamard128(query).to(torch.bfloat16).float()
    expected_q, expected_scale = portable._quantize_fp8_vector(
        rotated, round_scale=True
    )
    actual_q, actual_scale = portable.fwht128_quant_fp8(query)

    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    torch.testing.assert_close(actual_scale, expected_scale.unsqueeze(-1))


def test_tail_seed_keeps_each_requests_last_pool() -> None:
    kpool = 4
    tail = torch.full((12, 2, kpool, 3), float("nan"))
    slots = torch.tensor(
        [
            *(5 * kpool + position % kpool for position in range(6)),
            *(9 * kpool + position % kpool for position in range(7)),
        ],
        dtype=torch.int64,
    )
    keys = torch.arange(slots.numel() * 3, dtype=torch.float32).reshape(-1, 3)
    scores = keys + 1000

    portable.kpool_seed_tail_cache(tail, keys, scores, slots, kpool)

    # Request 0 retains positions 2..5, request 1 retains positions 3..6.
    for block, start, length, offset in ((5, 2, 6, 0), (9, 3, 7, 6)):
        for position in range(start, length):
            source = offset + position
            ring_offset = position % kpool
            torch.testing.assert_close(tail[block, 0, ring_offset], keys[source])
            torch.testing.assert_close(tail[block, 1, ring_offset], scores[source])


def test_kpool_fp8_cache_layout_matches_returned_compression() -> None:
    try:
        portable.get_fp8_dtype_and_max()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    torch.manual_seed(19)
    cache = torch.zeros(1, 2, 132, dtype=torch.uint8)
    keys = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    scores = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    ape = torch.randn(4, 128)
    locations = torch.tensor([1], dtype=torch.int64)

    try:
        quantized, scales = portable.kpool_compress_and_write_cache(
            cache,
            keys,
            scores,
            ape,
            locations,
            pool_size=4,
            return_compressed=True,
        )
    except RuntimeError as exc:
        pytest.skip(f"CPU float8 conversion is unavailable: {exc}")

    cache_values, cache_scales = portable._cache_views(cache, 128)
    torch.testing.assert_close(
        cache_values[0, 1], quantized[0].view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(cache_scales[0, 1, 0], scales[0])


def test_pool_expansion_and_tail_append_keep_request_local_indices() -> None:
    pools = torch.tensor([[2, 5], [0, 3]], dtype=torch.int32)
    valid = torch.tensor([[True, False], [True, True]])
    expanded = portable.expand_pools_to_tokens(pools, valid, topk=8, pool_size=4)
    expected = torch.tensor(
        [[8, 9, 10, 11, -1, -1, -1, -1], [0, 1, 2, 3, 12, 13, 14, 15]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(expanded, expected)

    with_tail = portable.append_tail_to_topk(
        expanded,
        seq_lens=torch.tensor([14, 18]),
        pool_lens=torch.tensor([3, 4]),
        pool_size=4,
    )
    torch.testing.assert_close(
        with_tail[:, -3:],
        torch.tensor([[12, 13, -1], [16, 17, -1]], dtype=torch.int32),
    )
