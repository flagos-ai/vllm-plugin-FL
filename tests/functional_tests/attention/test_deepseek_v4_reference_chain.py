# SPDX-License-Identifier: Apache-2.0
"""Paged compressed-attention microcontracts on NPU, without model weights."""

import math

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(current_platform.device_type != "npu", reason="Requires Ascend"),
]


@pytest.fixture(autouse=True)
def reference_dispatch():
    from vllm_fl.dispatch import get_default_manager, with_strict_mode

    with with_strict_mode():
        for name in ("save_state", "compress", "indexer", "sparse_attention"):
            assert (
                get_default_manager().get_selected_impl_id("deepseek_v4_" + name)
                == "reference.torch"
            )
        yield


def _call(name, *args, **kwargs):
    from vllm_fl.dispatch import call_op

    return call_op("deepseek_v4_" + name, *args, **kwargs)


PAGES = [[6, 0, 4, 2], [7, 1, 5, 3]]


def _state_case(ratio):
    block = 4 if ratio == 4 else 64
    n, d, coff = 2 * ratio, 4, 2 if ratio == 4 else 1
    pos_list = list(range(n)) * 2
    slots = [
        PAGES[req][p // block] * block + p % block for req in range(2) for p in range(n)
    ]
    values = torch.arange(2 * n * coff * d, device="npu").reshape(2 * n, coff * d)
    # Projection outputs are FP32; rounding them before state storage changes math.
    kv = ((values % 37 - 18) * 0.03).float()
    gate = ((values % 13 - 6) * 0.07).float()
    ape = (
        torch.arange(ratio * coff * d, device="npu").reshape(ratio, coff * d)
        % 17
        * 0.01
    ).float()
    cache = torch.zeros(8, block, 2 * coff * d, device="npu", dtype=torch.float32)
    positions = torch.tensor(pos_list, device="npu", dtype=torch.int64)
    slot_mapping = torch.tensor(slots, device="npu", dtype=torch.int64)
    table = torch.tensor(PAGES, device="npu", dtype=torch.int32)
    weight = torch.tensor([0.7, 1.1, 0.8, 1.3], device="npu")
    angle = torch.arange(n, device="npu").float() * 0.013
    rope = torch.stack([angle.cos(), angle.sin()], dim=-1)
    return kv, gate, ape, positions, slot_mapping, cache, table, weight, rope


def _compress_oracle(
    cache, reqs, ends, weight, rope, ratio, output_dtype=torch.bfloat16
):
    d, block = weight.numel(), cache.shape[1]
    coff = 2 if ratio == 4 else 1
    outputs = []
    for req, end in zip(reqs, ends):
        if req < 0 or end < 0:
            outputs.append(torch.zeros_like(weight))
            continue
        start = end - ratio * coff + 1
        rows, scores = [], []
        for time in range(max(0, start), end + 1):
            raw = cache[PAGES[req][time // block], time % block]
            half = int(coff == 2 and time >= end - ratio + 1)
            rows.append(raw[half * d : (half + 1) * d])
            scores.append(raw[coff * d + half * d : coff * d + (half + 1) * d])
        # Independently compute each channel's weighted history reduction.
        result = torch.stack(
            [
                sum(
                    row[c] * score
                    for row, score in zip(
                        rows, torch.softmax(torch.stack([s[c] for s in scores]), dim=0)
                    )
                )
                for c in range(d)
            ]
        )
        result = result * torch.rsqrt(result.square().mean() + 1e-6) * weight
        cosine, sine = rope[(end // ratio) * ratio]
        a, b = result[-2], result[-1]
        outputs.append(
            torch.stack(
                [result[0], result[1], a * cosine - b * sine, b * cosine + a * sine]
            )
        )
    return torch.stack(outputs).to(output_dtype)


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("chunked", [False, True])
def test_save_state_chunking_and_nan_padding(ratio, chunked):
    kv, gate, ape, pos, slots, cache, *_ = _state_case(ratio)
    expected = cache.clone()
    for i in range(kv.shape[0]):
        slot = int(slots[i].item())
        expected[slot // cache.shape[1], slot % cache.shape[1]] = torch.cat(
            [
                kv[i].float(),
                gate[i].float() + ape[i % (2 * ratio) % ratio],
            ]
        )
    if chunked:
        cuts = [0, 3, kv.shape[0] - 1, kv.shape[0]]
        for start, end in zip(cuts, cuts[1:]):
            _call(
                "save_state",
                kv[start:end],
                gate[start:end],
                ape,
                pos[start:end],
                slots[start:end],
                cache,
                ratio,
            )
    else:
        _call("save_state", kv, gate, ape, pos, slots, cache, ratio)
    before = cache.clone()
    _call(
        "save_state",
        torch.full_like(kv[:1], float("nan")),
        torch.full_like(gate[:1], float("nan")),
        ape,
        torch.tensor([-1], device="npu"),
        torch.tensor([-1], device="npu"),
        cache,
        ratio,
    )
    torch.testing.assert_close(cache, before, rtol=0, atol=0)
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    assert torch.isfinite(cache).all()


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("boundary", [1, 2])
def test_compression_boundaries_and_pages(ratio, boundary):
    kv, gate, ape, pos, slots, cache, table, weight, rope = _state_case(ratio)
    _call("save_state", kv, gate, ape, pos, slots, cache, ratio)
    reqs, ends = [0, 1, -1], [ratio * boundary - 1, ratio - 1, -1]
    dtype = torch.bfloat16 if boundary == 1 else torch.float16
    output = _call(
        "compress",
        cache,
        table,
        torch.tensor(reqs, device="npu"),
        torch.tensor(ends, device="npu"),
        weight,
        rope,
        compress_ratio=ratio,
        output_dtype=dtype,
    )
    expected = _compress_oracle(cache, reqs, ends, weight, rope, ratio, dtype)
    assert output.dtype == dtype
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
    assert torch.count_nonzero(output[-1]) == 0


def _keys(dtype=torch.bfloat16):
    cache = torch.arange(4 * 4 * 4, device="npu").reshape(4, 4, 4) % 23 - 11
    return cache.to(dtype), torch.tensor(
        [[2, 0], [3, 1]], device="npu", dtype=torch.int32
    )


@pytest.mark.parametrize("quantized", [False, True])
def test_indexer_causality_and_int8_scales(quantized):
    keys, table = _keys(torch.int8 if quantized else torch.bfloat16)
    q = torch.tensor([[[2, -1, 3, 1], [-1, 2, 1, 4]]] * 3, device="npu")
    q = q.to(torch.int8 if quantized else torch.bfloat16)
    weights = torch.tensor([[0.3, 0.7]] * 3, device="npu")
    qscale = torch.tensor([[0.2, 0.4]] * 3, device="npu") if quantized else None
    kscale = torch.full((4, 4, 1), 0.125, device="npu") if quantized else None
    reqs, ends, lengths = [0, 1, -1], [7, 3, -1], [3, 3]
    actual = _call(
        "indexer",
        q,
        keys,
        weights,
        table,
        torch.tensor(reqs, device="npu"),
        torch.tensor(ends, device="npu"),
        torch.tensor(lengths, device="npu"),
        3,
        query_scale=qscale,
        key_scale_cache=kscale,
    )
    expected = torch.full((3, 3), -1, dtype=torch.int32, device="npu")
    for row, (req, end) in enumerate(zip(reqs, ends)):
        if req < 0:
            continue
        count = min(lengths[req], (end + 1) // 4)
        scores = []
        for key in range(count):
            physical = [2, 3][req] if key < 4 else [0, 1][req]
            keyrow = keys[physical, key % 4].float()
            query = q[row].float()
            if quantized:
                query = query * qscale[row, :, None]
                keyrow = keyrow * kscale[physical, key % 4]
            scores.append(((query * keyrow).sum(-1).relu() * weights[row]).sum())
        expected[row, :count] = torch.stack(scores).argsort(descending=True).int()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("quantized", [False, True])
def test_indexer_query_tiles_match_independent_requests(quantized):
    keys, table = _keys()
    q = torch.arange(39 * 2 * 4, device="npu").reshape(39, 2, 4) % 7 + 1
    q = q.to(torch.int8 if quantized else torch.bfloat16)
    if quantized:
        keys = keys.to(torch.int8)
    weights = torch.tensor([[0.3, 0.7]] * 39, device="npu")
    reqs = torch.tensor([0, 1, -1] * 13, device="npu")
    ends = torch.tensor([7, 3, -1] * 13, device="npu")
    lengths = torch.tensor([3, 3], device="npu")
    scales = torch.full((39, 2), 0.25, device="npu") if quantized else None
    key_scales = torch.full((4, 4, 1), 0.125, device="npu") if quantized else None
    actual = _call(
        "indexer",
        q,
        keys,
        weights,
        table,
        reqs,
        ends,
        lengths,
        3,
        query_scale=None if scales is None else scales.flatten(),
        key_scale_cache=key_scales,
    )
    expected = torch.cat(
        [
            _call(
                "indexer",
                q[row : row + 1],
                keys,
                weights[row : row + 1],
                table,
                reqs[row : row + 1],
                ends[row : row + 1],
                lengths,
                3,
                query_scale=None if scales is None else scales[row : row + 1],
                key_scale_cache=key_scales,
            )
            for row in range(39)
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _attention_oracle(q, swa, compact, tables, reqs, ends, sink, window, value_dim):
    result = torch.zeros((*q.shape[:2], value_dim), device="npu", dtype=torch.float32)
    swa_table, compact_table = tables
    for row, (req, end) in enumerate(zip(reqs, ends)):
        if req < 0 or end < 0:
            continue
        values = [
            swa[swa_table[req][i // 4], i % 4].float()
            for i in range(max(0, end - window + 1), min(end + 1, 6))
        ]
        if compact is not None:
            # Test selects [0,1,3,-1]; causal compact bound excludes future 3.
            values += [
                compact[compact_table[req][i // 4], i % 4].float()
                for i in (0, 1, 3)
                if i < min(2, (end + 1) // 4)
            ]
        if not values:
            continue
        matrix = torch.stack(values)
        for head in range(q.shape[1]):
            logits = (matrix * q[row, head].float()).sum(-1) / math.sqrt(4)
            denominator = logits.exp().sum()
            if sink is not None:
                denominator += sink[head].exp()
            result[row, head] = (logits.exp()[:, None] * matrix[:, :value_dim]).sum(
                0
            ) / denominator
    return result.to(q.dtype)


@pytest.mark.parametrize("with_sink", [False, True])
@pytest.mark.parametrize("swa_only", [False, True])
def test_combined_attention_masks_sink_and_swa_only(with_sink, swa_only):
    swa, table = _keys()
    swa = swa * 0.03
    compact = None if swa_only else (swa + 0.2)
    q = torch.full((3, 2, 4), 0.4, device="npu", dtype=torch.bfloat16)
    reqs, ends = [0, 1, -1], [5, 3, -1]
    sink = torch.tensor([0.5, -0.2], device="npu") if with_sink else None
    indices = None if swa_only else torch.tensor([[0, 1, 3, -1]] * 3, device="npu")
    lengths = torch.tensor([6, 6], device="npu")
    actual = _call(
        "sparse_attention",
        q,
        swa,
        compact,
        table,
        None if swa_only else table,
        torch.tensor(reqs, device="npu"),
        torch.tensor(ends, device="npu"),
        lengths,
        None if swa_only else torch.tensor([2, 2], device="npu"),
        indices,
        3,
        compress_ratio=0 if swa_only else 4,
        attn_sink=sink,
        value_dim=3,
    )
    expected = _attention_oracle(
        q, swa, compact, ([[2, 0], [3, 1]], [[2, 0], [3, 1]]), reqs, ends, sink, 3, 3
    )
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    assert torch.count_nonzero(actual[-1]) == 0


@pytest.mark.parametrize("ratio", [4, 128])
def test_integrated_prefill_decode_chain(ratio):
    kv, gate, ape, pos, slots, cache, table, weight, rope = _state_case(ratio)
    cuts = [0, ratio - 1, 2 * ratio - 1, kv.shape[0]]
    for start, end in zip(cuts, cuts[1:]):
        _call(
            "save_state",
            kv[start:end],
            gate[start:end],
            ape,
            pos[start:end],
            slots[start:end],
            cache,
            ratio,
        )
    reqs = torch.tensor([0, 1], device="npu")
    end = 2 * ratio - 1
    ends = torch.tensor([end, end], device="npu")
    compressed = _call(
        "compress",
        cache,
        table,
        torch.tensor([0, 0, 1, 1], device="npu"),
        torch.tensor([ratio - 1, end, ratio - 1, end], device="npu"),
        weight,
        rope,
        compress_ratio=ratio,
    )
    torch.testing.assert_close(
        compressed,
        _compress_oracle(
            cache, [0, 0, 1, 1], [ratio - 1, end, ratio - 1, end], weight, rope, ratio
        ),
        rtol=0.01,
        atol=0.01,
    )
    compact = torch.zeros(4, 4, 4, device="npu", dtype=torch.bfloat16)
    compact[2, :2], compact[3, :2] = compressed[:2], compressed[2:]
    q = torch.full((2, 2, 4), 0.25, device="npu", dtype=torch.bfloat16)
    compact_table = torch.tensor([[2], [3]], device="npu", dtype=torch.int32)
    counts = torch.full((2,), 2, device="npu", dtype=torch.int32)
    if ratio == 4:
        indices = _call(
            "indexer",
            q,
            compact,
            torch.ones(2, 2, device="npu"),
            compact_table,
            reqs,
            ends,
            counts,
            2,
        )
        torch.testing.assert_close(
            indices.sort(dim=-1).values,
            torch.tensor([[0, 1], [0, 1]], dtype=torch.int32, device="npu"),
        )
    else:
        # C128 uses every completed compact row and has no learned indexer.
        indices = torch.tensor([[0, 1], [0, 1]], device="npu", dtype=torch.int32)
    swa = cache[..., :4].to(torch.bfloat16) * 0.03
    swa_table = table
    result = _call(
        "sparse_attention",
        q,
        swa,
        compact,
        swa_table,
        compact_table,
        reqs,
        ends,
        torch.full((2,), 2 * ratio, device="npu"),
        counts,
        indices,
        3,
        compress_ratio=ratio,
    )
    expected = torch.zeros_like(q)
    for row in range(2):
        # Retain both streams even if their underlying token coverage overlaps.
        original_rows = [
            swa[PAGES[row][i // swa.shape[1]], i % swa.shape[1]].float()
            for i in range(max(0, end - 2), end + 1)
        ]
        values = torch.stack(
            original_rows + list(compressed[row * 2 : row * 2 + 2].float())
        )
        logits = (q[row].float() @ values.t()) / 2
        expected[row] = (logits.softmax(-1) @ values).to(q.dtype)
    torch.testing.assert_close(result, expected, rtol=0.01, atol=0.01)


def test_compress_fixed_shape_changed_state_graph():
    import torch_npu  # noqa: F401

    kv, gate, ape, pos, slots, cache, table, weight, rope = _state_case(4)
    _call("save_state", kv, gate, ape, pos, slots, cache, 4)
    reqs = torch.tensor([0, 1], device="npu")
    ends = torch.tensor([7, 3], device="npu")

    def run():
        return _call("compress", cache, table, reqs, ends, weight, rope)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = run()
    for _ in range(2):
        cache[..., :8].add_(0.07)
        expected = _compress_oracle(cache, [0, 1], [7, 3], weight, rope, 4)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)


@pytest.mark.parametrize("ratio", [4, 128])
def test_compression_query_tiles_match_independent_requests(ratio):
    kv, gate, ape, pos, slots, cache, table, weight, rope = _state_case(ratio)
    _call("save_state", kv, gate, ape, pos, slots, cache, ratio)
    reqs = torch.tensor([0, 1, -1] * 13, device="npu")
    ends = torch.tensor([ratio - 1, 2 * ratio - 1, -1] * 13, device="npu")
    actual = _call("compress", cache, table, reqs, ends, weight, rope, ratio)
    expected = torch.cat(
        [
            _call(
                "compress",
                cache,
                table,
                reqs[row : row + 1],
                ends[row : row + 1],
                weight,
                rope,
                ratio,
            )
            for row in range(39)
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_attention_query_tiles_match_oracle():
    swa, table = _keys()
    swa = swa * 0.03
    compact = swa + 0.2
    q = torch.full((39, 2, 4), 0.4, device="npu", dtype=torch.bfloat16)
    reqs, ends = [0, 1, -1] * 13, [5, 3, -1] * 13
    sink = torch.tensor([0.5, -0.2], device="npu")
    indices = torch.tensor([[0, 1, 3, -1]] * 39, device="npu")
    actual = _call(
        "sparse_attention",
        q,
        swa,
        compact,
        table,
        table,
        torch.tensor(reqs, device="npu"),
        torch.tensor(ends, device="npu"),
        torch.tensor([6, 6], device="npu"),
        torch.tensor([2, 2], device="npu"),
        indices,
        3,
        compress_ratio=4,
        attn_sink=sink,
        value_dim=3,
    )
    tables = ([[2, 0], [3, 1]], [[2, 0], [3, 1]])
    expected = _attention_oracle(q, swa, compact, tables, reqs, ends, sink, 3, 3)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)


def test_empty_chain_inputs():
    kv, gate, ape, pos, slots, cache, table, weight, rope = _state_case(4)
    _call("save_state", kv[:0], gate[:0], ape, pos[:0], slots[:0], cache, 4)
    output = _call("compress", cache, table, pos[:0], pos[:0], weight, rope)
    assert output.shape == (0, 4)
    keys, key_table = _keys()
    q = torch.empty(0, 2, 4, device="npu", dtype=torch.bfloat16)
    lengths = torch.tensor([2, 2], device="npu")
    ids = _call(
        "indexer",
        q,
        keys,
        torch.empty(0, 2, device="npu"),
        key_table,
        pos[:0],
        pos[:0],
        lengths,
        2,
    )
    assert ids.shape == (0, 2)
    attention = _call(
        "sparse_attention",
        q,
        keys,
        None,
        key_table,
        None,
        pos[:0],
        pos[:0],
        lengths,
        None,
        None,
        3,
        compress_ratio=0,
    )
    assert attention.shape == (0, 2, 4)


def test_save_real_slot_zero_with_nan_padding():
    cache = torch.full((2, 4, 16), -17.0, device="npu")
    kv = torch.stack(
        [
            torch.arange(8, device="npu").float() + 0.25,
            torch.full((8,), float("nan"), device="npu"),
        ]
    )
    gate = kv * 0.2
    ape = torch.full((4, 8), 0.125, device="npu")
    expected = cache.clone()
    expected[0, 0] = torch.cat([kv[0], gate[0] + ape[1]])
    _call(
        "save_state",
        kv,
        gate,
        ape,
        torch.tensor([1, -1], device="npu"),
        torch.tensor([0, -1], device="npu"),
        cache,
        4,
    )
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    assert torch.isfinite(cache).all()


def test_save_noncontiguous_page_backing_preserves_padding():
    backing = torch.full((2, 4, 20), 123.0, device="npu")
    cache = backing[..., :16]
    assert not cache.is_contiguous()
    kv = torch.arange(16, device="npu").reshape(2, 8).float() * 0.03
    gate = kv * -0.5
    ape = torch.full((4, 8), 0.0625, device="npu")
    expected = backing.clone()
    for row, slot in enumerate((0, 5)):
        expected[slot // 4, slot % 4, :16] = torch.cat([kv[row], gate[row] + ape[row]])
    _call(
        "save_state",
        kv,
        gate,
        ape,
        torch.tensor([0, 1], device="npu"),
        torch.tensor([0, 5], device="npu"),
        cache,
        4,
    )
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        backing[..., 16:], torch.full_like(backing[..., 16:], 123)
    )


def test_full_reference_pipeline_changed_inputs_graph():
    import torch_npu  # noqa: F401

    kv, gate, ape, pos, slots, cache, table, weight, rope = _state_case(4)
    compressed_reqs = torch.tensor([0, 0, 1, 1], device="npu")
    compressed_ends = torch.tensor([3, 7, 3, 7], device="npu")
    reqs = torch.tensor([0, 1], device="npu")
    ends = torch.tensor([7, 7], device="npu")
    compact = torch.zeros(4, 4, 4, device="npu", dtype=torch.bfloat16)
    compact_table = torch.tensor([[2], [3]], device="npu", dtype=torch.int32)
    compact_lengths = torch.full((2,), 2, device="npu", dtype=torch.int32)
    swa_lengths = torch.full((2,), 8, device="npu", dtype=torch.int32)
    q = torch.full((2, 2, 4), 0.25, device="npu", dtype=torch.bfloat16)
    head_weights = torch.ones(2, 2, device="npu")

    def run():
        _call("save_state", kv, gate, ape, pos, slots, cache, 4)
        compressed = _call(
            "compress", cache, table, compressed_reqs, compressed_ends, weight, rope
        )
        compact[2, :2].copy_(compressed[:2])
        compact[3, :2].copy_(compressed[2:])
        ids = _call(
            "indexer",
            q,
            compact,
            head_weights,
            compact_table,
            reqs,
            ends,
            compact_lengths,
            2,
        )
        swa = cache[..., :4].to(torch.bfloat16) * 0.03
        result = _call(
            "sparse_attention",
            q,
            swa,
            compact,
            table,
            compact_table,
            reqs,
            ends,
            swa_lengths,
            compact_lengths,
            ids,
            3,
        )
        return result, compressed, ids

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, compressed_output, selected = run()
    torch.npu.synchronize()
    previous = output.clone()
    for _ in range(2):
        kv.add_(0.03125)
        gate.mul_(1.03)
        q.add_(torch.tensor([0.1, -0.05, 0.08, -0.04], device="npu", dtype=q.dtype))
        expected = tuple(value.clone() for value in run())
        oracle_compressed = _compress_oracle(
            cache, [0, 0, 1, 1], [3, 7, 3, 7], weight, rope, 4
        )
        oracle_output = torch.empty_like(q)
        for req in range(2):
            swa_values = [
                (cache[PAGES[req][i // 4], i % 4, :4].to(q.dtype) * 0.03).float()
                for i in (5, 6, 7)
            ]
            matrix = torch.stack(
                swa_values + list(oracle_compressed[req * 2 : req * 2 + 2].float())
            )
            logits = q[req].float() @ matrix.t() / 2
            oracle_output[req] = (logits.softmax(-1) @ matrix).to(q.dtype)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected[0], rtol=0.01, atol=0.01)
        torch.testing.assert_close(compressed_output, expected[1], rtol=0.01, atol=0.01)
        torch.testing.assert_close(selected, expected[2])
        torch.testing.assert_close(output, oracle_output, rtol=0.01, atol=0.01)
        torch.testing.assert_close(
            compressed_output, oracle_compressed, rtol=0.01, atol=0.01
        )
        assert torch.isfinite(output).all()
        assert not torch.equal(output, previous)
        previous = output.clone()
