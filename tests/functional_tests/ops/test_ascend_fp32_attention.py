# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP32 attention replay, lifecycle and RoPE contracts on an Ascend NPU."""

from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        getattr(current_platform, "vendor_name", None) != "ascend",
        reason="Requires Ascend NPU",
    ),
]


@pytest.mark.parametrize("limit", [4096, 4097])
@pytest.mark.parametrize("outer_dtype", [torch.float16, torch.bfloat16])
def test_fp32_decode_replay_and_owner_lifecycle(limit, outer_dtype):
    from vllm.forward_context import BatchDescriptor

    from vllm_fl.dispatch.backends.vendor.ascend.impl import fp32_attention as fp32

    n, heads, dim, block_size = 2, 2, 16, 128
    width = (limit + block_size - 1) // block_size
    cache = torch.randn(2, width * n, block_size, heads, dim, device="npu")
    layer = NS(layer_name="layer", kv_cache=cache)
    model = NS(num_heads=heads, num_kv_heads=heads, head_size=dim, scale=dim**-0.5)
    table = torch.arange(width * n, device="npu").reshape(n, width).int()
    lengths = torch.tensor([7, 13], dtype=torch.int32)
    slots = table[:, 0].long() * block_size + lengths.to("npu") - 1
    meta = NS(
        num_actual_tokens=n,
        seq_lens_list=[7, 13],
        seq_lens=lengths,
        slot_mapping=slots,
        block_tables=table,
        actual_seq_lengths_q=[1, 2],
    )
    descriptor = BatchDescriptor(num_tokens=n)
    wrapper = object.__new__(fp32.FP32GraphWrapper)
    wrapper.vllm_config = NS(model_config=NS(max_model_len=limit))
    wrapper.concrete_graph_entries = {}
    context = NS(
        attn_metadata={"layer": meta},
        batch_descriptor=descriptor,
        no_compile_layers={"layer": layer},
    )
    q, k, v = [torch.randn(n, heads, dim, device="npu") for _ in range(3)]
    output = torch.empty_like(q)
    with patch("vllm.forward_context.get_forward_context", return_value=context):
        assert fp32.prepare_graph(wrapper, context)
        state = wrapper._fp32_decode_states[descriptor]["layer"]
        with fp32.graph_owner(wrapper), torch.autocast("npu", dtype=outer_dtype):
            for _ in range(3):
                fp32._full_attention(layer, model, q, k, v, cache, meta, output)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                fp32._full_attention(layer, model, q, k, v, cache, meta, output)
            for step in range(2):
                for tensor in (q, k, v):
                    tensor.copy_(torch.randn_like(tensor))
                lengths.add_(1)
                slots.copy_(table[:, 0].long() * block_size + lengths.to("npu") - 1)
                meta.seq_lens_list = [8 + step, 14 + step]
                fp32.prepare_graph(wrapper, context)
                graph.replay()
                torch.npu.synchronize()
                expected = torch.empty_like(output)
                copied_cache = cache.clone()
                fp32._paged_attention(
                    q,
                    k,
                    v,
                    copied_cache[0],
                    copied_cache[1],
                    meta,
                    heads,
                    heads,
                    dim,
                    model.scale,
                    expected,
                )
                torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)
        wrapper.concrete_graph_entries[descriptor] = NS(graph=graph)
        layer.kv_cache = cache.clone()
        assert fp32.prepare_graph(wrapper, context)
        assert descriptor not in wrapper.concrete_graph_entries
        assert wrapper._fp32_decode_states[descriptor]["layer"] is not state
        fp32.FP32GraphWrapper.clear_graphs(wrapper)
        assert not wrapper._fp32_decode_states


@pytest.mark.parametrize("interleaved", [False, True])
def test_fp32_partial_rotary_matches_upstream(interleaved):
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    from vllm_fl.dispatch.backends.vendor.ascend.impl.rotary import (
        rotary_embedding_ascend,
    )

    cos_sin_cache = torch.randn(32, 4, device="npu")
    positions = torch.tensor([2, 7, 13], device="npu")
    q = torch.randn(3, 2, 8, device="npu")
    k = torch.randn(3, 1, 8, device="npu")
    expected = RotaryEmbedding.forward_static(
        positions, q.flatten(1), k.flatten(1), 8, 4, cos_sin_cache, not interleaved
    )
    cos, sin = cos_sin_cache.chunk(2, -1)
    actual = rotary_embedding_ascend(
        None,
        q[..., :4],
        k[..., :4],
        cos,
        sin,
        positions,
        interleaved,
    )
    for result, original, reference in zip(actual, (q, k), expected):
        combined = torch.cat((result, original[..., 4:]), -1)
        torch.testing.assert_close(combined.flatten(1), reference)


@pytest.mark.parametrize("unsupported", ["tp", "pcp", "dcp", "noncausal", "cross"])
def test_fp32_rejects_unsupported_attention(unsupported):
    from vllm.v1.attention.backend import AttentionType

    from vllm_fl.dispatch.backends.vendor.ascend.impl.fp32_attention import forward_fp32

    names = {
        "tp": "tensor_parallel_size",
        "pcp": "prefill_context_parallel_size",
        "dcp": "decode_context_parallel_size",
    }
    parallel = {name: 1 for name in names.values()}
    if unsupported in names:
        parallel[names[unsupported]] = 2
    model = NS(
        vllm_config=NS(parallel_config=NS(**parallel)),
        sliding_window=None,
        alibi_slopes=None,
        sinks=None,
        attn_type=(
            AttentionType.ENCODER_DECODER
            if unsupported == "cross"
            else AttentionType.DECODER
        ),
    )
    q = torch.zeros(1, 2, 16, device="npu")
    with pytest.raises(NotImplementedError):
        forward_fp32(
            model, None, q, q, q, None, NS(causal=unsupported != "noncausal"), q
        )


@pytest.mark.parametrize("slots", [(-1, -1), (-1, 0), (0, -1)])
@pytest.mark.parametrize("full_graph", [False, True])
def test_fp32_padding_preserves_cache(slots, full_graph):
    from vllm.forward_context import BatchDescriptor

    from vllm_fl.dispatch.backends.vendor.ascend.impl import fp32_attention as fp32

    heads, dim, block_size = 2, 16, 128
    cache = torch.randn(2, 2, block_size, heads, dim, device="npu")
    layer = NS(layer_name="layer", kv_cache=cache)
    model = NS(num_heads=heads, num_kv_heads=heads, head_size=dim, scale=dim**-0.5)
    meta = NS(
        num_actual_tokens=2,
        seq_lens_list=[1, 1],
        seq_lens=torch.ones(2, dtype=torch.int32),
        slot_mapping=torch.tensor(slots, device="npu", dtype=torch.int64),
        block_tables=torch.tensor([[0], [1]], device="npu", dtype=torch.int32),
        actual_seq_lengths_q=[1, 2],
    )
    descriptor = BatchDescriptor(num_tokens=2)
    wrapper = object.__new__(fp32.FP32GraphWrapper)
    wrapper.vllm_config = NS(model_config=NS(max_model_len=block_size))
    wrapper.concrete_graph_entries = {}
    context = NS(
        attn_metadata={"layer": meta},
        batch_descriptor=descriptor,
        no_compile_layers={"layer": layer},
    )
    q, k, v = [torch.randn(2, heads, dim, device="npu") for _ in range(3)]
    output = torch.empty_like(q)

    def eager():
        fp32._paged_attention(
            q,
            k,
            v,
            cache[0],
            cache[1],
            meta,
            heads,
            heads,
            dim,
            model.scale,
            output,
        )

    with patch("vllm.forward_context.get_forward_context", return_value=context):
        # Exercise the mandatory NONE-mode warmup before capture too.
        before = cache.clone()
        eager()
        expected = before.clone()
        for row, slot in enumerate(slots):
            if slot >= 0:
                expected[0].view(-1, heads, dim)[slot].copy_(k[row])
                expected[1].view(-1, heads, dim)[slot].copy_(v[row])
        torch.testing.assert_close(cache, expected, atol=0, rtol=0)
        graph = None
        if full_graph:
            assert fp32.prepare_graph(wrapper, context)
            with fp32.graph_owner(wrapper):
                fp32._full_attention(layer, model, q, k, v, cache, meta, output)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    fp32._full_attention(layer, model, q, k, v, cache, meta, output)
        # Reverse real/padded order across replays to catch a padded write
        # overwriting a real slot-zero update and stale captured masks.
        for active_slots in (slots, tuple(reversed(slots))):
            meta.slot_mapping.copy_(torch.tensor(active_slots, device="npu"))
            k.copy_(torch.randn_like(k))
            v.copy_(torch.randn_like(v))
            expected = cache.clone()
            for row, slot in enumerate(active_slots):
                if slot >= 0:
                    expected[0].view(-1, heads, dim)[slot].copy_(k[row])
                    expected[1].view(-1, heads, dim)[slot].copy_(v[row])
            if full_graph:
                fp32.prepare_graph(wrapper, context)
                graph.replay()
            else:
                eager()
            torch.npu.synchronize()
            torch.testing.assert_close(cache, expected, atol=0, rtol=0)
            assert torch.isfinite(output).all()
