# Copyright (c) 2026 BAAI. All rights reserved.

"""PPU FA3 regressions for paged GQA and reusable AOT graph metadata.

These exercise the vendor wheel on THead hardware. Standard CUDA runners must
skip explicitly: an upstream NVIDIA FA3 build is not the PPU kernel under test.
The numerical oracle is float32 CPU SDPA with a bottom-right causal mask.
"""

import importlib
import math

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def ppu_attention():
    if not torch.cuda.is_available():
        pytest.skip("THead PPU accelerator required")
    pytest.importorskip("flash_attn_3._C", reason="PPU flash_attn_3 wheel required")
    pytest.importorskip("vllm", reason="vLLM required for the PPU bridge")
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "thead":
        pytest.skip("THead PPU FA3 kernel required; standard CUDA FA3 is excluded")
    # An import failure on the intended platform is an adaptation failure.
    return importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.thead.impl.attention"
    )


def _paged_inputs(head_dim, query_lens, capacity=2057, dtype=torch.bfloat16):
    """Use shuffled pages and the vLLM 0.28 packed physical HND cache."""
    batch_size = len(query_lens)
    num_query_heads, num_kv_heads, block_size = 8, 2, 16
    pages_per_request = math.ceil(capacity / block_size)
    num_blocks = batch_size * pages_per_request
    generator = torch.Generator().manual_seed(1729 + head_dim + batch_size)
    block_table_cpu = (
        torch.randperm(num_blocks, generator=generator)
        .reshape(batch_size, pages_per_request)
        .to(torch.int32)
    )
    cache_shape = (num_blocks, num_kv_heads, block_size, 2 * head_dim)
    packed_cache_cpu = torch.randn(cache_shape, generator=generator).to(dtype)
    # Exactly the inherited FlashAttentionImpl split: these K/V tensors share
    # storage and their token/head strides include the unused other half of KV.
    _, value_cache_cpu = packed_cache_cpu.transpose(1, 2).split(head_dim, dim=-1)
    # A position-dependent value component makes truncated/wrong KV ranges
    # visible even when the random values average out over thousands of keys.
    position_signal = torch.linspace(-1.0, 1.0, pages_per_request * block_size)
    for request in range(batch_size):
        for page in range(pages_per_request):
            block = int(block_table_cpu[request, page])
            signal = position_signal[page * block_size : (page + 1) * block_size]
            value_cache_cpu[block] += signal[:, None, None].to(dtype)
    query_cpu = torch.randn(
        (sum(query_lens), num_query_heads, head_dim), generator=generator
    ).to(dtype)
    query_starts = [0]
    for length in query_lens:
        query_starts.append(query_starts[-1] + length)
    packed_cache = packed_cache_cpu.cuda()
    key_cache, value_cache = packed_cache.transpose(1, 2).split(head_dim, dim=-1)
    return {
        "query": query_cpu.cuda(),
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table_cpu.cuda(),
        "cu_seqlens_q": torch.tensor(query_starts, dtype=torch.int32, device="cuda"),
        "query_lens": query_lens,
    }


def _sdpa_oracle(inputs, sequence_lens):
    query = inputs["query"].cpu().float()
    key_cache = inputs["key_cache"].cpu()
    value_cache = inputs["value_cache"].cpu()
    block_table = inputs["block_table"].cpu().long()
    block_size = key_cache.shape[1]
    num_query_heads, head_dim = query.shape[1:]
    num_kv_heads = key_cache.shape[2]
    outputs = []
    query_start = 0
    for request, (query_len, sequence_len) in enumerate(
        zip(inputs["query_lens"], sequence_lens, strict=True)
    ):
        pages = block_table[request, : math.ceil(sequence_len / block_size)]
        keys = key_cache.index_select(0, pages).flatten(0, 1)[:sequence_len].float()
        values = value_cache.index_select(0, pages).flatten(0, 1)[:sequence_len].float()
        keys = keys.transpose(0, 1).repeat_interleave(
            num_query_heads // num_kv_heads, dim=0
        )
        values = values.transpose(0, 1).repeat_interleave(
            num_query_heads // num_kv_heads, dim=0
        )
        queries = query[query_start : query_start + query_len].transpose(0, 1)
        # is_causal=True in SDPA uses upper-left alignment for rectangular QK.
        # FA3 paged decode instead aligns query positions with the end of KV.
        query_positions = torch.arange(query_len) + sequence_len - query_len
        mask = torch.arange(sequence_len)[None, :] <= query_positions[:, None]
        output = F.scaled_dot_product_attention(
            queries[None],
            keys[None],
            values[None],
            attn_mask=mask,
            dropout_p=0.0,
            scale=head_dim**-0.5,
        )
        outputs.append(output[0].transpose(0, 1))
        query_start += query_len
    return torch.cat(outputs)


def _metadata(bridge, inputs, sequence_lens, num_splits):
    return bridge._thead_get_scheduler_metadata(
        batch_size=len(inputs["query_lens"]),
        max_seqlen_q=max(inputs["query_lens"]),
        # Match the inherited vLLM builder: the PPU fwd bridge independently
        # forces max_seqlen_k=1. This regression detects inconsistent scheduling.
        max_seqlen_k=max(sequence_lens),
        num_heads_q=inputs["query"].shape[1],
        num_heads_kv=inputs["key_cache"].shape[2],
        headdim=inputs["query"].shape[2],
        cache_seqlens=torch.tensor(sequence_lens, device="cuda", dtype=torch.int32),
        qkv_dtype=inputs["query"].dtype,
        cu_seqlens_q=inputs["cu_seqlens_q"],
        page_size=inputs["key_cache"].shape[1],
        causal=True,
        num_splits=num_splits,
    )


def _forward(
    bridge,
    inputs,
    sequence_lens,
    *,
    metadata=None,
    num_splits=0,
    output=None,
    sequence_tensor=None,
):
    if sequence_tensor is None:
        sequence_tensor = torch.tensor(sequence_lens, device="cuda", dtype=torch.int32)
    return bridge._thead_flash_attn_varlen_func(
        q=inputs["query"],
        k=inputs["key_cache"],
        v=inputs["value_cache"],
        max_seqlen_q=max(inputs["query_lens"]),
        cu_seqlens_q=inputs["cu_seqlens_q"],
        max_seqlen_k=max(sequence_lens),
        seqused_k=sequence_tensor,
        block_table=inputs["block_table"],
        causal=True,
        softmax_scale=inputs["query"].shape[2] ** -0.5,
        scheduler_metadata=metadata,
        num_splits=num_splits,
        out=output,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize(
    "query_lens,sequence_lens",
    [
        ((1,), (2057,)),
        ((1, 3), (1025, 2057)),
    ],
)
@pytest.mark.parametrize("num_splits", [0, 4])
def test_paged_long_kv_eager_and_aot_match_sdpa(
    ppu_attention, head_dim, query_lens, sequence_lens, num_splits
):
    inputs = _paged_inputs(head_dim, query_lens)
    expected = _sdpa_oracle(inputs, sequence_lens)
    eager = _forward(ppu_attention, inputs, sequence_lens, num_splits=num_splits)
    scheduler = _metadata(ppu_attention, inputs, sequence_lens, num_splits)
    aot = _forward(
        ppu_attention, inputs, sequence_lens, metadata=scheduler, num_splits=num_splits
    )
    torch.cuda.synchronize()
    for name, output in (("eager", eager), ("AOT(actual max KV)", aot)):
        torch.testing.assert_close(
            output.cpu().float(),
            expected,
            rtol=0.015,
            atol=0.003,
            msg=lambda message, name=name: (
                f"PPU FA3 {name}, head_dim={head_dim}: {message}"
            ),
        )
    torch.testing.assert_close(aot, eager, rtol=0.015, atol=0.003)


@pytest.mark.gpu
def test_aot_metadata_buffer_refreshes_long_kv_on_graph_replay(ppu_attention):
    inputs = _paged_inputs(256, (1, 3))
    initial_lens, next_lens = (1025, 2057), (2037, 769)
    num_splits = 4
    initial_metadata = _metadata(ppu_attention, inputs, initial_lens, num_splits)
    assert initial_metadata.dtype == torch.int32
    assert initial_metadata.is_cuda and initial_metadata.is_contiguous()
    assert initial_metadata.ndim == 1 and initial_metadata.numel() > 0
    # The inherited vLLM FULL graph builder uses four vectors per rounded batch
    # and one semaphore. Check the vendor result fits its persistent allocation.
    buffer_size = 1 + math.ceil(len(initial_lens) / 4) * 4 * 4
    assert initial_metadata.numel() <= buffer_size
    metadata_buffer = torch.zeros(buffer_size, device="cuda", dtype=torch.int32)
    metadata_buffer[: initial_metadata.numel()].copy_(initial_metadata)
    captured_metadata = metadata_buffer[: initial_metadata.numel()]
    sequence_tensor = torch.tensor(initial_lens, device="cuda", dtype=torch.int32)
    output = torch.empty_like(inputs["query"])
    main_stream, warmup_stream = torch.cuda.current_stream(), torch.cuda.Stream()
    warmup_stream.wait_stream(main_stream)
    with torch.cuda.stream(warmup_stream):
        _forward(
            ppu_attention,
            inputs,
            initial_lens,
            metadata=captured_metadata,
            num_splits=num_splits,
            output=output,
            sequence_tensor=sequence_tensor,
        )
    main_stream.wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _forward(
            ppu_attention,
            inputs,
            initial_lens,
            metadata=captured_metadata,
            num_splits=num_splits,
            output=output,
            sequence_tensor=sequence_tensor,
        )

    for lengths in (initial_lens, next_lens, initial_lens):
        refreshed = _metadata(ppu_attention, inputs, lengths, num_splits)
        assert refreshed.shape == initial_metadata.shape
        metadata_buffer.zero_()
        captured_metadata.copy_(refreshed)
        sequence_tensor.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.cpu().float(), _sdpa_oracle(inputs, lengths), rtol=0.015, atol=0.003
        )
