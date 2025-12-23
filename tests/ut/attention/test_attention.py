import math
from dataclasses import dataclass

import pytest
import torch

from vllm_fl.attention.attention import (
    AttentionFLBackend,
    AttentionFLImpl,
    AttentionFLMetadata,
)


class MockLayer(torch.nn.Module):
    def __init__(self, num_heads, num_kv_heads, head_size):
        super().__init__()
        self.register_buffer("_q_scale", torch.tensor(1.0, device="cuda"))
        self.register_buffer("_k_scale", torch.tensor(1.0, device="cuda"))
        self.register_buffer("_v_scale", torch.tensor(1.0, device="cuda"))


def create_kv_cache(num_blocks, block_size, num_kv_heads, head_size, dtype, device):
    return torch.zeros(
        (2, num_blocks, block_size, num_kv_heads, head_size), dtype=dtype, device=device
    )


def manual_attention_ref(q, k, v, scale):
    q = q.transpose(0, 1)
    k = k.transpose(0, 1).transpose(1, 2)
    v = v.transpose(0, 1)

    attn = torch.matmul(q, k) * scale
    attn = torch.softmax(attn, dim=-1)

    out = torch.matmul(attn, v)
    return out.transpose(0, 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("block_size", [16])
def test_prefill_attention(dtype, num_heads, num_kv_heads, head_size, block_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    torch.manual_seed(42)

    batch_size = 2
    seq_lens = [32, 45]
    max_seq_len = max(seq_lens)
    scale = 1.0 / (head_size**0.5)

    total_tokens = sum(seq_lens)
    query = torch.randn(total_tokens, num_heads, head_size, dtype=dtype, device=device)
    key = torch.randn(total_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
    value = torch.randn(
        total_tokens, num_kv_heads, head_size, dtype=dtype, device=device
    )

    num_blocks_per_seq = [(sl + block_size - 1) // block_size for sl in seq_lens]
    total_blocks = sum(num_blocks_per_seq)

    block_table = torch.zeros(
        batch_size, max(num_blocks_per_seq), dtype=torch.int32, device=device
    )
    slot_mapping = torch.empty(total_tokens, dtype=torch.long, device=device)

    block_allocator = 0
    token_offset = 0

    ref_q_list = []
    ref_k_list = []
    ref_v_list = []

    for i, seq_len in enumerate(seq_lens):
        blocks_needed = num_blocks_per_seq[i]
        block_ids = list(range(block_allocator, block_allocator + blocks_needed))
        block_table[i, :blocks_needed] = torch.tensor(block_ids, device=device)
        block_allocator += blocks_needed

        current_seq_slots = []
        for t in range(seq_len):
            b_idx = block_ids[t // block_size]
            b_offset = t % block_size
            current_seq_slots.append(b_idx * block_size + b_offset)

        slot_mapping[token_offset : token_offset + seq_len] = torch.tensor(
            current_seq_slots, device=device
        )

        # Data for ref (padding later)
        q_slice = query[token_offset : token_offset + seq_len]
        k_slice = key[token_offset : token_offset + seq_len]
        v_slice = value[token_offset : token_offset + seq_len]

        # GQA Expand for Ref if needed
        if num_kv_heads != num_heads:
            ratio = num_heads // num_kv_heads
            k_slice = k_slice.repeat_interleave(ratio, dim=1)
            v_slice = v_slice.repeat_interleave(ratio, dim=1)

        ref_q_list.append(
            torch.nn.functional.pad(q_slice, (0, 0, 0, 0, 0, max_seq_len - seq_len))
        )
        ref_k_list.append(
            torch.nn.functional.pad(k_slice, (0, 0, 0, 0, 0, max_seq_len - seq_len))
        )
        ref_v_list.append(
            torch.nn.functional.pad(v_slice, (0, 0, 0, 0, 0, max_seq_len - seq_len))
        )

        token_offset += seq_len

    ref_q = torch.stack(ref_q_list).transpose(1, 2)  # [B, H, S, D]
    ref_k = torch.stack(ref_k_list).transpose(1, 2)
    ref_v = torch.stack(ref_v_list).transpose(1, 2)

    query_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.cumsum(torch.tensor(seq_lens, device=device), 0)

    attn_metadata = AttentionFLMetadata(
        num_actual_tokens=total_tokens,
        max_query_len=max_seq_len,
        query_start_loc=query_start_loc,
        max_seq_len=max_seq_len,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        block_table=block_table,
        slot_mapping=slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        scheduler_metadata=None,
        max_num_splits=0,
        causal=True,
    )

    kv_cache = create_kv_cache(
        total_blocks, block_size, num_kv_heads, head_size, dtype, device
    )

    impl = AttentionFLImpl(
        num_heads, head_size, scale, num_kv_heads, None, None, "auto"
    )
    layer = MockLayer(num_heads, num_kv_heads, head_size)
    output = torch.empty_like(query)

    impl.forward(layer, query, key, value, kv_cache, attn_metadata, output)

    # Run Ref (SDPA handles causal masking automatically for squared input)
    ref_out_padded = torch.nn.functional.scaled_dot_product_attention(
        ref_q, ref_k, ref_v, is_causal=True, scale=scale
    ).transpose(
        1, 2
    )  # [B, S, H, D]

    ref_out_flat_list = []
    for i, seq_len in enumerate(seq_lens):
        ref_out_flat_list.append(ref_out_padded[i, :seq_len, :, :])
    ref_out_flat = torch.cat(ref_out_flat_list, dim=0)

    atol = 2e-2 if dtype == torch.bfloat16 else 2e-3
    assert torch.allclose(output, ref_out_flat, atol=atol), "Prefill output mismatch"


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_size", [128])
def test_decode_attention(dtype, num_heads, head_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    block_size = 16
    num_kv_heads = 4
    scale = 1.0 / (head_size**0.5)

    batch_size = 2
    past_lens = [32, 45]

    total_past_tokens = sum(past_lens)
    num_blocks_per_seq = [(pl + 1 + block_size - 1) // block_size for pl in past_lens]
    total_blocks = sum(num_blocks_per_seq)

    kv_cache = create_kv_cache(
        total_blocks, block_size, num_kv_heads, head_size, dtype, device
    )
    block_table = torch.zeros(
        batch_size, max(num_blocks_per_seq), dtype=torch.int32, device=device
    )
    block_allocator = 0

    # Lists to store full KV history for reference
    context_k_list = []
    context_v_list = []

    # 1. Setup Past KV Cache
    for i, pl in enumerate(past_lens):
        hist_k = torch.randn(pl, num_kv_heads, head_size, dtype=dtype, device=device)
        hist_v = torch.randn(pl, num_kv_heads, head_size, dtype=dtype, device=device)
        context_k_list.append(hist_k)
        context_v_list.append(hist_v)

        blocks = list(range(block_allocator, block_allocator + num_blocks_per_seq[i]))
        block_table[i, : len(blocks)] = torch.tensor(blocks, device=device)
        block_allocator += num_blocks_per_seq[i]

        for t in range(pl):
            b_idx = blocks[t // block_size]
            b_off = t % block_size
            kv_cache[0, b_idx, b_off] = hist_k[t]
            kv_cache[1, b_idx, b_off] = hist_v[t]

    # 2. Prepare Current Step Input
    query = torch.randn(
        batch_size, num_heads, head_size, dtype=dtype, device=device
    )  # [B, H, D] flattened
    curr_k = torch.randn(
        batch_size, num_kv_heads, head_size, dtype=dtype, device=device
    )
    curr_v = torch.randn(
        batch_size, num_kv_heads, head_size, dtype=dtype, device=device
    )

    # Slot Mapping for current step
    slot_mapping = torch.empty(batch_size, dtype=torch.long, device=device)
    for i, pl in enumerate(past_lens):
        t = pl
        b_idx = block_table[i, t // block_size]
        b_off = t % block_size
        slot_mapping[i] = b_idx * block_size + b_off

        # Update Reference Context
        context_k_list[i] = torch.cat([context_k_list[i], curr_k[i : i + 1]])
        context_v_list[i] = torch.cat([context_v_list[i], curr_v[i : i + 1]])

    # 3. Run Impl
    seq_lens_tensor = torch.tensor(
        [pl + 1 for pl in past_lens], dtype=torch.int32, device=device
    )
    query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32, device=device)

    attn_metadata = AttentionFLMetadata(
        num_actual_tokens=batch_size,
        max_query_len=1,
        query_start_loc=query_start_loc,
        max_seq_len=max(past_lens) + 1,
        seq_lens=seq_lens_tensor,
        block_table=block_table,
        slot_mapping=slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_num_splits=0,
        causal=True,
    )

    impl = AttentionFLImpl(
        num_heads, head_size, scale, num_kv_heads, None, None, "auto"
    )
    layer = MockLayer(num_heads, num_kv_heads, head_size)

    # Output buffer must be [Total_Tokens, Heads, Dim] for impl, where Total=BatchSize here
    output_buffer = torch.zeros(
        batch_size, num_heads, head_size, dtype=dtype, device=device
    )

    impl.forward(
        layer=layer,
        query=query.view(batch_size, num_heads, head_size),
        key=curr_k,
        value=curr_v,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output_buffer,
    )

    # 4. Run Reference (Per Sample loop to avoid masking hell)
    ref_outputs = []
    for i in range(batch_size):
        # Q: [1, H, D]
        q_item = query[i : i + 1]

        # K, V: [Seq, KV_H, D] -> GQA expand -> [Seq, H, D]
        k_item = context_k_list[i]
        v_item = context_v_list[i]

        if num_kv_heads != num_heads:
            ratio = num_heads // num_kv_heads
            k_item = k_item.repeat_interleave(ratio, dim=1)
            v_item = v_item.repeat_interleave(ratio, dim=1)

        # Manual Attention
        out_item = manual_attention_ref(q_item, k_item, v_item, scale)
        ref_outputs.append(out_item)

    ref_out_flat = torch.cat(ref_outputs, dim=0).squeeze(1)  # [B, H, D]

    # 5. Compare
    assert torch.allclose(
        output_buffer, ref_out_flat, atol=1e-3, rtol=1e-3
    ), f"Decode mismatch. Diff: {(output_buffer - ref_out_flat).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__])
