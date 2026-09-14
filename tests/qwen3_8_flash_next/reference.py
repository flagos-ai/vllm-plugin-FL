"""Small, device-independent references for the Qwen3.8-Flash-Next path.

These functions deliberately avoid importing vLLM.  They are used by both the
CPU correctness tests and the optional CUDA/Triton comparisons, so a checkout
without vLLM can still validate the model's indexing and recurrence contracts.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB


def splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    """Match the prime sequence used by ``Qwen3_8FlashNextNGramEmbedding``."""

    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


def ple_hash_ids_reference(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    *,
    ngram_size: int,
    heads_per_ngram: int,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """Return per-token hash-table IDs using the exact PLE boundary rules.

    ``ngram_context`` is the ``ngram_size - 1`` token prefix for every
    request.  ``query_start_loc`` is the flattened request boundary array.
    The implementation mirrors the model code: EOS fills the left context and
    prevents n-grams from crossing a segment boundary.
    """

    input_ids = input_ids.reshape(-1).long()
    query_start_loc = query_start_loc.reshape(-1).long()
    ngram_context = ngram_context.long()
    if query_start_loc.numel() == 0:
        raise ValueError("query_start_loc must contain at least the end marker")
    num_reqs = query_start_loc.numel() - 1
    if ngram_context.shape != (num_reqs, ngram_size - 1):
        raise ValueError("ngram_context must be [requests, ngram_size - 1]")
    if int(query_start_loc[-1]) != input_ids.numel():
        raise ValueError("query_start_loc does not cover input_ids")
    total_tokens = input_ids.numel()
    if total_tokens == 0:
        return input_ids.new_empty((0, (ngram_size - 1) * heads_per_ngram))

    max_query_len = int(torch.diff(query_start_loc).max().item()) if num_reqs else 0
    packed = input_ids.new_full((num_reqs, max_query_len), eos_token_id)
    positions = torch.arange(total_tokens, device=input_ids.device)
    request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
    request_indices.clamp_(max=max(num_reqs - 1, 0))
    columns = positions - query_start_loc[request_indices]
    packed[request_indices, columns] = input_ids

    context = torch.cat([ngram_context, packed], dim=-1)
    positions_2d = torch.arange(context.shape[1], device=input_ids.device)
    eos_positions = torch.where(
        context == eos_token_id,
        positions_2d.unsqueeze(0),
        torch.full_like(context, -1),
    )
    previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
    previous_eos = torch.cat(
        [
            eos_positions.new_full((num_reqs, 1), -1),
            previous_eos_inclusive[:, :-1],
        ],
        dim=1,
    )
    position_in_segment = positions_2d.unsqueeze(0) - previous_eos - 1

    shifted = [context]
    for shift in range(1, ngram_size):
        source = positions_2d - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(num_reqs, -1)
        shifted_tokens = context.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        shifted.append(torch.where(valid, shifted_tokens, context.new_full((), eos_token_id)))

    if multipliers.numel() != ngram_size:
        raise ValueError("multipliers must have one entry per n-gram position")
    ngram_heads = (ngram_size - 1) * heads_per_ngram
    if vocab_sizes.numel() != ngram_heads or offsets.numel() != ngram_heads:
        raise ValueError("vocab_sizes and offsets must cover every hash head")

    adjusted_columns = columns + ngram_size - 1
    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for index in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[index] * multipliers[index])
        ids = torch.remainder(
            mixed.unsqueeze(-1), vocab_sizes[start:end]
        ) + offsets[start:end]
        blocks.append(ids[request_indices, adjusted_columns])
    return torch.cat(blocks, dim=-1)


def dilated_short_conv_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    dilation: int,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal SiLU depthwise dilated short-conv with explicit state update."""

    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x must be [tokens, hidden], weight must be [hidden, kernel]")
    if x.shape[1] != weight.shape[0] or dilation <= 0:
        raise ValueError("short-conv dimensions or dilation are invalid")
    hidden, kernel = weight.shape
    state_len = (kernel - 1) * dilation
    if state is None:
        state = x.new_zeros((hidden, state_len))
    if state.shape != (hidden, state_len):
        raise ValueError("state shape does not match the dilated kernel")

    current_state = state.clone()
    outputs = []
    for token in x:
        history = torch.cat([current_state, token.view(hidden, 1)], dim=-1)
        taps = history[:, ::dilation]
        # history has exactly (kernel - 1) * dilation + 1 values, so the
        # sampled tap matrix has precisely ``kernel`` columns.
        y = (taps * weight).sum(dim=-1)
        outputs.append(F.silu(y))
        if state_len:
            current_state = history[:, -state_len:]
    if outputs:
        output = torch.stack(outputs, dim=0)
    else:
        output = x.new_empty((0, hidden))
    return output, current_state


def logical_to_physical_slots_reference(
    block_table: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if block_size <= 0 or block_table.ndim != 2:
        raise ValueError("invalid QSA page table or block size")
    request_indices, logical_positions = torch.broadcast_tensors(
        request_indices.long(), logical_positions.long()
    )
    result = torch.full_like(logical_positions, -1)
    if block_table.numel() == 0:
        return result
    valid = (
        (request_indices >= 0)
        & (request_indices < block_table.shape[0])
        & (logical_positions >= 0)
    )
    logical_block = torch.div(logical_positions.clamp_min(0), block_size, rounding_mode="floor")
    valid &= logical_block < block_table.shape[1]
    if not valid.any():
        return result
    safe_req = request_indices.clamp(0, block_table.shape[0] - 1)
    safe_block = logical_block.clamp(0, block_table.shape[1] - 1)
    physical_block = block_table[safe_req, safe_block].long()
    valid &= physical_block >= 0
    result = torch.where(
        valid,
        physical_block * block_size + logical_positions.remainder(block_size),
        result,
    )
    return result


def compressed_slot_mapping_reference(
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    storage_block_size: int,
    compress_ratio: int,
) -> torch.Tensor:
    if storage_block_size <= 0 or compress_ratio <= 0:
        raise ValueError("invalid QSA cache geometry")
    compressed_positions = torch.div(
        logical_positions.clamp_min(0), compress_ratio, rounding_mode="floor"
    )
    slots = logical_to_physical_slots_reference(
        block_table,
        token_to_req,
        compressed_positions,
        storage_block_size,
    )
    valid = (logical_positions >= 0) & (
        (logical_positions + 1).remainder(compress_ratio) == 0
    )
    return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int64)


def qsa_store_cache_rows_reference(
    cache: torch.Tensor, slot_mapping: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    result = cache.clone()
    if rows.ndim == 3:
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[-1]):
        raise ValueError("rows and slots have incompatible shapes")
    page_size = cache.shape[1]
    for row, slot in zip(rows, slot_mapping.reshape(-1).tolist()):
        if 0 <= int(slot) < cache.shape[0] * page_size:
            physical = int(slot)
            result[physical // page_size, physical % page_size, 0].copy_(row)
    return result


def qsa_compress_groups_reference(
    raw_cache: torch.Tensor,
    raw_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for pooling one completed group per metadata row."""

    rows = token_to_req.numel()
    pooled = torch.zeros(
        (rows, 1, raw_cache.shape[-1]), dtype=raw_cache.dtype, device=raw_cache.device
    )
    first_positions = torch.zeros((rows, 3), dtype=torch.int64, device=raw_cache.device)
    raw_page_size = raw_cache.shape[1]
    for row in range(rows):
        request = int(token_to_req[row])
        end_position = int(logical_positions[row])
        slot = int(compressed_slots[row])
        valid = (
            0 <= request < raw_block_table.shape[0]
            and end_position >= compress_ratio - 1
            and slot >= 0
        )
        if not valid:
            continue
        values = []
        for group_offset in range(compress_ratio):
            position = end_position - (compress_ratio - 1 - group_offset)
            page = position // raw_page_size
            offset = position % raw_page_size
            if 0 <= page < raw_block_table.shape[1]:
                physical_page = int(raw_block_table[request, page])
                if 0 <= physical_page < raw_cache.shape[0]:
                    values.append(raw_cache[physical_page, offset, 0].float())
        if values:
            pooled[row, 0] = torch.stack(values).mean(dim=0).to(raw_cache.dtype)
        if rope_cache is None:
            # The text-only kernel stores the scalar first position into all
            # three lanes of the position tuple (the consumer treats them as
            # equal axes when no packed MRoPE cache is present).
            first_positions[row].fill_(end_position - compress_ratio + 1)
        else:
            first_position = end_position - compress_ratio + 1
            rope_page = first_position // rope_cache.shape[1]
            rope_offset = first_position % rope_cache.shape[1]
            if 0 <= rope_page < raw_block_table.shape[1]:
                physical_page = int(raw_block_table[request, rope_page])
                if 0 <= physical_page < rope_cache.shape[0]:
                    first_positions[row] = rope_cache[physical_page, rope_offset, 0]
    return pooled, first_positions


def qsa_mqa_paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    *,
    num_columns: int | None = None,
    score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for the positive-dot MQA indexer score kernel."""

    rows = q.shape[0]
    columns = page_table.shape[1] * k_cache.shape[1] if num_columns is None else num_columns
    logits = torch.full((rows, columns), -float("inf"), dtype=torch.float32, device=q.device)
    visible_blocks = torch.zeros((rows,), dtype=torch.int32, device=q.device)
    divisor = math.sqrt(q.shape[-1]) if score_scale is None else score_scale
    page_size = k_cache.shape[1]
    for row in range(rows):
        request = int(token_to_req[row])
        if not (0 <= request < page_table.shape[0]):
            continue
        query_position = int(query_positions[row])
        sequence_length = int(sequence_lengths[request])
        visible = min((query_position + 1) // compress_ratio, sequence_length // compress_ratio)
        visible_blocks[row] = visible
        for column in range(min(columns, max(visible, 0))):
            page = column // page_size
            offset = column % page_size
            if not (0 <= page < page_table.shape[1]):
                continue
            physical = int(page_table[request, page])
            if not (0 <= physical < k_cache.shape[0]):
                continue
            key = k_cache[physical, offset, 0].float()
            score = 0.0
            for head in range(q.shape[1]):
                score += max(float(torch.dot(q[row, head].float(), key)), 0.0)
            logits[row, column] = score / divisor
    return logits, visible_blocks


def expand_qsa_indices_reference(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Reference for compressed block -> token index expansion."""

    if token_topk % compress_ratio:
        raise ValueError("token_topk must be divisible by compress_ratio")
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    output = torch.full(
        (block_indices.shape[0], output_width), -1, dtype=torch.int32, device=block_indices.device
    )
    for row in range(block_indices.shape[0]):
        request = int(token_to_req[row])
        sequence_length = int(sequence_lengths[request]) if 0 <= request < sequence_lengths.numel() else 0
        query_position = int(query_positions[row])
        complete_blocks = min(
            min((query_position + 1) // compress_ratio, sequence_length // compress_ratio),
            block_topk,
        )
        expanded_count = complete_blocks * compress_ratio
        tail_start = ((query_position + 1) // compress_ratio) * compress_ratio
        tail_count = (query_position + 1) - tail_start
        for column in range(output_width):
            if column < expanded_count:
                block_rank = column // compress_ratio
                token = int(block_indices[row, block_rank]) * compress_ratio + column % compress_ratio
                valid = 0 <= token < sequence_length
            else:
                tail_offset = column - expanded_count
                valid = tail_offset < tail_count and tail_offset < compress_ratio - 1
                token = tail_start + tail_offset
                valid = valid and 0 <= token < sequence_length
            if valid:
                output[row, column] = token
    return output


def qsa_sparse_paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Reference for sparse GQA over selected logical token positions."""

    rows, query_heads, head_dim = q.shape
    kv_heads = k_cache.shape[2]
    group_size = query_heads // kv_heads
    scale = head_dim ** -0.5 if softmax_scale is None else softmax_scale
    output = torch.zeros_like(q)
    page_size = k_cache.shape[1]
    for row in range(rows):
        request = int(token_to_req[row])
        if not (0 <= request < block_table.shape[0]):
            continue
        positions = logical_indices[row].long()
        for query_head in range(query_heads):
            kv_head = query_head // group_size
            keys = []
            values = []
            for position in positions.tolist():
                if position < 0:
                    continue
                page = position // page_size
                offset = position % page_size
                if not (0 <= page < block_table.shape[1]):
                    continue
                physical = int(block_table[request, page])
                if not (0 <= physical < k_cache.shape[0]):
                    continue
                keys.append(k_cache[physical, offset, kv_head].float())
                values.append(v_cache[physical, offset, kv_head].float())
            if not keys:
                continue
            key_tensor = torch.stack(keys)
            value_tensor = torch.stack(values)
            scores = torch.matmul(key_tensor, q[row, query_head].float()) * scale
            probs = torch.softmax(scores, dim=0)
            output[row, query_head] = torch.matmul(probs, value_tensor).to(output.dtype)
    return output
