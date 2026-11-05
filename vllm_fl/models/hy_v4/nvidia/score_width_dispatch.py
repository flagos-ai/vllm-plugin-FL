# SPDX-License-Identifier: Apache-2.0
"""Bounded 16k/32k paged-score dispatch outside capture.

The current immutable input-step proof binds prefill request-local lengths.
PIECEWISE selection runs only inside the matching attention eager segment;
FULL and decode retain their original width. Score capacity uses the maximum
request length, while native flattened gather offsets use their sum.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_ATTENTION_CONTEXT = ContextVar("hy4_score_width_attention_context", default=None)


def _score_capacity(bound, width, topk):
    """Smallest qualified request-local capacity; no device-value reads.

    These are score columns, not a concatenated KV gather allocation. A
    selector still proves row/table/layout/step and uncaptured execution.
    FULL replay and unproven/decode metadata never enter this function.
    """
    if type(bound) is not int or type(width) is not int or type(topk) is not int:
        return None
    if bound <= 0 or topk <= 0:
        return None
    for capacity in (16384, 32768):
        if max(bound, topk) <= capacity < width:
            return capacity
    return None


@contextmanager
def score_width_attention_scope(context):
    """Limit PIECEWISE selection to this invocation of attention's eager body."""
    token = _ATTENTION_CONTEXT.set(context)
    try:
        yield
    finally:
        _ATTENTION_CONTEXT.reset(token)


def select_eager_score_capacity(
    indexer, context, q, weights, table, rows, lengths, output, width, page_size
):
    mode = getattr(context.cudagraph_runtime_mode, "name", None)
    if (
        not indexer._hy4_int8_bounded_score_enabled
        or not indexer._hy4_int8_empty_tile_enabled
        or not indexer._hy4_int8_prefix_topk_enabled
        or indexer._hy4_int8_paged_logits_symbol
        != "int8_indexer_paged_logits_empty_tile"
        or mode not in ("NONE", "PIECEWISE")
        or (mode == "PIECEWISE" and _ATTENTION_CONTEXT.get() is not context)
        or context.ubatch_slices is not None
        or context.dp_metadata is not None
        or torch.is_grad_enabled()
    ):
        return None
    if q.ndim != 3:
        return None
    m = q.shape[0]
    if m not in (256, 1024, 2048):
        return None
    descriptor = context.batch_descriptor
    if (
        descriptor is None
        or type(descriptor.num_tokens) is not int
        or descriptor.num_tokens != m
        or descriptor.num_reqs not in (None, 1)
        or descriptor.uniform
        or descriptor.has_lora
        or descriptor.num_active_loras
    ):
        return None
    metadata = context.attn_metadata
    meta = metadata.get(indexer.cache.prefix) if isinstance(metadata, dict) else None
    types = indexer._hy4_int8_bounded_score_metadata_types
    if (
        not isinstance(types, tuple)
        or len(types) != 2
        or not isinstance(meta, types[0])
        or type(meta.max_seq_len) is not int
        or not m <= meta.max_seq_len <= 32768
        or meta.num_prefills != 1
        or meta.num_decodes != 0
        or meta.num_decode_tokens != 0
        or meta.num_prefill_tokens != m
        or meta.prefill is None
        or len(meta.prefill.chunks) != 1
        or tuple(meta.seq_lens.shape) != (1,)
        or tuple(meta.slot_mapping.shape) != (m,)
        or meta.slot_mapping.dtype != torch.int64
        or meta.slot_mapping.stride() != (1,)
    ):
        return None
    chunk = meta.prefill.chunks[0]
    if (
        not isinstance(chunk, types[1])
        or chunk.num_reqs != 1
        or chunk.total_seq_lens != meta.max_seq_len
        or chunk.local_total_seq_lens != meta.max_seq_len
        or chunk.token_start != 0
        or chunk.token_end != m
        or chunk.block_table is not table
        or tuple(chunk.local_cu_seq_lens.shape) != (2,)
        or chunk.local_cu_seq_lens.dtype != torch.int32
        or tuple(chunk.cu_seqlen_ks.shape) != (m,)
        or tuple(chunk.cu_seqlen_ke.shape) != (m,)
        or chunk.cu_seqlen_ks.dtype != torch.int32
        or chunk.cu_seqlen_ke.dtype != torch.int32
    ):
        return None
    cache = indexer.cache.kv_cache
    if (
        tuple(q.shape) != (m, 32, 128)
        or q.dtype != torch.int8
        or not q.is_cuda
        or q.stride() != (4096, 128, 1)
        or tuple(weights.shape) != (m, 32)
        or weights.dtype != torch.float32
        or weights.stride() != (32, 1)
        or table.ndim != 2
        or table.shape[0] != 1
        or table.dtype != torch.int32
        or table.stride(1) != 1
        or table.stride(0) < table.shape[1]
        or tuple(rows.shape) != (m,)
        or rows.dtype != torch.int64
        or rows.stride() != (1,)
        or tuple(lengths.shape) != (m,)
        or lengths.dtype != torch.int32
        or lengths.stride() != (1,)
        or tuple(output.shape) != (m, 2048)
        or output.dtype != torch.int32
        or output.stride() != (2048, 1)
        or cache.ndim != 3
        or tuple(cache.shape[1:]) != (64, 132)
        or cache.dtype != torch.uint8
        or cache.stride() != (8448, 132, 1)
        or type(width) is not int
        or width <= 16384
        or page_size != 64
        or table.shape[1] * page_size < width
        or indexer.topk != 2048
    ):
        return None
    if any(
        x.device != q.device
        for x in (
            weights,
            table,
            rows,
            lengths,
            output,
            cache,
            meta.seq_lens,
            meta.slot_mapping,
            chunk.local_cu_seq_lens,
            chunk.cu_seqlen_ks,
            chunk.cu_seqlen_ke,
        )
    ):
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    return _score_capacity(meta.max_seq_len, width, indexer.topk)


def select_mixed_score_capacity(
    indexer, context, q, weights, table, rows, lengths, output, width, page_size
):
    """Narrow proven prefill rows in an uncaptured attention invocation.

    Request count and query count are not algorithm qualification keys. A
    current immutable step proof bounds each request-local score row. FULL
    and decode rows retain the original provider/width.
    """
    mode = getattr(context.cudagraph_runtime_mode, "name", None)
    if (
        not indexer._hy4_int8_bounded_score_enabled
        or not indexer._hy4_int8_empty_tile_enabled
        or not indexer._hy4_int8_prefix_topk_enabled
        or indexer._hy4_int8_paged_logits_symbol
        != "int8_indexer_paged_logits_empty_tile"
        or mode not in ("NONE", "PIECEWISE")
        or (mode == "PIECEWISE" and _ATTENTION_CONTEXT.get() is not context)
        or context.ubatch_slices is not None
        or context.dp_metadata is not None
        or torch.is_grad_enabled()
        or q.ndim != 3
    ):
        return None
    m = q.shape[0]
    desc = context.batch_descriptor
    if (
        type(m) is not int
        or m <= 0
        or desc is None
        or type(desc.num_tokens) is not int
        or desc.num_tokens < m
        or desc.has_lora
        or desc.num_active_loras
    ):
        return None
    types = indexer._hy4_int8_bounded_score_metadata_types
    metadata = context.attn_metadata
    meta = metadata.get(indexer.cache.prefix) if isinstance(metadata, dict) else None
    if (
        not isinstance(types, tuple)
        or len(types) != 2
        or not isinstance(meta, types[0])
        or type(meta.num_decodes) is not int
        or meta.num_decodes <= 0
        or type(meta.num_decode_tokens) is not int
        or meta.num_decode_tokens <= 0
        or type(meta.num_prefills) is not int
        or meta.num_prefills <= 0
        or type(meta.num_prefill_tokens) is not int
        or meta.num_prefill_tokens <= 0
        or meta.num_decode_tokens + meta.num_prefill_tokens > desc.num_tokens
    ):
        return None
    prefill = getattr(meta, "prefill", None)
    chunks = getattr(prefill, "chunks", ()) if prefill is not None else ()
    matching = [c for c in chunks if c.block_table is table]
    decode = getattr(meta, "decode", None)
    if len(matching) != 1 or (decode is not None and decode.block_table is table):
        return None
    chunk = matching[0]
    proof_type = indexer._hy4_int8_cpu_bound_proof_type
    current_fn = indexer._hy4_int8_current_cpu_proof
    current = current_fn() if callable(current_fn) else None
    proof = getattr(chunk, "cpu_seq_len_bound", None)
    if (
        not isinstance(chunk, types[1])
        or proof_type is None
        or not isinstance(proof, proof_type)
        or current is None
        or proof.step_proof is not current
        or meta.cpu_seq_lens_proof_generation != current.generation
        or not meta.num_decodes
        <= proof.request_start
        < proof.request_stop
        <= meta.num_decodes + meta.num_prefills
        or type(chunk.num_reqs) is not int
        or chunk.num_reqs != len(proof.seq_lens)
        or chunk.num_reqs != proof.request_stop - proof.request_start
        or (chunk.token_start, chunk.token_end) != (proof.chunk_start, proof.chunk_end)
        or not meta.num_decode_tokens
        <= chunk.token_start
        < chunk.token_end
        <= meta.num_decode_tokens + meta.num_prefill_tokens
        or chunk.token_end - chunk.token_start != m
        or chunk.total_seq_lens != sum(proof.seq_lens)
        or chunk.local_total_seq_lens != chunk.total_seq_lens
        or not proof.seq_lens
        or max(proof.seq_lens) > 32768
    ):
        return None
    cache = indexer.cache.kv_cache
    if (
        tuple(q.shape) != (m, 32, 128)
        or q.dtype != torch.int8
        or not q.is_cuda
        or q.stride() != (4096, 128, 1)
        or tuple(weights.shape) != (m, 32)
        or weights.dtype != torch.float32
        or weights.stride() != (32, 1)
        or table.ndim != 2
        or table.shape[0] != chunk.num_reqs
        or table.dtype != torch.int32
        or table.stride(1) != 1
        or table.stride(0) < table.shape[1]
        or tuple(rows.shape) != (m,)
        or rows.dtype != torch.int64
        or rows.stride() != (1,)
        or tuple(lengths.shape) != (m,)
        or lengths.dtype != torch.int32
        or lengths.stride() != (1,)
        or tuple(output.shape) != (m, 2048)
        or output.dtype != torch.int32
        or output.stride() != (2048, 1)
        or cache.ndim != 3
        or tuple(cache.shape[1:]) != (64, 132)
        or cache.dtype != torch.uint8
        or cache.stride() != (8448, 132, 1)
        or tuple(chunk.local_cu_seq_lens.shape) != (chunk.num_reqs + 1,)
        or chunk.local_cu_seq_lens.dtype != torch.int32
        or chunk.local_cu_seq_lens.stride() != (1,)
        or tuple(chunk.cu_seqlen_ks.shape) != (m,)
        or tuple(chunk.cu_seqlen_ke.shape) != (m,)
        or chunk.cu_seqlen_ks.dtype != torch.int32
        or chunk.cu_seqlen_ke.dtype != torch.int32
        or chunk.cu_seqlen_ks.stride() != (1,)
        or chunk.cu_seqlen_ke.stride() != (1,)
        or type(width) is not int
        or width <= 16384
        or page_size != 64
        or table.shape[1] * page_size < width
        or indexer.topk != 2048
    ):
        return None
    tensors = (
        weights,
        table,
        rows,
        lengths,
        output,
        cache,
        chunk.local_cu_seq_lens,
        chunk.cu_seqlen_ks,
        chunk.cu_seqlen_ke,
    )
    if (
        any(x.device != q.device for x in tensors)
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    return _score_capacity(max(proof.seq_lens), width, indexer.topk)
