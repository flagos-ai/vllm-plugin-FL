# Copyright (c) 2026 BAAI. All rights reserved.
"""Attach CPU request spans at metadata construction; no forward-time D2H."""
import functools


def attach_request_spans(chunk, start_idx, end_idx, query_start_loc_cpu,
                         compressed_seq_lens_cpu):
    if chunk is None:
        return chunk
    # vLLM's builder guarantees the CPU lengths are exact for prefill rows;
    # decode/speculative upper bounds must not be sent through this function.
    if query_start_loc_cpu.device.type != 'cpu' or compressed_seq_lens_cpu.device.type != 'cpu':
        return chunk
    q = query_start_loc_cpu[start_idx:end_idx + 1].tolist()
    lengths = compressed_seq_lens_cpu[start_idx:end_idx].tolist()
    offset = 0
    spans = []
    for begin, end, length in zip(q[:-1], q[1:], lengths):
        lo, hi = max(begin, chunk.token_start), min(end, chunk.token_end)
        if lo < hi:
            spans.append((lo - chunk.token_start, hi - chunk.token_start, offset, offset + length))
        offset += length
    if offset != chunk.total_seq_lens:
        return chunk
    chunk.hygon_request_spans = tuple(spans)
    # The exact device ends are retained, including graph-replay updates.
    # Localize once per metadata build, not in each transformer layer.
    if len(spans) > 1:
        chunk.hygon_local_ends = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks
        chunk.hygon_local_starts = chunk.cu_seqlen_ks - chunk.cu_seqlen_ks
    return chunk


def patch_prefill_indexer_metadata():
    from vllm.v1.attention.backends.mla import indexer
    original = indexer.build_prefill_chunk_metadata
    if getattr(original, '_hygon_request_spans', False):
        return

    @functools.wraps(original)
    def build(start_idx, end_idx, query_start_loc, query_start_loc_cpu,
              uncompressed_seq_lens, compressed_seq_lens, compressed_seq_lens_cpu,
              block_table, compress_ratio, query_slice=None, skip_kv_gather=False):
        result = original(start_idx, end_idx, query_start_loc, query_start_loc_cpu,
                          uncompressed_seq_lens, compressed_seq_lens, compressed_seq_lens_cpu,
                          block_table, compress_ratio, query_slice, skip_kv_gather)
        return attach_request_spans(result, start_idx, end_idx, query_start_loc_cpu,
                                    compressed_seq_lens_cpu)

    build._hygon_request_spans = True
    indexer.build_prefill_chunk_metadata = build
