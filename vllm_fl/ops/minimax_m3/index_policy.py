# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

from vllm_fl.minimax_m3_config import load, thaw


def choose(total_q, max_query_len, max_seq_len):
    p = load().attention["base_index"]
    if max_query_len == 1:
        return thaw(p["vector"])
    if max_query_len <= p["small_query_max"] and total_q <= p["small_total_max"]:
        bq, chunks = p["small_bq"], p["small_chunks"]
    elif max_query_len <= p["medium_query_max"] and total_q <= p["medium_total_max"]:
        bq, chunks = p["medium_bq"], p["medium_chunks"]
    elif max_seq_len > p["long_context_min_exclusive"]:
        bq, chunks = p["default_bq"], p["long_chunks"]
    elif max_seq_len <= p["short_context_max"] and total_q >= p["large_total_min"]:
        bq, chunks = p["default_bq"], p["short_chunks"]
    else:
        bq, chunks = p["default_bq"], p["default_chunks"]
    return dict(
        mode="dot", bq=bq, ktile=p["dot_ktile"], chunks=chunks, warps=p["dot_warps"]
    )
