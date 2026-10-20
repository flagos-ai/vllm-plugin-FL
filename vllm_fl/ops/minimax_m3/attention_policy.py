# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

from vllm_fl.minimax_m3_config import load, thaw


def choose_index(total_q, batch, max_query_len, max_seq_len):
    p = load()
    fw = p.framework
    a = p.attention
    if not (0 < max_seq_len <= fw.max_context):
        return None
    if (
        max_query_len == 1
        and total_q == batch
        and str(total_q) in a["index_q1_chunks"]
        and max_seq_len >= fw.long_index_min_context
    ):
        return dict(
            mode="vector",
            chunks=a["index_q1_chunks"][str(total_q)],
            ktile=128,
            warps=4,
            stages=1,
        )
    if (
        fw.long_prefill_query_lower < max_query_len <= fw.long_prefill_query_upper
        and max_query_len <= total_q <= fw.long_prefill_query_upper
        and max_seq_len >= fw.long_index_min_context
    ):
        return thaw(a["long_index"])
    return None


def choose_prefill(total_q, max_query_len):
    p = load()
    fw = p.framework
    if (
        fw.long_prefill_query_lower < max_query_len <= fw.long_prefill_query_upper
        and max_query_len <= total_q <= fw.long_prefill_query_upper
    ):
        return thaw(p.attention["long_prefill"])
    return None


def choose_decode(total_q):
    a = load().attention
    if str(total_q) in a["decode_parts"]:
        return dict(
            page_parts=a["decode_parts"][str(total_q)], ktile=16, warps=1, stages=1
        )
    return None
