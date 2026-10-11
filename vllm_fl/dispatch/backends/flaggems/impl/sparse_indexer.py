# SPDX-License-Identifier: Apache-2.0
"""FlagGems sparse-indexer operations used by the GLM BF16 contract."""


def rotate_indexer_query(q):
    from flag_gems.ops.hadamard_transform import hadamard_transform

    return hadamard_transform(q.float(), scale=128**-0.5).to(q.dtype)


def mqa_logits(*args, **kwargs):
    from flaggems_vllm import fp8_fp4_mqa_logits

    return fp8_fp4_mqa_logits(*args, **kwargs)


def gather_cache(*args, **kwargs):
    from flaggems_vllm import cp_gather_indexer_k_bf16_cache

    return cp_gather_indexer_k_bf16_cache(*args, **kwargs)


def paged_mqa_logits(q, *args, **kwargs):
    from flaggems_vllm import bf16_paged_mqa_logits_graph_safe

    if q[1] is not None:
        raise ValueError("Query scales must already be folded into weights")
    if "max_model_len" in kwargs:
        kwargs["max_context_len"] = kwargs.pop("max_model_len")
    return bf16_paged_mqa_logits_graph_safe(q[0], *args, **kwargs)


def persist_prefill_tail(*args, **kwargs):
    from flaggems_vllm import persist_prefill_tail as flaggems_impl

    return flaggems_impl(*args, **kwargs)


def kpool_compress_and_write_cache(*args, **kwargs):
    from flaggems_vllm import kpool_compress_and_write_cache as flaggems_impl

    return flaggems_impl(*args, **kwargs)


def kpool_decode_update_and_maybe_write_cache_batched(*args, **kwargs):
    from flaggems_vllm import (
        kpool_decode_update_and_maybe_write_cache_batched as flaggems_impl,
    )

    return flaggems_impl(*args, **kwargs)
