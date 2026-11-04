# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX QSA selection and attention policies."""

from importlib import import_module

import torch

from .shape_limits import max_prefill_rows
from vllm_fl.dispatch.backends.flaggems.impl import qwen3_8_flash_next as portable


def qsa_select_paged_tokens(
    q,
    k_cache,
    page_table,
    token_to_req,
    query_positions,
    sequence_lengths,
    token_topk,
    compress_ratio,
    out=None,
    num_columns=None,
    *,
    long_context=False,
    pack_rows=None,
):
    """Select with the measured non-TLE FGV paths; skip fully visible buckets.

    Small batches use the corrected native per-row selector. Large batches
    retain radix for short score buckets; long buckets prefer the per-row
    provider after measuring the actual score-width dependence.
    This local VENDOR policy leaves other FG/FGV consumers unchanged.
    """
    block_topk, width = portable._check_selection_geometry(compress_ratio, token_topk)
    count = q.shape[0]
    out = portable._selection_output(count, width, q.device, out)
    if count == 0:
        return out
    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if not isinstance(columns, int) or not 0 <= columns <= capacity:
        raise ValueError("QSA score width must fit the allocated cache capacity")
    if columns <= block_topk:
        # This graph bucket fits entirely in the selection budget. Expansion
        # applies device-side visibility, request validity and incomplete tails.
        indices = torch.arange(block_topk, dtype=torch.int32, device=q.device)
        indices = indices.unsqueeze(0).expand(count, -1)
    else:
        if (
            long_context
            and q.device.type == "cuda"
            and columns <= 17600
            and count <= max_prefill_rows()
            and q.dtype == k_cache.dtype == torch.bfloat16
            and q.shape[1:] == (4, 128)
            and q.stride(-1) == k_cache.stride(-1) == 1
            and (pack_rows is not None or 8 <= count <= 64)
        ):
            from flaggems_vllm.ops.qwen4.qsa_indexer_fast import (
                qsa_indexer_fast,
            )

            from .qsa_packing import indexer_launch

            scores, visible = qsa_indexer_fast(
                q,
                k_cache,
                page_table,
                token_to_req,
                query_positions,
                sequence_lengths,
                columns,
                pack_rows,
                **indexer_launch(count, columns, pack_rows is not None),
            )
        else:
            scores, visible = portable.qsa_mqa_paged(
                q,
                k_cache,
                page_table,
                token_to_req,
                query_positions,
                sequence_lengths,
                compress_ratio,
                num_columns=columns,
            )
        if block_topk == 512 and columns <= 32768:
            if count <= 128 or (
                long_context
                and count <= max_prefill_rows()
                and 8192 <= columns <= 17600
            ):
                native = import_module("flaggems_vllm.ops.top_k_per_row_decode")
                if native.HAS_TLE:
                    raise RuntimeError(
                        "Qwen QSA requires the validated FGV non-TLE selector; "
                        "disable per-row TLE before worker startup"
                    )
                indices = torch.empty(
                    (count, block_topk), device=scores.device, dtype=torch.int32
                )
                native.top_k_per_row_decode(
                    scores,
                    1,
                    visible,
                    indices,
                    count,
                    scores.stride(0),
                    scores.stride(1),
                    block_topk,
                )
            else:
                select = import_module(
                    "flaggems_vllm.runtime.backend._metax.ops.radix_topk"
                ).radix_topk
                indices = select(scores, visible, block_topk)
        else:
            # Preserve the original provider outside the measured QSA geometry.
            import flag_gems

            _, selected = flag_gems.topk(
                scores, block_topk, dim=-1, largest=True, sorted=False
            )
            ascending = torch.arange(block_topk, dtype=torch.int32, device=q.device)
            indices = torch.where(
                visible[:, None] <= block_topk, ascending, selected.to(torch.int32)
            )
    # QSA block/tail semantics are model-owned. Fuse only this composition;
    # keep individual FG operators available to all other consumers.
    expand = portable.expand_qsa_block_indices
    if indices.is_cuda and sequence_lengths.numel():
        from vllm_fl.models.qwen3_8_flash_next.gpu.ops.qsa import (
            expand_qsa_block_indices,
        )

        expand = expand_qsa_block_indices
    return expand(
        indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio,
        token_topk,
        out,
    )


def qsa_sparse_split_count(q, k_cache, topk):
    """Use graph-static split-KV only for the measured MetaX QSA bucket."""
    if (
        q.device.type == "cuda"
        and q.dtype == k_cache.dtype == torch.bfloat16
        and 0 < q.shape[0] <= 64
        and q.shape[1:] == (3, 256)
        and k_cache.ndim == 4
        and k_cache.shape[2:] == (1, 256)
        and topk == 2051
    ):
        return 16 if q.shape[0] <= 32 else 8
    return 1


def qsa_sparse_paged_attention(
    q,
    k,
    v,
    indices,
    table,
    mapping,
    scale,
    out,
    *,
    split_workspace=None,
    gate=None,
    packed_workspace_cache=None,
    pack_rows=None,
    score_columns=None,
):
    from .qsa_packing import packed_attention

    if packed_workspace_cache is not None and packed_attention(
        q,
        k,
        v,
        indices,
        table,
        mapping,
        scale,
        out,
        gate,
        packed_workspace_cache,
        pack_rows,
        score_columns=score_columns,
    ):
        return out
    # Preserve the fallback's initialized-output contract.
    out.zero_()
    portable.qsa_sparse_paged_attention(
        q,
        k,
        v,
        indices,
        table,
        mapping,
        scale,
        out,
        split_workspace=split_workspace,
    )
    if gate is not None:
        out.mul_(torch.sigmoid(gate))
    return out
