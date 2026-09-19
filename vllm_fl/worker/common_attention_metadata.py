# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

from vllm_fl.compilation.graph import Graph

logger = init_logger(__name__)

_GRAPH_DEVICE_TYPES = frozenset({"cuda", "npu", "musa", "ptpu"})


def supports_accelerator_graph() -> bool:
    """Return whether the active platform exposes the plugin graph API."""
    return (
        current_platform.device_type in _GRAPH_DEVICE_TYPES
        and hasattr(Graph, "graph")
        and hasattr(current_platform.torch_device_fn, "graph")
    )


# Adapted from vLLM's BlockTable slot-mapping kernel. The padding boundary is
# read from query_start_loc on device so one captured graph can replay with
# different request lengths without a host-derived launch argument.
@triton.jit(do_not_specialize=["max_num_tokens"])
def _compute_slot_mapping_graph_kernel(
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    slot_mapping_ptr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx == tl.num_programs(0) - 1:
        actual_num_tokens = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
        for i in range(actual_num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    # Padded request rows are not refreshed by BlockTable.commit_block_table().
    # Clear them in the existing per-group producer instead of launching one
    # eager fill per cache group. seq_lens is a fixed-address device buffer, so
    # the same captured shape can replay with a different actual request count.
    # A graph-padded row has no scheduled query tokens. Do not use seq_len as
    # the predicate: valid scheduler rows can transiently carry seq_len == 0.
    if start_idx == end_idx:
        row_offset = req_idx * block_table_stride
        for i in range(0, block_table_stride, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                block_table_ptr + row_offset + offsets,
                NULL_BLOCK_ID,
                mask=offsets < block_table_stride,
            )

    virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        block_indices = pos // virtual_block_size
        block_numbers = tl.load(block_table_ptr + row_offset + block_indices).to(
            tl.int64
        )

        virtual_block_offsets = pos - block_indices * virtual_block_size
        is_local = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        local_block_offsets = (
            virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )

        slot_ids = block_numbers * block_size + local_block_offsets
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


@triton.jit
def _compute_num_computed_tokens_kernel(
    query_start_loc_ptr,
    seq_lens_ptr,
    num_computed_tokens_ptr,
    num_reqs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_reqs
    query_start = tl.load(query_start_loc_ptr + offsets, mask=mask, other=0)
    query_end = tl.load(query_start_loc_ptr + offsets + 1, mask=mask, other=0)
    seq_len = tl.load(seq_lens_ptr + offsets, mask=mask, other=0)
    tl.store(
        num_computed_tokens_ptr + offsets,
        seq_len - (query_end - query_start),
        mask=mask,
    )


def compute_common_attention_metadata(
    block_table: Any,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None:
    """Populate fixed-address metadata buffers consumed by attention backends."""
    for table in block_table.block_tables:
        total_cp_world_size = table.pcp_world_size * table.dcp_world_size
        total_cp_rank = table.pcp_rank * table.dcp_world_size + table.dcp_rank
        _compute_slot_mapping_graph_kernel[(num_reqs + 1,)](
            table.max_num_batched_tokens,
            query_start_loc,
            positions,
            table.block_table.gpu,
            table.block_table.gpu.stride(0),
            table.block_size,
            table.slot_mapping.gpu,
            TOTAL_CP_WORLD_SIZE=total_cp_world_size,
            TOTAL_CP_RANK=total_cp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=table.cp_kv_cache_interleave_size,
            NULL_BLOCK_ID=NULL_BLOCK_ID,
            PAD_ID=PAD_SLOT_ID,
            BLOCK_SIZE=1024,
        )
    _compute_num_computed_tokens_kernel[(triton.cdiv(num_reqs, 256),)](
        query_start_loc,
        seq_lens,
        num_computed_tokens,
        num_reqs=num_reqs,
        BLOCK_SIZE=256,
    )


class CommonAttentionMetadataGraphRunner:
    """Capture and replay common attention metadata on accelerator graphs."""

    def __init__(self) -> None:
        self.graphs: dict[tuple[int, int], Any] = {}
        self.graph_pool = current_platform.get_global_graph_pool()
        self._missing_graph_keys: set[tuple[int, int]] = set()
        self._graph_capture_supported = supports_accelerator_graph()
        self._warned_graph_unavailable = False

    def clear(self) -> None:
        self.graphs.clear()
        self._missing_graph_keys.clear()

    def run(
        self,
        block_table: Any,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        *,
        use_graph: bool,
        capture: bool,
        compute: Callable[
            [
                Any,
                int,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
            None,
        ] = compute_common_attention_metadata,
    ) -> bool:
        if use_graph and not self._graph_capture_supported:
            if not self._warned_graph_unavailable:
                logger.warning(
                    "Accelerator graph capture is unavailable on %s; falling "
                    "back to eager common attention metadata generation.",
                    current_platform.device_type,
                )
                self._warned_graph_unavailable = True
            use_graph = False

        if not use_graph:
            compute(
                block_table,
                num_reqs,
                query_start_loc,
                positions,
                seq_lens,
                num_computed_tokens,
            )
            return False

        key = (id(block_table), num_reqs)
        graph = self.graphs.get(key)
        if capture:
            if graph is not None:
                graph.replay()
                return True

            graph = Graph.graph()
            with current_platform.torch_device_fn.graph(graph, pool=self.graph_pool):
                compute(
                    block_table,
                    num_reqs,
                    query_start_loc,
                    positions,
                    seq_lens,
                    num_computed_tokens,
                )
            self.graphs[key] = graph
            return True

        if graph is None:
            if key not in self._missing_graph_keys:
                logger.warning(
                    "Common attention metadata graph for %d requests was not "
                    "captured; falling back to eager execution.",
                    num_reqs,
                )
                self._missing_graph_keys.add(key)
            compute(
                block_table,
                num_reqs,
                query_start_loc,
                positions,
                seq_lens,
                num_computed_tokens,
            )
            return False

        graph.replay()
        return True
