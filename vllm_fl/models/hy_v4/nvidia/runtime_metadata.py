# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-plugin-FL PR442 and pinned vLLM 0.28 block_table.py.
import weakref
from numbers import Integral

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID
from vllm.v1.worker.block_table import SlotMappingMode

from .runtime_request_ids import populate_request_metadata

logger = init_logger(__name__)


def supports_accelerator_graph() -> bool:
    return callable(getattr(torch.cuda, "CUDAGraph", None)) and callable(
        getattr(torch.cuda, "graph", None)
    )


@triton.jit(do_not_specialize=["max_num_tokens"])
def _compute_slot_mapping_graph_kernel(
    max_num_tokens,
    query_start_loc_ptr,  # [num_reqs + 1], int32
    positions_ptr,  # [num_tokens], int64
    block_table_ptr,  # [max_num_reqs, max_num_blocks_per_req], int32 (flat)
    block_table_stride,  # max_num_blocks_per_req
    block_size,
    slot_mapping_ptr,  # [max_num_tokens], int64
    KV_CACHE_BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_KV_BLOCK: tl.constexpr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    MAP_SLOTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx == tl.num_programs(0) - 1:
        num_tokens = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
        # Device boundary changes on every replay, including shrinking batches.
        for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    if start_idx == end_idx:
        for i in range(0, block_table_stride, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                block_table_ptr + req_idx * block_table_stride + offsets,
                NULL_BLOCK_ID,
                mask=offsets < block_table_stride,
            )
    if not MAP_SLOTS:
        return

    virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        virtual_block_indices = pos // virtual_block_size
        virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
        is_local = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        local_block_offsets = (
            virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )

        block_indices = (
            virtual_block_indices * BLOCKS_PER_KV_BLOCK
            + local_block_offsets // block_size
        )
        block_numbers = tl.load(
            block_table_ptr + row_offset + block_indices,
            mask=mask & is_local,
            other=0,
        ).to(tl.int64)
        slot_offsets = local_block_offsets % block_size
        slot_ids = block_numbers * block_size + slot_offsets
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
    block_table, num_reqs, query_start_loc, positions, seq_lens, num_computed_tokens
):
    # Group iteration occurs during eager fallback/warm/capture only.
    for group in block_table.block_tables:
        mode = group.slot_mapping_mode
        if mode not in (SlotMappingMode.TOKEN_TO_KV_SLOT, SlotMappingMode.NONE):
            raise ValueError("unsupported slot mapping mode")
        _compute_slot_mapping_graph_kernel[(num_reqs + 1,)](
            group.max_num_batched_tokens,
            query_start_loc,
            positions,
            group.block_table.gpu,
            group.block_table.gpu.stride(0),
            group.block_size,
            group.slot_mapping.gpu,
            KV_CACHE_BLOCK_SIZE=group.kv_cache_block_size,
            BLOCKS_PER_KV_BLOCK=group.blocks_per_kv_block,
            TOTAL_CP_WORLD_SIZE=group.dcp_world_size,
            TOTAL_CP_RANK=group.dcp_rank,
            CP_KV_CACHE_INTERLEAVE_SIZE=group.cp_kv_cache_interleave_size,
            PAD_ID=PAD_SLOT_ID,
            NULL_BLOCK_ID=NULL_BLOCK_ID,
            MAP_SLOTS=mode == SlotMappingMode.TOKEN_TO_KV_SLOT,
            BLOCK_SIZE=1024,
        )
    _compute_num_computed_tokens_kernel[(triton.cdiv(num_reqs, 256),)](
        query_start_loc,
        seq_lens,
        num_computed_tokens,
        num_reqs=num_reqs,
        BLOCK_SIZE=256,
    )


def _tensor_key(t):
    return (
        t.data_ptr(),
        tuple(t.shape),
        tuple(t.stride()),
        str(t.dtype),
        str(t.device),
    )


def _bindings(block_table, num_reqs, tensors):
    owners = list(tensors)
    groups = []
    for g in block_table.block_tables:
        owners.extend((g.block_table.gpu, g.slot_mapping.gpu))
        groups.append(
            (
                _tensor_key(g.block_table.gpu),
                _tensor_key(g.slot_mapping.gpu),
                g.block_size,
                g.kv_cache_block_size,
                g.blocks_per_kv_block,
                g.max_num_batched_tokens,
                g.dcp_world_size,
                g.dcp_rank,
                g.cp_kv_cache_interleave_size,
                g.slot_mapping_mode.value,
            )
        )
    return (
        (id(block_table), num_reqs, tuple(map(_tensor_key, tensors)), tuple(groups)),
        (block_table, *owners),
    )


class CommonAttentionMetadataGraphRunner:
    def __init__(self):
        self.graphs = {}
        self.graph_pool = current_platform.get_global_graph_pool()
        self._graph_capture_supported = supports_accelerator_graph()
        self._missing_graph_keys = set()
        self._request_id_map = None
        self._request_id_counts = None
        self._request_id_generation = 0
        self._request_id_current = None

    def clear(self):
        # Caller synchronizes before releasing profile-time KV/input owners.
        self.graphs.clear()
        self._missing_graph_keys.clear()
        self._request_id_current = None
        self._request_id_map = None
        self._request_id_counts = None
        self._request_id_generation += 1

    def run(
        self,
        block_table,
        num_reqs,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
        *,
        use_graph,
        capture,
        compute=compute_common_attention_metadata,
        extra_tensors=(),
    ):
        # Expire every prior metadata lease before attempting a new submission.
        self._request_id_current = None
        self._request_id_generation += 1
        if compute is not compute_step_common_attention_metadata:
            return self._run_bound(
                block_table,
                num_reqs,
                query_start_loc,
                positions,
                seq_lens,
                num_computed_tokens,
                use_graph=use_graph,
                capture=capture,
                compute=compute,
                extra_tensors=extra_tensors,
            )
        if not _request_id_inputs_supported(num_reqs, query_start_loc, positions):
            return self._run_bound(
                block_table,
                num_reqs,
                query_start_loc,
                positions,
                seq_lens,
                num_computed_tokens,
                use_graph=use_graph,
                capture=capture,
                compute=compute,
                extra_tensors=extra_tensors,
            )
        output = self._request_id_map
        capacity = positions.numel()
        if (
            output is None
            or output.numel() != capacity
            or output.device != positions.device
        ):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("request-ID map must be allocated before capture")
            output = torch.empty(
                (capacity,), dtype=torch.int32, device=positions.device
            )
            self._request_id_map = output
        counts = self._request_id_counts
        if (
            counts is None
            or counts.numel() != num_reqs
            or counts.device != positions.device
        ):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("request counts must be allocated before capture")
            counts = torch.empty(
                (num_reqs,), dtype=torch.int32, device=positions.device
            )
            self._request_id_counts = counts
        result = self._run_bound(
            block_table,
            num_reqs,
            query_start_loc,
            positions,
            seq_lens,
            num_computed_tokens,
            use_graph=use_graph,
            capture=capture,
            compute=_compute_step_common_with_request_ids,
            extra_tensors=(output, counts, *extra_tensors),
        )
        # Publish only after the exact eager/captured/replay producer succeeded.
        self._request_id_current = (query_start_loc, output, num_reqs, counts)
        return result

    def attach_req_id_per_token(self, common):
        """One native cm_base call, before shallow group copies or slicing.

        The full native QSL suffix must already be current, as required by the
        existing common producer. This method is never called on a sub-batch.
        """
        common._model_request_id_lease = None
        current = self._request_id_current
        if current is None:
            return False
        source, output, extent, counts = current
        qsl = common.query_start_loc
        if (
            type(qsl) is not torch.Tensor
            or qsl.dtype != torch.int32
            or qsl.device != source.device
            or qsl.dim() != 1
            or qsl.stride() != (1,)
            or qsl.data_ptr() != source.data_ptr()
            or qsl.untyped_storage() is not source.untyped_storage()
            or type(common.num_reqs) is not int
            or not 0 <= common.num_reqs <= extent
            or qsl.numel() != common.num_reqs + 1
            or type(common.num_actual_tokens) is not int
            or not 0 <= common.num_actual_tokens <= output.numel()
            or common.query_start_loc_cpu.numel() != qsl.numel()
        ):
            return False
        common._model_request_id_lease = _RequestIdMapLease(
            self, self._request_id_generation, common, output, counts
        )
        return True

    def _run_bound(
        self,
        block_table,
        num_reqs,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
        *,
        use_graph,
        capture,
        compute=compute_common_attention_metadata,
        extra_tensors=(),
    ):
        args = (
            block_table,
            num_reqs,
            query_start_loc,
            positions,
            seq_lens,
            num_computed_tokens,
            *extra_tensors,
        )
        if not use_graph or not self._graph_capture_supported:
            compute(*args)
            return False
        key, owners = _bindings(block_table, num_reqs, args[2:])
        key = (id(compute), *key)
        owners = (compute, *owners)
        entry = self.graphs.get(key)
        if entry is not None:
            entry[0].replay()
            return True
        if not capture:
            if key not in self._missing_graph_keys:
                logger.warning(
                    "Common metadata binding was not captured; using eager producer"
                )
                self._missing_graph_keys.add(key)
            compute(*args)
            return False
        # Native helper only writes metadata; no KV stores. Compile eagerly,
        # capture, replay once to materialize outputs, then retain all owners.
        compute(*args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool):
            compute(*args)
        graph.replay()
        self.graphs[key] = (graph, owners)
        return True


@triton.jit
def _prepare_positions_seq_lens_kernel(
    query_start_loc, computed, positions, seq_lens, BLOCK_SIZE: tl.constexpr
):
    req = tl.program_id(0)
    start = tl.load(query_start_loc + req).to(tl.int64)
    end = tl.load(query_start_loc + req + 1).to(tl.int64)
    if start == end:
        tl.store(seq_lens + req, 0)
        return
    base = tl.load(computed + req).to(tl.int64)
    tl.store(seq_lens + req, base + end - start)
    for i in range(start, end, BLOCK_SIZE):
        token = i + tl.arange(0, BLOCK_SIZE)
        tl.store(positions + token, base + token - start, mask=token < end)


def compute_step_common_attention_metadata(
    block_table,
    num_reqs,
    query_start_loc,
    positions,
    seq_lens,
    num_computed_tokens,
    mrope_state=None,
    mrope_output=None,
):
    _prepare_positions_seq_lens_kernel[(num_reqs,)](
        query_start_loc, num_computed_tokens, positions, seq_lens, BLOCK_SIZE=256
    )
    compute_common_attention_metadata(
        block_table, num_reqs, query_start_loc, positions, seq_lens, num_computed_tokens
    )

    if mrope_state is not None:
        decode_mrope_kernel[(num_reqs,)](
            query_start_loc,
            num_computed_tokens,
            mrope_state,
            mrope_output,
            ROW_STRIDE=mrope_output.stride(0),
            COL_STRIDE=mrope_output.stride(1),
            BLOCK_SIZE=256,
        )


def select_decode_deltas(
    req_ids,
    requests,
    scheduled_tokens,
    computed_cpu,
    prompt_length,
    *,
    uses_mrope,
    uses_xdrope_dim=0,
    multimodal_pruning=False,
    scheduled_encoder_inputs=None,
):
    """Return current ordered deltas only when every token is completion.

    prompt_length must be native length_from_prompt_token_ids_or_embeds.
    This makes no tensor/metadata cache and performs no GPU work.
    """
    if (
        not uses_mrope
        or uses_xdrope_dim
        or multimodal_pruning
        or scheduled_encoder_inputs
        or not req_ids
    ):
        return None
    deltas = []
    for i, req_id in enumerate(req_ids):
        req = requests[req_id]
        count = computed_cpu[i]
        scheduled = scheduled_tokens[req_id]
        delta = req.mrope_position_delta
        if (
            req.mrope_positions is None
            or not isinstance(delta, Integral)
            or isinstance(delta, bool)
            or not -(1 << 63) <= int(delta) < (1 << 63)
            or not isinstance(count, Integral)
            or not isinstance(scheduled, Integral)
            or scheduled <= 0
            or count < prompt_length(req.prompt_token_ids, req.prompt_embeds)
        ):
            return None
        deltas.append(int(delta))
    return tuple(deltas)


@triton.jit
def decode_mrope_kernel(
    query_start_loc,
    computed_tokens,
    state,
    output,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.load(state) == 0:
        return
    req = tl.program_id(0)
    start = tl.load(query_start_loc + req).to(tl.int64)
    end = tl.load(query_start_loc + req + 1).to(tl.int64)
    # Empty padded rows must not read stale request delta/count values.
    if start == end:
        return
    base = tl.load(computed_tokens + req).to(tl.int64) + tl.load(state + 1 + req).to(
        tl.int64
    )
    for i in range(start, end, BLOCK_SIZE):
        token = i + tl.arange(0, BLOCK_SIZE)
        value = base + token - start
        for axis in range(3):
            tl.store(
                output + axis * ROW_STRIDE + token * COL_STRIDE, value, mask=token < end
            )


class _RequestIdMapLease:
    """Strong tensor owners, weak runner, and one successful submission token."""

    __slots__ = (
        "runner",
        "generation",
        "qsl",
        "qsl_cpu",
        "num_reqs",
        "num_tokens",
        "output",
        "counts",
    )

    def __init__(self, runner, generation, common, output, counts):
        self.runner = weakref.ref(runner)
        self.generation = generation
        self.qsl = common.query_start_loc
        self.qsl_cpu = common.query_start_loc_cpu
        self.num_reqs = common.num_reqs
        self.num_tokens = common.num_actual_tokens
        self.output = output
        self.counts = counts


def _request_id_inputs_supported(num_reqs, query_start_loc, positions):
    return (
        type(num_reqs) is int
        and num_reqs > 0
        and type(query_start_loc) is torch.Tensor
        and type(positions) is torch.Tensor
        and query_start_loc.device.type == "cuda"
        and query_start_loc.device == positions.device
        and query_start_loc.dtype == torch.int32
        and query_start_loc.dim() == positions.dim() == 1
        and query_start_loc.numel() == num_reqs + 1
        and query_start_loc.stride() == positions.stride() == (1,)
        and not query_start_loc.is_neg()
        and not query_start_loc.is_conj()
        and positions.numel() > 0
    )


def _request_id_lease_from_common(common):
    lease = getattr(common, "_model_request_id_lease", None)
    if type(lease) is not _RequestIdMapLease:
        return None
    runner = lease.runner()
    if (
        runner is None
        or runner._request_id_current is None
        or runner._request_id_generation != lease.generation
        or runner._request_id_current[1] is not lease.output
        or runner._request_id_current[3] is not lease.counts
        or common.query_start_loc is not lease.qsl
        or common.query_start_loc_cpu is not lease.qsl_cpu
        or type(common.num_reqs) is not int
        or common.num_reqs != lease.num_reqs
        or type(common.num_actual_tokens) is not int
        or common.num_actual_tokens != lease.num_tokens
    ):
        return None
    return lease


def request_id_per_token_from_common(common):
    lease = _request_id_lease_from_common(common)
    return None if lease is None else lease.output[: lease.num_tokens]


def decode_lens_from_common(common, num_decodes):
    """Current full-batch prefix only; CP/spec/expand eligibility is the caller's."""
    lease = _request_id_lease_from_common(common)
    if (
        lease is None
        or type(num_decodes) is not int
        or not 0 <= num_decodes <= lease.num_reqs
    ):
        return None
    return lease.counts[:num_decodes]


def _compute_step_common_with_request_ids(
    block_table,
    num_reqs,
    query_start_loc,
    positions,
    seq_lens,
    num_computed_tokens,
    request_ids,
    request_counts,
    *extra_tensors,
):
    compute_step_common_attention_metadata(
        block_table,
        num_reqs,
        query_start_loc,
        positions,
        seq_lens,
        num_computed_tokens,
        *extra_tensors,
    )
    populate_request_metadata(query_start_loc, request_ids, request_counts)
