# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA packet staging and graph-bound metadata owners for Hy4.

GPU producers are imported when submitting work. CPU request facts and lease
readers belong to the platform-independent model metadata module.
"""

from __future__ import annotations

import logging
import uuid
import weakref
from types import SimpleNamespace

import torch

from ..metadata import (
    _cpu_sequence_proof,
    _RequestIdMapLease,
    build_input_maps,
    pack_host_packet,
)

logger = logging.getLogger(__name__)


class AsyncInputStaging:
    def __init__(self, runner):
        self.runner = weakref.ref(runner)
        self.capacity = int(runner.max_num_reqs)
        self.map_capacity = int(runner.max_num_tokens)
        size = 2 * self.map_capacity + 3 * self.capacity
        self.cpu = torch.empty(size, dtype=torch.int64, device="cpu", pin_memory=True)
        self.host = self.cpu.numpy()
        self.gpu = torch.empty(size, dtype=torch.int64, device=runner.device)
        self.plan = None
        self.in_step = False
        self._cpu_proof_generation = 0
        self._cpu_proof_issuer = uuid.uuid4().hex

    def begin_step(self):
        # Caller has just synchronized native prepare_inputs_event. Previous
        # H2D and GPU packet consumers are complete; only now overwrite/release.
        if self.in_step:
            raise RuntimeError("overlapping native input preparation")
        self.plan = None
        self._cpu_proof_generation += 1
        self.in_step = True

    def end_step(self):
        # Keep packet and dynamic source owners until NEXT begin after wait.
        self.in_step = False

    def prepare_and_correct(self, scheduler_output, num_reqs, total_tokens, cu_tokens):
        r = self.runner()
        if not self.in_step:
            raise RuntimeError("packet preparation outside native input-prep lease")
        if (
            r._pp_recv_work is not None
            or r.common_attention_metadata_graph is None
            or r.input_batch.prev_sampled_token_ids is None
            or not 1 <= num_reqs <= self.capacity <= 1024
        ):
            return False
        sample = r.input_batch.prev_sampled_token_ids
        draft = r._draft_token_ids
        if (
            not sample.is_cuda
            or sample.dtype != torch.int32
            or sample.ndim != 2
            or sample.shape[1] < 1
            or min(sample.stride()) < 0
        ):
            return False
        m = build_input_maps(r, scheduler_output, num_reqs, total_tokens, cu_tokens)
        if m is None:
            return False
        if m.draft_dst and (
            draft is None
            or draft.ndim != 2
            or draft.dtype not in (torch.int32, torch.int64)
            or not draft.is_cuda
            or draft.shape[1] < 1
            or min(draft.stride()) < 0
        ):
            return False
        if not m.draft_dst:
            draft = sample  # Never dereferenced by a draft lane.
        out = r.input_ids.gpu
        values = (
            out,
            r.num_computed_tokens,
            r.num_accepted_tokens.gpu,
            r.valid_sampled_token_count_gpu,
            r.prev_positions.gpu,
            r.prev_num_draft_tokens.gpu,
        )
        dtypes = (
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int64,
            torch.int32,
        )
        if any(
            not x.is_cuda
            or x.device != out.device
            or x.dtype != dtype
            or x.ndim != 1
            or x.stride(0) != 1
            for x, dtype in zip(values, dtypes)
        ):
            return False
        if sample.device != out.device or draft.device != out.device:
            return False
        if any(
            x.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
            for x in (sample, draft)
        ):
            return False
        dst, src = m.sample_dst + m.draft_dst, m.sample_src + m.draft_src
        n = len(dst)
        if (
            n > min(self.map_capacity, 2**31 - 1)
            or len(set(dst)) != n
            or any(x < 0 or x >= total_tokens for x in dst)
            or any(x >= sample.shape[0] for x in m.sample_src)
            or any(x < 0 or x >= draft.numel() for x in m.draft_src)
        ):
            return False
        if (
            out.numel() < total_tokens
            or sample.numel() > 2**31 - 1
            or draft.numel() > 2**31 - 1
        ):
            return False
        if len({x.untyped_storage().data_ptr() for x in values}) != len(values):
            return False
        prev = r.prev_positions.np[:num_reqs]
        valid = r.valid_sampled_token_count_gpu
        if (
            any(x < -1 or x >= valid.numel() for x in prev)
            or r.num_computed_tokens.numel() < self.capacity
            or r.prev_num_draft_tokens.gpu.numel() != self.capacity
            or r.prev_positions.gpu.numel() < num_reqs
            or r.num_accepted_tokens.gpu.numel() < num_reqs
        ):
            return False
        used = pack_host_packet(
            self.host,
            dst,
            src,
            prev,
            r.input_batch.num_computed_tokens_cpu_tensor[:num_reqs].numpy(),
            r.prev_num_draft_tokens.np,
        )
        self.gpu[:used].copy_(self.cpu[:used], non_blocking=True)
        stream = torch.cuda.current_stream(r.device).cuda_stream
        self.plan = SimpleNamespace(
            num_maps=n,
            num_reqs=num_reqs,
            total_tokens=total_tokens,
            req_ids=tuple(r.input_batch.req_ids),
            copy_base=m.copy_base,
            num_sampled=m.num_sampled,
            sampled=sample,
            draft=draft,
            valid_counts=valid,
            stream=stream,
            original_draft=r._draft_token_ids if m.draft_dst else None,
        )
        from ..metadata_kernels import async_input_correct

        async_input_correct(
            self.gpu,
            r.num_computed_tokens,
            r.num_accepted_tokens.gpu,
            valid,
            r.prev_positions.gpu,
            r.prev_num_draft_tokens.gpu,
            num_maps=n,
            num_reqs=num_reqs,
            prev_capacity=self.capacity,
        )
        self.plan.cpu_seq_lens_proof = _cpu_sequence_proof(
            self._cpu_proof_issuer,
            self._cpu_proof_generation,
            r.input_batch.req_ids,
            self.host,
            n,
            num_reqs,
            self.capacity,
            cu_tokens,
            r.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs],
            r.input_batch.prev_req_id_to_index,
        )
        return True

    def apply_if_prepared(self, scheduler_output, num_reqs, total_tokens):
        if self.plan is None:
            return False
        r, p = self.runner(), self.plan
        if (
            not self.in_step
            or p.num_reqs != num_reqs
            or p.total_tokens != total_tokens
            or p.req_ids != tuple(r.input_batch.req_ids)
            or p.sampled is not r.input_batch.prev_sampled_token_ids
            or p.valid_counts is not r.valid_sampled_token_count_gpu
            or (
                p.original_draft is not None
                and p.original_draft is not r._draft_token_ids
            )
            or torch.cuda.current_stream(r.device).cuda_stream != p.stream
        ):
            raise RuntimeError("native packet input-prep lease/stream changed")
        # Preserve original uploads for prompt/mixed new requests. CPU token
        # construction, bookkeeping, PP wait and all noneligible branches remain native.
        if r.enable_prompt_embeds:
            r.is_token_ids.copy_to_gpu(total_tokens)
        if p.copy_base:
            r.input_ids.copy_to_gpu(total_tokens)
            if r.enable_prompt_embeds:
                r.inputs_embeds.copy_to_gpu(total_tokens)
        from ..metadata_kernels import async_input_gather

        async_input_gather(
            self.gpu,
            r.input_ids.gpu,
            p.sampled,
            p.draft,
            num_maps=p.num_maps,
            num_sampled=p.num_sampled,
        )
        return True


def make_async_input_staging(runner):
    if not runner.use_async_spec_decode or runner.prepare_inputs_event is None:
        return None
    return AsyncInputStaging(runner)


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
        from vllm.platforms import current_platform

        self.graphs = {}
        self.graph_pool = current_platform.get_global_graph_pool()
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
        extra_tensors=(),
    ):
        # Expire every prior metadata lease before attempting a new submission.
        self._request_id_current = None
        self._request_id_generation += 1
        if not _request_id_inputs_supported(num_reqs, query_start_loc, positions):
            raise ValueError("unsupported native request-ID metadata layout")
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
            qsl.dtype != torch.int32
            or qsl.device != source.device
            or qsl.dim() != 1
            or qsl.stride() != (1,)
            or qsl.data_ptr() != source.data_ptr()
            or qsl.untyped_storage() is not source.untyped_storage()
            or not 0 <= common.num_reqs <= extent
            or qsl.numel() != common.num_reqs + 1
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
        extra_tensors=(),
    ):
        from ..metadata_kernels import _compute_step_common_with_request_ids

        compute = _compute_step_common_with_request_ids
        args = (
            block_table,
            num_reqs,
            query_start_loc,
            positions,
            seq_lens,
            num_computed_tokens,
            *extra_tensors,
        )
        if not use_graph:
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


def _request_id_inputs_supported(num_reqs, query_start_loc, positions):
    return (
        num_reqs > 0
        and query_start_loc.device.type == "cuda"
        and query_start_loc.device == positions.device
        and query_start_loc.dtype == torch.int32
        and query_start_loc.dim() == positions.dim() == 1
        and query_start_loc.numel() == num_reqs + 1
        and query_start_loc.stride() == positions.stride() == (1,)
        and positions.numel() > 0
    )
