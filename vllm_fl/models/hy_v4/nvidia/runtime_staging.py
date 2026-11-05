# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packet orchestration inside the native input-preparation event lease."""

from __future__ import annotations

import uuid
import weakref
from types import SimpleNamespace

import torch

from .runtime_input_kernels import async_input_correct, async_input_gather


def build_input_maps(runner, scheduler_output, num_reqs, total_tokens, cu_tokens):
    """Exact native generic async recipe; normal/unchanged routes stay native."""
    sample_dst, sample_src, draft_dst, draft_src = [], [], [], []
    total_spec, same, maximum = 0, True, -1
    for current in range(num_reqs):
        previous = int(runner.prev_positions.np[current])
        if previous < 0:
            continue
        req_id = runner.input_batch.req_ids[current]
        draft_len = len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
        total_spec += draft_len
        last = int(cu_tokens[current]) - 1
        sample_dst.append(last - draft_len)
        sample_src.append(previous)
        draft_dst.extend(range(last - draft_len + 1, last + 1))
        start = previous * runner.prev_num_spec_tokens
        draft_src.extend(range(start, start + draft_len))
        same &= previous == last
        maximum = max(maximum, last)
    count = len(sample_dst)
    if not count or (same and maximum == count - 1):
        return None
    return SimpleNamespace(
        sample_dst=sample_dst,
        sample_src=sample_src,
        draft_dst=draft_dst,
        draft_src=draft_src,
        num_sampled=count,
        copy_base=count < total_tokens - total_spec,
    )


def pack_host_packet(host, dst, src, prev, cpu_values, previous_drafts):
    """CPU metadata only; its allocation and event lease belong to the runner."""
    n, b, capacity = len(dst), len(prev), len(previous_drafts)
    used = 2 * n + 2 * b + capacity
    host[:n], host[n : 2 * n] = dst, src
    host[2 * n : 2 * n + b] = prev
    host[2 * n + b : 2 * n + 2 * b] = cpu_values
    host[2 * n + 2 * b : used] = previous_drafts
    return used


def _cpu_sequence_proof(
    proof_type,
    issuer,
    generation,
    request_ids,
    packet,
    num_maps,
    num_reqs,
    capacity,
    query_ends,
    prompt_lengths,
    previous_req_map,
):
    """Copy only the packet values actually sent for this input-prep lease."""
    try:
        ids = tuple(request_ids)
        ends = tuple(int(x) for x in query_ends)
        prompts = tuple(int(x) for x in prompt_lengths)
        if (
            proof_type is None
            or not isinstance(issuer, str)
            or not issuer
            or type(generation) is not int
            or generation <= 0
            or len(ids) != num_reqs
            or len(set(ids)) != num_reqs
            or not all(isinstance(x, str) for x in ids)
            or len(ends) != num_reqs
            or len(prompts) != num_reqs
        ):
            return None
        n, b = num_maps, num_reqs
        prev = tuple(int(x) for x in packet[2 * n : 2 * n + b])
        computed = tuple(int(x) for x in packet[2 * n + b : 2 * n + 2 * b])
        if (
            previous_req_map is None
            or tuple(int(previous_req_map.get(r, -1)) for r in ids) != prev
        ):
            return None
        drafts = tuple(
            (
                -1
                if p == -1
                else int(packet[2 * n + 2 * b + p])
                if 0 <= p < capacity
                else -2
            )
            for p in prev
        )
        starts = (0,) + ends
        if (
            any(p < -1 or p >= capacity for p in prev)
            or any(d < -1 for d in drafts)
            or any(a > b for a, b in zip(starts, starts[1:]))
            or any(v < 0 for v in computed)
            or any(v < 0 for v in prompts)
        ):
            return None
        return proof_type(
            issuer, generation, ids, starts, computed, prompts, prev, drafts
        )
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


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
            not torch.is_tensor(sample)
            or not sample.is_cuda
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
            not torch.is_tensor(draft)
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
            not torch.is_tensor(x)
            or not x.is_cuda
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
            n > self.map_capacity
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
        from vllm.v1.attention import backend as attention_backend

        proof_type = getattr(attention_backend, "CPUSequenceLengthsProof", None)
        self.plan.cpu_seq_lens_proof = _cpu_sequence_proof(
            proof_type,
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
