# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host request facts and metadata leases shared by Hy4 platform adapters.

Proof construction and validation use current CPU packet facts. This module
owns the single proof type and context and does not import a device backend.
"""

from __future__ import annotations

import weakref
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class CPUSequenceLengthsProof:
    """Immutable packet facts in the current reordered request order."""

    issuer: str
    generation: int
    request_ids: tuple[str, ...]
    query_starts: tuple[int, ...]
    computed: tuple[int, ...]
    prompt_lengths: tuple[int, ...]
    previous_positions: tuple[int, ...]
    previous_drafts: tuple[int, ...]


@dataclass(frozen=True)
class PrefillChunkSequenceBounds:
    """Per-request KV extents for a current paged prefill query slice."""

    step_proof: CPUSequenceLengthsProof
    request_start: int
    request_stop: int
    chunk_start: int
    chunk_end: int
    seq_lens: tuple[int, ...]


_current_cpu_seq_lens_proof: ContextVar[CPUSequenceLengthsProof | None] = ContextVar(
    "current_cpu_seq_lens_proof", default=None
)


def set_current_cpu_seq_lens_proof(proof: CPUSequenceLengthsProof | None) -> None:
    _current_cpu_seq_lens_proof.set(proof)


def get_current_cpu_seq_lens_proof() -> CPUSequenceLengthsProof | None:
    return _current_cpu_seq_lens_proof.get()


def validated_cpu_seq_lens_proof(
    stager,
    request_ids,
    previous_req_map,
    query_starts,
    upper,
    prefilling,
    prompt_lengths,
    num_reqs,
    num_reqs_padded,
    num_tokens,
    num_tokens_padded,
    for_capture,
    ubatch_slices,
):
    """Accept only immutable host facts from the current successful packet."""
    if (
        for_capture
        or ubatch_slices is not None
        or stager is None
        or not stager.in_step
        or stager.plan is None
        or num_reqs_padded != num_reqs
        or num_tokens_padded != num_tokens
        or any(
            x.device.type != "cpu"
            for x in (query_starts, upper, prefilling, prompt_lengths)
        )
        or previous_req_map is None
    ):
        return None
    plan = stager.plan
    proof = plan.cpu_seq_lens_proof
    if (
        proof is None
        or proof.issuer != stager._cpu_proof_issuer
        or proof.generation != stager._cpu_proof_generation
        or plan.num_reqs != num_reqs
        or plan.total_tokens != num_tokens
        or tuple(request_ids) != plan.req_ids
        or tuple(request_ids) != proof.request_ids
        or tuple(int(previous_req_map.get(r, -1)) for r in request_ids)
        != proof.previous_positions
        or len(proof.request_ids) != num_reqs
        or len(proof.query_starts) != num_reqs + 1
        or any(
            len(x) != num_reqs
            for x in (
                proof.computed,
                proof.prompt_lengths,
                proof.previous_positions,
                proof.previous_drafts,
            )
        )
    ):
        return None
    starts = tuple(int(x) for x in query_starts[: num_reqs + 1])
    if starts != proof.query_starts or starts[0] != 0 or starts[-1] != num_tokens:
        return None
    for i in range(num_reqs):
        q = starts[i + 1] - starts[i]
        computed = proof.computed[i]
        prompt = proof.prompt_lengths[i]
        if (
            q < 0
            or computed < 0
            or prompt < 0
            or int(upper[i]) != computed + q
            or int(prompt_lengths[i]) != prompt
            or bool(prefilling[i]) != (computed < prompt)
        ):
            return None
    return proof


def validated_prefill_chunk_bound(
    common,
    chunk,
    req_start,
    req_stop,
    num_decodes,
    decode_threshold,
    compress_ratio,
    dcp_world,
    pcp_world,
):
    """Bind a paged prefill chunk to immutable facts from this input step.

    Per-request lengths bound score columns; their sum bounds flattened
    gather offsets. They intentionally have different capacity contracts.
    """
    proof = common.cpu_seq_lens_proof
    if (
        proof is None
        or compress_ratio != 1
        or dcp_world != 1
        or pcp_world != 1
        or not num_decodes <= req_start < req_stop <= common.num_reqs
        or chunk.num_reqs != req_stop - req_start
        or common.num_reqs != len(proof.request_ids)
        or any(
            x.device.type != "cpu"
            for x in (
                common.query_start_loc_cpu,
                common.seq_lens_cpu_upper_bound,
                common.is_prefilling,
            )
        )
        or tuple(int(x) for x in common.query_start_loc_cpu) != proof.query_starts
    ):
        return None
    starts = proof.query_starts[req_start : req_stop + 1]
    if not (starts[0] <= chunk.token_start < chunk.token_end <= starts[-1]):
        return None
    bounds = []
    for i in range(req_start, req_stop):
        prev, draft = proof.previous_positions[i], proof.previous_drafts[i]
        query_len = proof.query_starts[i + 1] - proof.query_starts[i]
        computed = proof.computed[i]
        bound = computed + query_len
        if (
            query_len <= decode_threshold
            or computed < 0
            or not computed < proof.prompt_lengths[i]
            or not bool(common.is_prefilling[i])
            or not ((prev == -1 and draft == -1) or (prev >= 0 and draft == 0))
            or int(common.seq_lens_cpu_upper_bound[i]) != bound
        ):
            return None
        bounds.append(bound)
    total = sum(bounds)
    if (
        chunk.total_seq_lens != total
        or chunk.local_total_seq_lens != total
        or chunk.num_reqs != chunk.block_table.shape[0]
    ):
        return None
    return PrefillChunkSequenceBounds(
        proof, req_start, req_stop, chunk.token_start, chunk.token_end, tuple(bounds)
    )


def push_current_cpu_seq_lens_proof(proof):
    """Start a nested worker/forward proof lifetime."""
    return _current_cpu_seq_lens_proof.set(proof)


def reset_current_cpu_seq_lens_proof(token):
    """Restore the caller's proof context, including exceptional exits."""
    _current_cpu_seq_lens_proof.reset(token)


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
    ids = tuple(request_ids)
    ends = tuple(int(x) for x in query_ends)
    prompts = tuple(int(x) for x in prompt_lengths)
    if (
        len(ids) != num_reqs
        or len(set(ids)) != num_reqs
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
        (-1 if p == -1 else int(packet[2 * n + 2 * b + p]) if 0 <= p < capacity else -2)
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
    return CPUSequenceLengthsProof(
        issuer, generation, ids, starts, computed, prompts, prev, drafts
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
            or delta is None
            or not -(1 << 63) <= int(delta) < (1 << 63)
            or scheduled <= 0
            or count < prompt_length(req.prompt_token_ids, req.prompt_embeds)
        ):
            return None
        deltas.append(int(delta))
    return tuple(deltas)


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


def _request_id_lease_from_common(common):
    lease = getattr(common, "_model_request_id_lease", None)
    if lease is None:
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
        or common.num_reqs != lease.num_reqs
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
    if lease is None or not 0 <= num_decodes <= lease.num_reqs:
        return None
    return lease.counts[:num_decodes]
