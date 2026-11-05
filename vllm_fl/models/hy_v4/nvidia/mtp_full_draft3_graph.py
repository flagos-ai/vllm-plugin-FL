# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 three-step draft graph with native target and rejection semantics."""

from __future__ import annotations

from types import SimpleNamespace

from .mtp_draft_graph import _Workspace, decode_plan, refresh_metadata
from .mtp_tail_graph import clone_metadata


def full_draft_plan(common, q4_sizes, q1_sizes):
    """Uniform verified q4 only; capacities never grow implicitly."""
    plan = decode_plan(common, q4_sizes, verified_step=True)
    if plan is None or plan[0] != 4:
        return None
    _, first_rows, first_reqs = plan
    rows = next((s for s in q1_sizes if s >= common.num_reqs), None)
    if rows is None or first_reqs != rows:
        return None
    return first_rows, rows


class HYV4FullDraft3Graph:
    """First q4 + native greedy + the existing direct two-q1 body chain.

    It shares the tail's persistent per-B q1 state, never replays a child graph
    inside capture, and owns separate first-q4 metadata per key. Only the
    first-q4 input hidden buffer is shared across its serialized graph calls.
    """

    def __init__(self, base):
        self.base, self.tail = base, base.tail
        self.proposer, self.torch = base.proposer, base.torch
        self.generation = base.generation
        self.entries, self.cache_bindings = {}, {}
        # Keep the input arena for this helper's lifetime, including clear().
        self.first_hidden = None
        self.captured = False
        p = self.proposer
        self.q1_sizes = tuple(self.tail.q1_sizes)
        self.q4_sizes = tuple(s for s in base.sizes if s % 4 == 0)
        self.max_batch = (
            min(self.q1_sizes[-1], self.q4_sizes[-1] // 4) if self.q4_sizes else 0
        )
        self.policy_supported = (
            self.tail.policy_supported
            and p.method == "mtp"
            and not p.needs_extra_input_slots
            and not p.parallel_drafting
            and not p._share_mtp_indices
        )

    def clear(self):
        if hasattr(self, "wrapper"):
            self.wrapper.clear_graphs()
        self.entries.clear()
        self.cache_bindings.clear()
        self.captured = False

    def _positions_source(self):
        p = self.proposer
        return p.mrope_positions if p.uses_mrope else p.positions

    def _first_body(self, entry):
        from vllm.forward_context import set_forward_context

        p, first = self.proposer, entry.first
        with set_forward_context(
            entry.first_metadata,
            p.vllm_config,
            num_tokens=entry.first_rows,
            cudagraph_runtime_mode=self.base.mode,
            batch_descriptor=entry.first_descriptor,
            slot_mapping={
                name: first.slots[: entry.first_rows]
                for name in p._draft_attn_layer_names
            },
        ):
            return p.model(
                input_ids=first.ids[: entry.first_rows],
                positions=first.positions[..., : entry.first_rows],
                inputs_embeds=None,
                hidden_states=first.hidden[: entry.first_rows],
            )

    def _chain(self, entry):
        # These are the exact live native indices, not fixed 4*r+3 indices.
        # Native discarded requests can contain -1. The complete native
        # position buffer snapshot preserves its negative-index semantics.
        hidden0 = self._first_body(entry)
        selected = hidden0[entry.sample_indices]
        seed = self.tail._greedy(selected)
        tail, n = entry.tail, entry.batch
        state = tail.base
        state.ids.zero_()
        state.hidden.zero_()
        state.positions.zero_()
        state.seq.zero_()
        state.ids[:n].copy_(seed)
        state.hidden[:n].copy_(selected)
        state.positions[..., :n].copy_(
            entry.native_positions[..., entry.sample_indices]
        )
        state.seq[:n].copy_(entry.first.seq[:n] - entry.rejected)
        # CPU QSL fixes the descriptor, but the GPU QSL is zero for cold
        # capture. Dividing uniform q4 QSL retains this inactive-row mask.
        state.qsl.copy_(entry.first.qsl[: entry.rows + 1] // 4)
        state.table.copy_(entry.first.table[: entry.rows])
        tail.seed.copy_(seed)
        for phase in tail.steps:
            phase.table.copy_(state.table)
        # Only call the already-verified direct body implementation. A
        # captured tail.wrapper.replay is deliberately never nested here.
        return self.tail._chain(tail)

    def _context_call(self, entry, *, warm=False):
        from vllm.forward_context import set_forward_context

        with set_forward_context(
            {},
            self.proposer.vllm_config,
            num_tokens=entry.first_rows,
            cudagraph_runtime_mode=self.base.mode,
            batch_descriptor=entry.descriptor,
            slot_mapping={},
        ):
            return self._chain(entry) if warm else self.wrapper(entry)

    def capture(self):
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.forward_context import BatchDescriptor
        from vllm.v1.attention.backend import CommonAttentionMetadata

        if not self.policy_supported:
            return
        if (
            self.captured
            or not self.tail.captured
            or self.generation != self.base.generation
        ):
            raise RuntimeError("full draft3 capture generation/order mismatch")
        t, p = self.torch, self.proposer
        self.wrapper = CUDAGraphWrapper(
            self._chain, p.vllm_config, runtime_mode=self.base.mode
        )
        with t.inference_mode():
            first_hidden = self.first_hidden
            for batch in range(self.max_batch, 0, -1):
                first_rows = next(s for s in self.q4_sizes if s >= batch * 4)
                rows = next(s for s in self.q1_sizes if s >= batch)
                if first_rows // 4 != rows:
                    continue
                tail_descriptor = BatchDescriptor(
                    num_tokens=rows, num_reqs=batch, uniform=True
                )
                if tail_descriptor not in self.tail.entries:
                    raise RuntimeError(
                        "full draft3 requires the exact captured tail key"
                    )
                # Keep every descriptor's metadata private. The first-body
                # input is consumed before the next serialized stage/replay;
                # all entries retain its one fixed-address arena.
                first = _Workspace(self.base, 4, hidden=first_hidden)
                self.first_hidden = first_hidden = first.hidden
                qsl_cpu = t.arange(rows + 1, dtype=t.int32) * 4
                common = CommonAttentionMetadata(
                    query_start_loc=qsl_cpu.to(p.device),
                    query_start_loc_cpu=qsl_cpu,
                    seq_lens=t.full((rows,), 4, dtype=t.int32, device=p.device),
                    num_reqs=rows,
                    num_actual_tokens=first_rows,
                    max_query_len=4,
                    max_seq_len=p.max_model_len,
                    block_table_tensor=t.zeros_like(first.table[:rows]),
                    slot_mapping=t.full_like(first.slots[:first_rows], -1),
                    causal=True,
                )
                staged, _ = first.stage(self.base, common, first_rows, rows)
                metadata = self.base._build(staged, 0)
                # Only GPU QSL is cleared. This is the same cold-write
                # suppression used by the current tail's per-request states.
                first.qsl.zero_()
                metadata = clone_metadata(metadata, t)
                descriptor = BatchDescriptor(
                    num_tokens=first_rows, num_reqs=batch, uniform=True
                )
                entry = SimpleNamespace(
                    batch=batch,
                    rows=rows,
                    first_rows=first_rows,
                    first=first,
                    first_metadata=metadata,
                    first_descriptor=BatchDescriptor(
                        num_tokens=first_rows, num_reqs=rows, uniform=True
                    ),
                    descriptor=descriptor,
                    tail=self.tail.entries[tail_descriptor],
                    sample_indices=t.arange(batch, device=p.device, dtype=t.int32) * 4
                    + 3,
                    rejected=t.zeros(batch, device=p.device, dtype=t.int32),
                    native_positions=t.zeros_like(self._positions_source()),
                )
                for _ in range(p.compilation_config.cudagraph_num_of_warmups):
                    self._context_call(entry, warm=True)
                t.accelerator.synchronize()
                self._context_call(entry)
                self.entries[descriptor] = entry
            t.accelerator.synchronize()
            self.cache_bindings = self.base._current_cache_bindings()
            self.captured = True

    def try_run(
        self,
        kwargs,
        common,
        sampling_metadata,
        *,
        token_indices_to_sample,
        num_rejected_tokens_gpu,
        group_metadata,
        mm_embed_inputs,
    ):
        from vllm.forward_context import BatchDescriptor

        p, t = self.proposer, self.torch
        if (
            not self.policy_supported
            or not self.captured
            or p.num_speculative_tokens != 3
            or not sampling_metadata.all_greedy
            or num_rejected_tokens_gpu is None
            or p.needs_extra_input_slots
            or p._share_mtp_indices
            or mm_embed_inputs is not None
            or getattr(p, "_enable_probabilistic_draft_probs", False)
            or common.max_seq_len + 2 >= p.max_model_len
        ):
            return None
        plan = full_draft_plan(common, self.q4_sizes, self.q1_sizes)
        if plan is None:
            return None
        first_rows, rows = plan
        n = common.num_reqs
        if (
            not t.is_tensor(token_indices_to_sample)
            or not t.is_tensor(num_rejected_tokens_gpu)
            or token_indices_to_sample.shape != (n,)
            or num_rejected_tokens_gpu.shape != (n,)
            or token_indices_to_sample.device != p.device
            or num_rejected_tokens_gpu.device != p.device
            or token_indices_to_sample.dtype not in (t.int32, t.int64)
            or num_rejected_tokens_gpu.dtype not in (t.int32, t.int64)
            or set(kwargs)
            != {"input_ids", "positions", "inputs_embeds", "hidden_states"}
            or kwargs["input_ids"] is None
            or kwargs["inputs_embeds"] is not None
        ):
            return None
        if p.allowed_attn_types is not None and any(
            not isinstance(md, p.allowed_attn_types) for md in group_metadata
        ):
            return None
        _, input_batch_size, across_dp = p._determine_batch_execution_and_padding(n)
        if (
            across_dp is not None
            or type(input_batch_size) is not int
            or not n <= input_batch_size <= p._slot_mapping_buffer.numel()
        ):
            return None
        self.base.validate_cache_binding()
        if (
            self.generation != self.base.generation
            or self.cache_bindings != self.base._current_cache_bindings()
        ):
            raise RuntimeError("full draft3 KV lease changed")
        descriptor = BatchDescriptor(num_tokens=first_rows, num_reqs=n, uniform=True)
        if descriptor not in self.entries:
            raise RuntimeError("eligible full draft3 descriptor was not captured")
        entry = self.entries[descriptor]
        source_positions = self._positions_source()
        if (
            source_positions.shape,
            source_positions.stride(),
            source_positions.dtype,
        ) != (
            entry.native_positions.shape,
            entry.native_positions.stride(),
            entry.native_positions.dtype,
        ):
            raise RuntimeError("full draft3 native position buffer layout changed")
        staged, _ = entry.first.stage(self.base, common, first_rows, rows, kwargs)
        metadata = self.base._build(staged, 0)
        refresh_metadata(
            entry.first_metadata,
            metadata,
            t,
            ignored_bool_fields=self.base.ignored_bool_fields,
        )
        entry.sample_indices.copy_(token_indices_to_sample)
        entry.rejected.copy_(num_rejected_tokens_gpu)
        entry.native_positions.copy_(source_positions)
        self._context_call(entry)
        final = entry.tail.steps[1]
        # Original proposer had changed common to q1 and subtracted GPU
        # rejections before entering tail.try_run. Preserve those side effects
        # here, after the single replay, without changing target rejection.
        p.input_ids[:n].copy_(final.ids[:n])
        p.hidden_states[:n].copy_(final.hidden[:n])
        p._set_positions(n, final.positions[..., :n])
        p._get_slot_mapping(input_batch_size, final.slots[:n])
        common.seq_lens[:n].copy_(final.seq[:n])
        common.slot_mapping = p._slot_mapping_buffer[:n]
        common.num_actual_tokens, common.max_query_len = n, 1
        common.query_start_loc = p.arange[: n + 1]
        common.query_start_loc_cpu = t.from_numpy(p.token_arange_np[: n + 1]).clone()
        common._seq_lens_cpu = common._num_computed_tokens_cpu = None
        common.max_seq_len = min(common.max_seq_len + 2, p.max_model_len)
        if common.seq_lens_cpu_upper_bound is not None:
            common.seq_lens_cpu_upper_bound += 2
        return entry.tail.proposals.clone(memory_format=t.contiguous_format)
