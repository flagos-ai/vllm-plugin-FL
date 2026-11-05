# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 two-step draft tail with native sampling and rejection semantics."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass

from .runtime_mtp_state import mtp_tail_step


def clone_metadata(value, torch, memo=None):
    """Detach builder workspaces; preserve repeated object identity per phase."""
    from vllm.v1.attention.ops.flashmla import FlashMLASchedMeta, get_mla_metadata

    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if type(value) is FlashMLASchedMeta:
        # Native scheduling buffers are opaque outputs, not builder tensor
        # leaves. Each phase starts with a fresh public native owner and warms
        # it before capture; never clone an initialized scheduler/descriptor.
        if (
            value.have_initialized
            or value.config is not None
            or value.tile_scheduler_metadata is not None
            or value.num_splits is not None
        ):
            raise ValueError("capture metadata requires a fresh native FP8 scheduler")
        result, _ = get_mla_metadata()
    elif torch.is_tensor(value):
        result = value.clone()
    elif is_dataclass(value):
        result = copy.copy(value)
        memo[id(value)] = result
        for field in fields(value):
            setattr(
                result,
                field.name,
                clone_metadata(getattr(value, field.name), torch, memo),
            )
    elif isinstance(value, dict):
        result = {k: clone_metadata(v, torch, memo) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        result = type(value)(clone_metadata(v, torch, memo) for v in value)
    elif value is None or isinstance(value, (str, bool, int, float)):
        return value
    else:
        raise ValueError(f"unsupported tail metadata leaf {type(value)}")
    memo[id(value)] = result
    return result


class _State:
    def __init__(self, owner, rows, batch):
        t, p = owner.torch, owner.proposer
        table = owner.base.runner().input_batch.block_table[p.kv_cache_gid]
        self.table = t.zeros_like(table.get_device_tensor(rows))
        self.seq = t.zeros(rows, device=p.device, dtype=t.int32)
        self.slots = t.full((rows,), -1, device=p.device, dtype=t.int64)
        self.counts = t.zeros_like(self.seq)
        self.ids = t.zeros_like(p.input_ids[:rows])
        self.hidden = t.zeros_like(p.hidden_states[:rows])
        self.positions = t.zeros_like(p._get_positions(rows))
        self.qsl_cpu = t.arange(rows + 1, dtype=t.int32, device="cpu").clamp_(max=batch)
        self.qsl = t.zeros(rows + 1, device=p.device, dtype=t.int32)
        self.upper = t.full((rows,), p.max_model_len, dtype=t.int32, device="cpu")
        self.prefill = t.zeros(rows, dtype=t.bool, device="cpu")

    def common(self, owner, rows):
        from vllm.v1.attention.backend import CommonAttentionMetadata

        return CommonAttentionMetadata(
            query_start_loc=self.qsl,
            query_start_loc_cpu=self.qsl_cpu,
            seq_lens=self.seq,
            num_reqs=rows,
            num_actual_tokens=rows,
            max_query_len=1,
            max_seq_len=owner.proposer.max_model_len,
            block_table_tensor=self.table,
            slot_mapping=self.slots,
            causal=True,
            seq_lens_cpu_upper_bound=self.upper,
            is_prefilling=self.prefill,
        )


def bind_phase_metadata(raw, state, torch):
    """Bind current pure-decode FlashMLA and independent INT8 phase inputs."""
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata

    result = clone_metadata(raw, torch)
    for md in result.values():
        if type(md) is FlashMLASparseMetadata:
            if md.num_prefills or md.prefill is not None:
                raise ValueError("tail graph requires pure-decode FlashMLA")
            if md.fp8_extra_metadata is not None and (
                type(md.fp8_extra_metadata)
                is not FlashMLASparseMetadata.FP8KernelMetadata
                or md.fp8_use_mixed_batch is not True
                or md.max_query_len != 1
                or md.num_actual_tokens != state.seq.numel()
                or md.block_size != 64
                or md.topk_tokens != 2048
                or md.fp8_extra_metadata.cache_lens.shape != (1,)
                or md.fp8_extra_metadata.dummy_block_table.shape != (1, 1)
            ):
                raise ValueError(
                    "tail graph requires current mixed-token FP8 decode metadata"
                )
            md.query_start_loc, md.slot_mapping = state.qsl, state.slots
            md.block_table, md.seq_lens = state.table, state.seq
            # req_id_per_token stays a private clone of the native CPU-QSL
            # builder result. This descriptor fixes actual B, including pads.
        elif type(md) is DeepseekV32IndexerMetadata:
            d = md.decode
            if (
                md.num_prefills
                or md.prefill is not None
                or d is None
                or d.global_seq_lens is not None
                or d.indices is not None
                or d.seq_lens.shape != (state.seq.numel(), 1)
                or d.block_table.shape != state.table.shape
            ):
                raise ValueError("tail graph requires current q1 INT8 metadata")
            md.seq_lens, md.slot_mapping = state.seq, state.slots
            d.block_table, d.seq_lens = state.table, state.seq[:, None]
            d.decode_lens = state.counts
        else:
            raise ValueError(f"unsupported tail backend metadata {type(md)}")
    return result


class HYV4TailGraph:
    """One outer native wrapper per (padded model rows, actual LM-head B)."""

    def __init__(self, base):
        self.base, self.proposer, self.torch = base, base.proposer, base.torch
        self.entries = {}
        self.captured = False
        self.generation = base.generation
        self.cache_bindings = {}
        p = self.proposer
        self.q1_sizes = tuple(
            s for s in getattr(base, "q1_sizes", base.sizes) if s <= 64
        )
        if not self.q1_sizes or self.q1_sizes[-1] != 64:
            raise ValueError("tail graph requires an explicit q1 capacity of 64")
        self.policy_supported = not (
            p.speculative_config.draft_sample_method != "greedy"
            or p.model_returns_tuple()
            or p.use_heterogeneous_vocab
            or p.use_local_argmax_reduction
            or p.constant_draft_positions
            or p.uses_xdrope_dim
            or p.supports_mm_inputs
            or not p.pass_hidden_states_to_model
            or p.eplb_state is not None
        )

    def clear(self):
        if hasattr(self, "wrapper"):
            self.wrapper.clear_graphs()
        self.entries.clear()
        self.cache_bindings.clear()
        self.captured = False

    def _step(self, entry, previous, state):
        # qsl and table are request-order owned and remain fixed for both
        # steps. Device counts also suppress every dummy capture KV write.
        mtp_tail_step(
            previous.positions,
            previous.seq,
            entry.base.table,
            entry.base.qsl,
            state.positions,
            state.seq,
            state.slots,
            state.counts,
            active_batch=entry.batch,
            block_size=self.proposer.block_size,
            max_model_len=self.proposer.max_model_len,
        )

    def _body(self, entry, state, metadata):
        from vllm.forward_context import set_forward_context

        p = self.proposer
        with set_forward_context(
            metadata,
            p.vllm_config,
            num_tokens=entry.rows,
            cudagraph_runtime_mode=self.base.mode,
            batch_descriptor=entry.descriptor,
            slot_mapping={name: state.slots for name in p._draft_attn_layer_names},
        ):
            # Deliberately bypass the existing single-body CUDAGraphWrapper.
            # No captured/replayed child graph is assumed capture-compatible.
            return p.model(
                input_ids=state.ids,
                positions=state.positions,
                inputs_embeds=None,
                hidden_states=state.hidden,
            )

    def _greedy(self, hidden):
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        # Native sampling runs after the model-body forward context exits.
        # Preserve its NONE dispatch context and its native LM-head/TP/argmax.
        with set_forward_context(
            None,
            self.proposer.vllm_config,
            num_tokens=hidden.shape[0],
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ):
            return self.proposer._greedy_sample(hidden)

    def _chain(self, entry):
        a, b, n = entry.steps[0], entry.steps[1], entry.batch
        entry.proposals[:, 0].copy_(entry.seed)
        a.ids.copy_(entry.base.ids)
        a.hidden.copy_(entry.base.hidden)
        self._step(entry, entry.base, a)
        a.qsl.copy_(entry.base.qsl)
        hidden1 = self._body(entry, a, entry.metadata[0])
        token1 = self._greedy(hidden1[:n])
        entry.proposals[:, 1].copy_(token1)
        b.ids.zero_()
        b.ids[:n].copy_(token1)
        b.hidden.zero_()
        b.hidden[:n].copy_(hidden1[:n])
        self._step(entry, a, b)
        b.qsl.copy_(entry.base.qsl)
        hidden2 = self._body(entry, b, entry.metadata[1])
        token2 = self._greedy(hidden2[:n])
        entry.proposals[:, 2].copy_(token2)
        return entry.proposals

    def _context_call(self, entry, warm=False):
        from vllm.forward_context import set_forward_context

        with set_forward_context(
            {},
            self.proposer.vllm_config,
            num_tokens=entry.rows,
            cudagraph_runtime_mode=self.base.mode,
            batch_descriptor=entry.descriptor,
            slot_mapping={},
        ):
            return self._chain(entry) if warm else self.wrapper(entry)

    def capture(self):
        from types import SimpleNamespace

        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.forward_context import BatchDescriptor

        if not self.policy_supported:
            return
        if self.captured or self.generation != self.base.generation:
            raise RuntimeError("tail capture generation mismatch")
        t, p = self.torch, self.proposer
        self.wrapper = CUDAGraphWrapper(
            self._chain, p.vllm_config, runtime_mode=self.base.mode
        )
        with t.inference_mode():
            # Exact B retains the native LM-head/AllGather/argmax input shape.
            # B7 in an M8 body is distinct from B8, instead of changing math.
            for batch in range(self.q1_sizes[-1], 0, -1):
                rows = next(s for s in self.q1_sizes if s >= batch)
                descriptor = BatchDescriptor(
                    num_tokens=rows, num_reqs=batch, uniform=True
                )
                entry = SimpleNamespace(
                    batch=batch,
                    rows=rows,
                    descriptor=descriptor,
                    base=_State(self, rows, batch),
                    steps=(_State(self, rows, batch), _State(self, rows, batch)),
                    seed=t.zeros(batch, device=p.device, dtype=t.int64),
                    proposals=t.zeros(
                        (3, batch), device=p.device, dtype=t.int64
                    ).transpose(0, 1),
                )
                # qsl_cpu establishes the fixed launch descriptor. The GPU
                # qsl is all-zero here, so both capture phases have slot=-1.
                phases = []
                for i, state in enumerate(entry.steps):
                    raw = self.base._build(state.common(self, rows), i + 1)
                    phases.append(bind_phase_metadata(raw, state, t))
                entry.metadata = tuple(phases)
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
        seed,
        hidden,
        positions,
        common,
        sampling_metadata,
        *,
        batch_size,
        input_batch_size,
        verified_step,
    ):
        from vllm.forward_context import BatchDescriptor

        from .mtp_draft_graph import decode_plan

        p, t = self.proposer, self.torch
        if (
            p.num_speculative_tokens != 3
            or not self.policy_supported
            or not self.captured
            or not verified_step
            or not sampling_metadata.all_greedy
            or getattr(p, "_enable_probabilistic_draft_probs", False)
            or common.max_seq_len + 2 >= p.max_model_len
            or type(input_batch_size) is not int
            or not batch_size <= input_batch_size <= p._slot_mapping_buffer.numel()
        ):
            return None
        plan = decode_plan(common, self.q1_sizes, verified_step=verified_step)
        if plan is None or plan[0] != 1 or common.num_reqs != batch_size:
            return None
        if (
            self.generation != self.base.generation
            or self.cache_bindings != self.base._current_cache_bindings()
        ):
            raise RuntimeError("tail graph KV lease changed")
        _, rows, _ = plan
        descriptor = BatchDescriptor(num_tokens=rows, num_reqs=batch_size, uniform=True)
        if descriptor not in self.entries:
            raise RuntimeError("eligible tail descriptor was not captured")
        entry = self.entries[descriptor]
        state = entry.base
        if common.block_table_tensor.shape[1:] != state.table.shape[1:]:
            raise RuntimeError("tail block table layout changed")
        state.table.zero_()
        state.seq.zero_()
        state.ids.zero_()
        state.hidden.zero_()
        state.positions.zero_()
        state.table[:batch_size].copy_(common.block_table_tensor[:batch_size])
        state.seq[:batch_size].copy_(common.seq_lens[:batch_size])
        state.qsl.copy_(state.qsl_cpu)
        state.ids[:batch_size].copy_(seed)
        state.hidden[:batch_size].copy_(hidden[:batch_size])
        state.positions[..., :batch_size].copy_(positions[..., :batch_size])
        entry.seed.copy_(seed)
        for phase in entry.steps:
            phase.table.copy_(state.table)
        self._context_call(entry)
        final = entry.steps[1]
        # Match native loop side effects: buffers hold the second model's
        # INPUT IDs/hidden, while positions/seq/slots are after two updates.
        p.input_ids[:batch_size].copy_(final.ids[:batch_size])
        p.hidden_states[:batch_size].copy_(final.hidden[:batch_size])
        p._set_positions(batch_size, final.positions[..., :batch_size])
        p._get_slot_mapping(input_batch_size, final.slots[:batch_size])
        common.seq_lens[:batch_size].copy_(final.seq[:batch_size])
        common.slot_mapping = p._slot_mapping_buffer[:batch_size]
        common.max_seq_len = min(common.max_seq_len + 2, p.max_model_len)
        for name in (
            "_seq_lens_cpu",
            "_num_computed_tokens_cpu",
            "seq_lens_cpu_upper_bound",
        ):
            value = getattr(common, name, None)
            if value is not None:
                value += 2
        # Native torch.stack returned fresh storage. Keep that output lease
        # under async scheduling instead of leaking a reused graph buffer.
        return entry.proposals.clone(memory_format=t.contiguous_format)
