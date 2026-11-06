# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4 mixed prefill and MTP first/full3/tail graphs for vLLM 0.28.

Native proposal, sampling, rejection and attention metadata builders retain
ownership of their semantics. Graph owners stage complete current inputs into
persistent buffers, retain their KV leases and return fresh proposal storage.
"""

from __future__ import annotations

import copy
import weakref
from dataclasses import fields, is_dataclass
from types import SimpleNamespace

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphWrapper,
    is_breakable_cudagraph_enabled,
)
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available

from ..runtime import decode_plan, full_draft_plan
from .metadata_kernels import mtp_tail_step


def supports_mixed_config(config):
    c, s, p = (
        config.compilation_config,
        config.speculative_config,
        config.parallel_config,
    )
    return not (
        not is_breakable_cudagraph_enabled()
        or c.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE
        or c.mode != CompilationMode.NONE
        or (c.max_cudagraph_capture_size, tuple(c.cudagraph_capture_sizes))
        not in (
            (1024, (4, 8, 16, 32, 64, 128, 256, 512, 1024)),
            (2048, (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)),
        )
        or c.cudagraph_num_of_warmups < 1
        or s is None
        or s.num_speculative_tokens != 3
        or p.pipeline_parallel_size != 1
        or p.data_parallel_size != 1
        or p.decode_context_parallel_size != 1
        or p.prefill_context_parallel_size != 1
        or p.use_ubatching
        or config.lora_config is not None
    )


def validate_mixed_config(config):
    if not supports_mixed_config(config):
        raise ValueError("unsupported HYV4 native prefill graph configuration")
    # Private draft capacities are independent of prefill graph capture sizes.
    return (1, 2, 4, 8, 16, 32, 64, 128, 256)


class HYV4MixedGraphWrapper(BreakableCUDAGraphWrapper):
    """Native FULL target plus native breakable PIECEWISE on the same model."""

    def __init__(self, runnable, vllm_config):
        validate_mixed_config(vllm_config)
        self.full_wrapper = CUDAGraphWrapper(
            runnable, vllm_config, runtime_mode=CUDAGraphMode.FULL
        )
        super().__init__(runnable, vllm_config)

    def __call__(self, *args, **kwargs):
        if (
            is_forward_context_available()
            and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
        ):
            return self.full_wrapper(*args, **kwargs)
        return super().__call__(*args, **kwargs)

    def clear_graphs(self):
        super().clear_graphs()
        self.full_wrapper.clear_graphs()


class HYV4DraftPiecewiseGraphWrapper(BreakableCUDAGraphWrapper):
    """FULL belongs to the existing private manager; PIECEWISE to native BCG."""

    def __init__(self, runnable, vllm_config):
        validate_mixed_config(vllm_config)
        super().__init__(runnable, vllm_config)

    def __call__(self, *args, **kwargs):
        if (
            not is_forward_context_available()
            or get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
        ):
            return self.runnable(*args, **kwargs)
        return super().__call__(*args, **kwargs)


def wrap_native_breakable(runnable, vllm_config, *, draft=False):
    wrapper = HYV4DraftPiecewiseGraphWrapper if draft else HYV4MixedGraphWrapper
    return wrapper(runnable, vllm_config)


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


def refresh_metadata(dst, src, torch, path="metadata", ignored_bool_fields=()):
    """Refresh capture-owned tensor leaves, refusing changed launch structure."""
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
    from vllm.v1.attention.ops.flashmla import FlashMLASchedMeta

    if type(dst) is FlashMLASchedMeta or type(src) is FlashMLASchedMeta:
        # This branch is reached only through the exact mixed FP8 metadata
        # checked below. Native warmup owns the opaque scheduler outputs.
        if (
            type(dst) is not FlashMLASchedMeta
            or type(src) is not FlashMLASchedMeta
            or not dst.have_initialized
            or dst.config is None
            or dst.tile_scheduler_metadata is None
            or dst.num_splits is None
            or src.have_initialized
            or src.config is not None
            or src.tile_scheduler_metadata is not None
            or src.num_splits is not None
        ):
            raise RuntimeError(f"{path}: native FP8 scheduler ownership changed")
        return
    if type(dst) is FlashMLASparseMetadata and dst.fp8_extra_metadata is not None:
        extra = dst.fp8_extra_metadata
        config = (
            extra.scheduler_metadata.config
            if type(extra) is FlashMLASparseMetadata.FP8KernelMetadata
            else None
        )
        # Current H4/SM90 mixed-token consumer uses fixed topk work. Real
        # sequence lengths update mapping/indexer tensors, never scheduler
        # lengths. The native cache_lens is the builder's constant model limit;
        # topk_length/extra_topk_length are absent from this reader call.
        expected = (
            1,
            4 if dst.num_actual_tokens == 1 else dst.num_actual_tokens,
            64,
            64,
            1,
            False,
            True,
            2048,
            None,
            None,
        )
        if (
            type(src) is not FlashMLASparseMetadata
            or type(extra) is not FlashMLASparseMetadata.FP8KernelMetadata
            or type(src.fp8_extra_metadata)
            is not FlashMLASparseMetadata.FP8KernelMetadata
            or dst.fp8_use_mixed_batch is not True
            or src.fp8_use_mixed_batch is not True
            or dst.num_prefills
            or src.num_prefills
            or dst.prefill is not None
            or src.prefill is not None
            or extra.cache_lens.shape != (1,)
            or src.fp8_extra_metadata.cache_lens.shape != (1,)
            or config is None
            or (
                config.b,
                config.s_q,
                config.h_q,
                config.page_block_size,
                config.h_k,
                config.causal,
                config.is_fp8_kvcache,
                config.topk,
                config.extra_page_block_size,
                config.extra_topk,
            )
            != expected
        ):
            raise RuntimeError(f"{path}: unsupported FP8 scheduler dependency contract")
    if torch.is_tensor(dst) or torch.is_tensor(src):
        if not (torch.is_tensor(dst) and torch.is_tensor(src)):
            raise RuntimeError(f"{path}: tensor structure changed")
        if (dst.shape, dst.dtype, dst.device, dst.stride()) != (
            src.shape,
            src.dtype,
            src.device,
            src.stride(),
        ):
            raise RuntimeError(f"{path}: tensor layout changed")
        if dst.data_ptr() != src.data_ptr():
            dst.copy_(src)
        return
    if type(dst) is not type(src):
        raise RuntimeError(f"{path}: metadata type changed")
    if is_dataclass(dst):
        for field in fields(dst):
            if (type(dst), field.name) in ignored_bool_fields:
                # The exact INT8 consumer never reads this FP8 scheduler hint.
                # Counts/2D lengths/table/slots still refresh normally. No
                # other metadata type or scalar is relaxed.
                continue
            refresh_metadata(
                getattr(dst, field.name),
                getattr(src, field.name),
                torch,
                f"{path}.{field.name}",
                ignored_bool_fields,
            )
    elif isinstance(dst, dict):
        if dst.keys() != src.keys():
            raise RuntimeError(f"{path}: metadata keys changed")
        for key in dst:
            refresh_metadata(
                dst[key], src[key], torch, f"{path}[{key}]", ignored_bool_fields
            )
    elif isinstance(dst, (list, tuple)):
        if len(dst) != len(src):
            raise RuntimeError(f"{path}: metadata length changed")
        for i, (a, b) in enumerate(zip(dst, src)):
            refresh_metadata(a, b, torch, f"{path}[{i}]", ignored_bool_fields)
    elif dst is None or isinstance(dst, (str, bool, int, float)):
        if dst != src:
            raise RuntimeError(f"{path}: launch scalar changed: {dst!r} -> {src!r}")
    elif dst != src:
        raise RuntimeError(f"{path}: unsupported changed metadata value")


def make_hyv4_draft_graph(proposer, runner):
    """Select this model's graph manager before allocating state."""
    spec = proposer.speculative_config
    arch = spec.draft_model_config.hf_config.architectures
    if spec.method != "mtp" or arch != ["HYV4MTP"]:
        return None
    return HYV4DraftGraph(proposer, runner)


class _Workspace:
    def __init__(self, owner, q, *, hidden=None):
        t, p = owner.torch, owner.proposer
        cap = owner.sizes[-1] if q == 4 else min(64, owner.sizes[-1])
        reqs = cap // q
        table = owner.runner().input_batch.block_table[p.kv_cache_gid]
        prototype = table.get_device_tensor(reqs)
        self.q, self.cap = q, cap
        self.table = t.zeros_like(prototype)
        self.seq = t.zeros((reqs,), dtype=t.int32, device=p.device)
        self.qsl = t.zeros((reqs + 1,), dtype=t.int32, device=p.device)
        self.qsl_cpu = t.zeros((reqs + 1,), dtype=t.int32, device="cpu")
        self.upper = t.full((reqs,), p.max_model_len, dtype=t.int32, device="cpu")
        self.prefill = t.zeros((reqs,), dtype=t.bool, device="cpu")
        self.slots = t.full_like(p._slot_mapping_buffer[:cap], -1)
        self.ids = t.zeros_like(p.input_ids[:cap])
        self.positions = t.zeros_like(p._get_positions(cap))
        self.hidden = t.zeros_like(p.hidden_states[:cap]) if hidden is None else hidden

    def stage(self, owner, common, rows, reqs, kwargs=None):
        p = owner.proposer
        b, n = common.num_reqs, common.num_actual_tokens
        if common.block_table_tensor.shape[1:] != self.table.shape[1:]:
            raise RuntimeError("HYV4 draft block-table width changed")
        self.table.zero_()
        self.table[:b].copy_(common.block_table_tensor[:b])
        self.seq.zero_()
        self.seq[:b].copy_(common.seq_lens[:b])
        self.slots.fill_(-1)
        self.slots[:n].copy_(common.slot_mapping[:n])
        self.qsl_cpu[: b + 1].copy_(common.query_start_loc_cpu[: b + 1])
        self.qsl_cpu[b + 1 :].fill_(n)
        # Native no-extra uniform q4 preserves the current device QSL mirror.
        # CPU QSL above is still updated for every original metadata builder.
        gpu_qsl = common.query_start_loc
        reuse_gpu_qsl = (
            self.q == 4
            and not p.needs_extra_input_slots
            and gpu_qsl.ndim == 1
            and gpu_qsl.numel() == b + 1
            and gpu_qsl.stride() == (1,)
            and gpu_qsl.dtype == self.qsl.dtype
            and gpu_qsl.device == self.qsl.device
        )
        if reuse_gpu_qsl:
            self.qsl[: b + 1].copy_(gpu_qsl)
            self.qsl[b + 1 :].fill_(n)
        else:
            self.qsl.copy_(self.qsl_cpu)
        staged = copy.copy(common)
        staged.query_start_loc = self.qsl[: reqs + 1]
        staged.query_start_loc_cpu = self.qsl_cpu[: reqs + 1]
        staged.seq_lens = self.seq[:reqs]
        staged.block_table_tensor = self.table[:reqs]
        staged.slot_mapping = self.slots[:rows]
        staged.num_reqs, staged.num_actual_tokens = reqs, rows
        staged.max_query_len, staged.max_seq_len = self.q, p.max_model_len
        # Exact GPU lengths include native rejection correction. Do not invent
        # an exact CPU shadow from the optimistic upper bound.
        staged._seq_lens_cpu = staged._num_computed_tokens_cpu = None
        staged.seq_lens_cpu_upper_bound = self.upper[:reqs]
        staged.is_prefilling = self.prefill[:reqs]
        if kwargs is not None:
            if kwargs["inputs_embeds"] is not None or kwargs["input_ids"] is None:
                raise RuntimeError("HYV4 text-only MTP draft input contract changed")
            self.ids.zero_()
            self.positions.zero_()
            self.hidden.zero_()
            self.ids[:n].copy_(kwargs["input_ids"][:n])
            self.positions[..., :n].copy_(kwargs["positions"][..., :n])
            self.hidden[:n].copy_(kwargs["hidden_states"][:n])
        return staged, dict(
            input_ids=self.ids[:rows],
            positions=self.positions[..., :rows],
            inputs_embeds=None,
            hidden_states=self.hidden[:rows],
        )


class HYV4DraftGraph:
    def __init__(self, proposer, runner):
        self.proposer, self.runner = proposer, weakref.ref(runner)
        self.initialized = self.captured = False
        self.entries = {}
        self.ignored_bool_fields = set()
        self.generation = 0
        self.captured_generation = None
        self.cache_bindings = {}
        self.tail = None
        self.full_draft3 = None
        self.wrapper = None
        self.workspaces = {}

    @property
    def graph_entry_count(self):
        return (
            len(self.entries)
            + (len(self.tail.entries) if self.tail is not None else 0)
            + (len(self.full_draft3.entries) if self.full_draft3 is not None else 0)
        )

    def clear(self):
        """Caller joins streams first; discard graphs before releasing KV."""
        if self.full_draft3 is not None:
            self.full_draft3.clear()
            self.full_draft3 = None
        if self.tail is not None:
            self.tail.clear()
            self.tail = None
        if self.wrapper is not None:
            self.wrapper.clear_graphs()
        self.entries.clear()
        self.workspaces.clear()
        self.cache_bindings.clear()
        self.ignored_bool_fields.clear()
        self.initialized = self.captured = False
        self.captured_generation = None

    def _current_cache_bindings(self):
        # Construction is already scoped to the HYV4MTP architecture. Bind
        # actual owners even if the model has a wrapper or a renamed class.
        if not self.proposer._draft_attn_layer_names:
            raise RuntimeError("draft graph requires nonempty actual KV owners")
        t = self.torch

        def tensors(value):
            if t.is_tensor(value):
                if value.numel() == 0:
                    raise RuntimeError("draft graph cache is empty")
                return [
                    (
                        value.data_ptr(),
                        tuple(value.shape),
                        tuple(value.stride()),
                        str(value.dtype),
                        str(value.device),
                    )
                ]
            if isinstance(value, (list, tuple)):
                return [item for child in value for item in tensors(child)]
            raise RuntimeError("unknown draft KV container")

        context = self.proposer.compilation_config.static_forward_context
        result = {}
        for name in sorted(self.proposer._draft_attn_layer_names):
            layer = context[name]
            result[name] = tensors(layer.kv_cache)
            if not result[name]:
                raise RuntimeError("draft graph cache owner has no storage")
        return result

    def validate_cache_binding(self):
        """Real proposal cannot replay temporary graphs or silently go eager."""
        if (
            not self.initialized
            or not self.captured
            or self.captured_generation != self.generation
        ):
            raise RuntimeError("HYV4 draft graphs require a fresh completed capture")
        if self._current_cache_bindings() != self.cache_bindings:
            raise RuntimeError("HYV4 draft KV ownership changed after capture")

    def initialize(self, mode):
        import torch

        p, spec = self.proposer, self.proposer.speculative_config
        if self.initialized:
            self.clear()
        parallel = p.vllm_config.parallel_config
        prefill_sizes = None
        if mode == CUDAGraphMode.FULL_AND_PIECEWISE:
            prefill_sizes = validate_mixed_config(p.vllm_config)
        legal_mode = (
            mode == CUDAGraphMode.FULL_DECODE_ONLY
            if prefill_sizes is None
            else mode == CUDAGraphMode.FULL_AND_PIECEWISE
        )
        if (
            not legal_mode
            or spec.enforce_eager
            or spec.num_speculative_tokens != 3
            or spec.parallel_drafting
            or spec.disable_padded_drafter_batch
            or p._share_mtp_indices
            or parallel.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.use_ubatching
            or p.vllm_config.lora_config is not None
            or p.supports_mm_inputs
        ):
            raise ValueError("unsupported HYV4 MTP3 draft graph configuration")
        self.torch, self.mode = torch, CUDAGraphMode.FULL
        cap = (
            p.compilation_config.max_cudagraph_capture_size
            if prefill_sizes is None
            else prefill_sizes[-1]
        )
        if type(cap) is not int or cap not in (64, 256):
            raise ValueError("HYV4 draft graphs require capture capacity 64 or 256")
        if p.compilation_config.cudagraph_num_of_warmups < 1:
            raise ValueError("HYV4 draft graph requires native startup warmup")
        # Private q1 keys restore 1/2 after target q4 rounding. Target config
        # and its capture-size list are never mutated.
        self.sizes = (
            tuple(sorted({1, 2, *p.compilation_config.cudagraph_capture_sizes}))
            if prefill_sizes is None
            else prefill_sizes
        )
        if any(type(s) is not int or s < 1 or s > cap for s in self.sizes):
            raise ValueError("invalid draft graph capture sizes")
        self.wrapper = CUDAGraphWrapper(p.model, p.vllm_config, runtime_mode=self.mode)
        # Allocate inside capture() so native profiling includes this storage.
        self.workspaces = {}
        self.generation += 1
        self.initialized = True
        self.tail = HYV4TailGraph(self)
        self.full_draft3 = HYV4FullDraft3Graph(self)

    def _build(self, common, draft_index):
        p, per_layer = self.proposer, {}
        for group in p.draft_attn_groups:
            metadata = group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common, draft_index=draft_index
            )
            from .indexer import HYV4Int8IndexerBackend

            if group.backend is HYV4Int8IndexerBackend:
                from vllm.v1.attention.backends.mla.indexer import (
                    DeepSeekV32IndexerDecodeMetadata,
                )

                self.ignored_bool_fields.add(
                    (DeepSeekV32IndexerDecodeMetadata, "requires_padding")
                )
            if metadata.num_prefills:
                raise RuntimeError("draft FULL graph unexpectedly contains prefill")
            for name in group.layer_names:
                per_layer[name] = metadata
        if set(per_layer) != p._draft_attn_layer_names:
            raise RuntimeError("draft metadata owners differ from actual draft layers")
        return per_layer

    def _call(self, descriptor, metadata, kwargs, slots, *, warm=False):
        from vllm.forward_context import set_forward_context

        with set_forward_context(
            metadata,
            self.proposer.vllm_config,
            num_tokens=descriptor.num_tokens,
            cudagraph_runtime_mode=self.mode,
            batch_descriptor=descriptor,
            slot_mapping={
                name: slots for name in self.proposer._draft_attn_layer_names
            },
        ):
            return self.proposer.model(**kwargs) if warm else self.wrapper(**kwargs)

    def capture(self):
        """Match native draft inference without changing caller grad state."""
        with self.torch.inference_mode():
            return self._capture()

    def _capture(self):
        """Called once inside native capture_model's enabled capture lifecycle."""
        from vllm.forward_context import BatchDescriptor
        from vllm.v1.attention.backend import CommonAttentionMetadata

        if not self.initialized or self.captured:
            raise RuntimeError("HYV4 draft graph capture lifecycle mismatch")
        t, p = self.torch, self.proposer
        self.workspaces = {q: _Workspace(self, q) for q in (1, 4)}
        for rows in reversed(self.sizes):
            for q in (4, 1):
                if rows % q or (q == 1 and rows > 64):
                    continue
                reqs, ws = rows // q, self.workspaces[q]
                # Dummy sources are distinct from stage destinations: stage()
                # clears padding before copying the current live inputs.
                dummy_qsl_cpu = t.arange(reqs + 1, dtype=t.int32) * q
                dummy_qsl = dummy_qsl_cpu.to(device=p.device)
                common = CommonAttentionMetadata(
                    query_start_loc=dummy_qsl,
                    query_start_loc_cpu=dummy_qsl_cpu,
                    seq_lens=t.full((reqs,), q, dtype=t.int32, device=p.device),
                    num_reqs=reqs,
                    num_actual_tokens=rows,
                    max_query_len=q,
                    max_seq_len=p.max_model_len,
                    block_table_tensor=t.zeros_like(ws.table[:reqs]),
                    slot_mapping=t.full_like(ws.slots[:rows], -1),
                    causal=True,
                )
                staged, kwargs = ws.stage(self, common, rows, reqs)
                metadata = clone_metadata(self._build(staged, 0), t)
                descriptor = BatchDescriptor(
                    num_tokens=rows, num_reqs=reqs, uniform=True
                )
                for _ in range(p.compilation_config.cudagraph_num_of_warmups):
                    self._call(
                        descriptor, metadata, kwargs, staged.slot_mapping, warm=True
                    )
                t.accelerator.synchronize()
                output = self._call(descriptor, metadata, kwargs, staged.slot_mapping)
                self.entries[descriptor] = metadata
                del output
        t.accelerator.synchronize()
        self.tail.capture()
        self.full_draft3.capture()
        self.cache_bindings = self._current_cache_bindings()
        self.captured_generation = self.generation
        self.captured = True

    def run(self, kwargs, common, *, draft_index, verified_step):
        from vllm.forward_context import BatchDescriptor

        if not self.captured:
            # Native profile/warmup before the explicit capture lifecycle.
            return self.proposer.model(**kwargs)
        plan = decode_plan(common, self.sizes, verified_step=verified_step)
        if plan is None:
            return self.proposer.model(**kwargs)
        q, rows, reqs = plan
        descriptor = BatchDescriptor(num_tokens=rows, num_reqs=reqs, uniform=True)
        if descriptor not in self.entries:
            raise RuntimeError("eligible HYV4 draft graph was not captured at startup")
        staged, inputs = self.workspaces[q].stage(self, common, rows, reqs, kwargs)
        metadata = self._build(staged, draft_index)
        persistent = self.entries[descriptor]
        refresh_metadata(
            persistent,
            metadata,
            self.torch,
            ignored_bool_fields=self.ignored_bool_fields,
        )
        return self._call(descriptor, persistent, inputs, staged.slot_mapping)


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
        self.wrapper = None
        self.generation = base.generation
        self.cache_bindings = {}
        p = self.proposer
        self.q1_sizes = tuple(s for s in base.sizes if s <= 64)
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
        if self.wrapper is not None:
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

        p, t = self.proposer, self.torch
        if (
            p.num_speculative_tokens != 3
            or not self.policy_supported
            or not self.captured
            or not verified_step
            or not sampling_metadata.all_greedy
            or p._enable_probabilistic_draft_probs
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
        for value in (
            common._seq_lens_cpu,
            common._num_computed_tokens_cpu,
            common.seq_lens_cpu_upper_bound,
        ):
            if value is not None:
                value += 2
        # Native torch.stack returned fresh storage. Keep that output lease
        # under async scheduling instead of leaking a reused graph buffer.
        return entry.proposals.clone(memory_format=t.contiguous_format)


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
        self.wrapper = None
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
        if self.wrapper is not None:
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
            or p._enable_probabilistic_draft_probs
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
