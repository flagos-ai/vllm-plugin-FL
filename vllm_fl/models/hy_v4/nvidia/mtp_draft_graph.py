# SPDX-License-Identifier: Apache-2.0
"""HYV4 MTP3 model-body graphs; native proposer owns all semantics.

This module does not capture sampling, rejection, input rotation, metadata
builders, or position/KV updates. It only stages their complete current results
before calling the unchanged draft model through vLLM's CUDA graph wrapper.
"""

from __future__ import annotations

import copy
import weakref
from dataclasses import fields, is_dataclass

from .mtp_tail_graph import clone_metadata


def decode_plan(common, sizes, *, verified_step):
    """Return (q, padded rows, padded requests), or native eager fallback.

    CPU QSL describes launch shape. Rejected rows remain present and their GPU
    slots/lengths remain authoritative; we never compact or accept tokens here.
    The first proposal after prefill is intentionally excluded.
    """
    if not verified_step:
        return None
    q = common.max_query_len
    b, n = common.num_reqs, common.num_actual_tokens
    if type(q) is not int or q not in (1, 4) or b < 1 or n != b * q:
        return None
    if q == 1 and n > 64:
        return None
    starts = common.query_start_loc_cpu.tolist()
    if len(starts) != b + 1 or starts[0] != 0 or starts[-1] != n:
        return None
    if any(v - u != q for u, v in zip(starts, starts[1:])):
        return None
    prefilling = getattr(common, "is_prefilling", None)
    if prefilling is not None and any(prefilling.tolist()):
        return None
    if common.causal is not True or getattr(common, "mm_req_doc_ranges", None):
        return None
    padded = next((s for s in sizes if s >= n and s % q == 0), None)
    return None if padded is None else (q, padded, padded // q)


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
            or not torch.is_tensor(dst.tile_scheduler_metadata)
            or not torch.is_tensor(dst.num_splits)
            or src.have_initialized
            or src.config is not None
            or src.tile_scheduler_metadata is not None
            or src.num_splits is not None
        ):
            raise RuntimeError(f"{path}: native FP8 scheduler ownership changed")
        return
    if type(dst) is FlashMLASparseMetadata and dst.fp8_extra_metadata is not None:
        extra = dst.fp8_extra_metadata
        config = getattr(getattr(extra, "scheduler_metadata", None), "config", None)
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
        config_fields = (
            "b",
            "s_q",
            "h_q",
            "page_block_size",
            "h_k",
            "causal",
            "is_fp8_kvcache",
            "topk",
            "extra_page_block_size",
            "extra_topk",
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
            or tuple(getattr(config, name) for name in config_fields) != expected
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
                if (
                    type(getattr(dst, field.name)) is not bool
                    or type(getattr(src, field.name)) is not bool
                ):
                    raise RuntimeError(f"{path}.{field.name}: expected bool hint")
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
    arch = getattr(spec.draft_model_config.hf_config, "architectures", None)
    if spec.method != "mtp" or arch != ["HYV4MTP"]:
        return None
    if runner is None:
        raise ValueError("HYV4 draft graph requires its native runner")
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
        t, p = owner.torch, owner.proposer
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
            and decode_plan(common, owner.sizes, verified_step=True) is not None
            and t.is_tensor(gpu_qsl)
            and gpu_qsl.ndim == 1
            and gpu_qsl.numel() == b + 1
            and gpu_qsl.stride() == (1,)
            and gpu_qsl.dtype == self.qsl.dtype
            and gpu_qsl.device == self.qsl.device
            and not gpu_qsl.is_conj()
            and not gpu_qsl.is_neg()
            and gpu_qsl.untyped_storage().data_ptr()
            != self.qsl.untyped_storage().data_ptr()
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
            if (
                kwargs.get("inputs_embeds") is not None
                or kwargs.get("input_ids") is None
            ):
                raise RuntimeError("HYV4 text-only MTP draft input contract changed")
            if set(kwargs) != {
                "input_ids",
                "positions",
                "inputs_embeds",
                "hidden_states",
            }:
                raise RuntimeError("HYV4 draft forward kwargs changed")
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
        if hasattr(self, "wrapper"):
            self.wrapper.clear_graphs()
        self.entries.clear()
        if hasattr(self, "workspaces"):
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

        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.config import CUDAGraphMode

        p, spec = self.proposer, self.proposer.speculative_config
        if self.initialized:
            self.clear()
        parallel = p.vllm_config.parallel_config
        prefill_sizes = None
        if mode == CUDAGraphMode.FULL_AND_PIECEWISE:
            from .prefill_graph import validate_mixed_config

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
            or getattr(
                spec.draft_model_config.hf_config,
                "index_share_for_mtp_iteration",
                False,
            )
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
        if spec.num_speculative_tokens == 3:
            from .mtp_tail_graph import HYV4TailGraph

            self.tail = HYV4TailGraph(self)
        if self.tail is not None:
            if self.tail is None:
                raise ValueError("full draft3 graph requires the existing tail2 owner")
            from .mtp_full_draft3_graph import HYV4FullDraft3Graph

            self.full_draft3 = HYV4FullDraft3Graph(self)

    def _build(self, common, draft_index):
        p, per_layer = self.proposer, {}
        for group in p.draft_attn_groups:
            metadata = group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=common, draft_index=draft_index
            )
            if hasattr(group, "backend"):
                from .int8_indexer_backend import HYV4Int8IndexerBackend

                if group.backend is HYV4Int8IndexerBackend:
                    from vllm.v1.attention.backends.mla.indexer import (
                        DeepSeekV32IndexerDecodeMetadata,
                    )

                    self.ignored_bool_fields.add(
                        (DeepSeekV32IndexerDecodeMetadata, "requires_padding")
                    )
            if getattr(metadata, "num_prefills", 0):
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
        if self.tail is not None:
            self.tail.capture()
        if self.full_draft3 is not None:
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
