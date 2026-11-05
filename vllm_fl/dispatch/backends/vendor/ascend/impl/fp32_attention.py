# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP32 causal attention and owner-scoped decode graphs for Ascend."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.v1.attention.backend import AttentionType

from vllm_fl.compilation.graph import GraphWrapper

_OWNER = ContextVar("fl_fp32_graph_owner", default=None)


@contextmanager
def graph_owner(wrapper):
    """Make the capturing wrapper available to attention custom operators."""
    token = _OWNER.set(wrapper)
    try:
        yield
    finally:
        _OWNER.reset(token)


def _tensor_geometry(tensor):
    return (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tensor.stride(),
        tensor.data_ptr(),
    )


@dataclass
class DecodeState:
    signature: tuple
    limit: int
    lengths: torch.Tensor
    slots: torch.Tensor
    table: torch.Tensor

    def update(self, meta):
        n = self.lengths.shape[0]
        self.lengths.copy_(meta.seq_lens[:n])
        self.slots.copy_(meta.slot_mapping[:n])
        self.table.copy_(meta.block_tables[:n])


def prepare_graph(wrapper, ctx):
    """Invalidate graphs before replacing metadata or changed cache geometry."""
    metas = ctx.attn_metadata
    if not isinstance(metas, dict) or ctx.batch_descriptor is None:
        return False
    n = ctx.batch_descriptor.num_tokens
    if any(
        meta is None
        or int(meta.num_actual_tokens) != n
        or len(meta.seq_lens_list) != n
        or any(int(end) != i + 1 for i, end in enumerate(meta.actual_seq_lengths_q))
        for meta in metas.values()
    ):
        return False
    limit = wrapper.vllm_config.model_config.max_model_len
    specifications = {}
    for name, meta in metas.items():
        layer = ctx.no_compile_layers.get(name)
        cache = getattr(layer, "kv_cache", None)
        if not isinstance(cache, torch.Tensor) or cache.ndim != 5:
            return False
        block_size = cache.shape[2]
        if meta.block_tables.shape[1] * block_size < limit:
            raise ValueError("FP32 graph block table cannot cover max_model_len")
        signature = (
            id(layer),
            limit,
            _tensor_geometry(cache),
            meta.block_tables.device,
            meta.block_tables.dtype,
            tuple(meta.block_tables[:n].shape),
            meta.seq_lens.device,
            meta.slot_mapping.device,
        )
        specifications[name] = (signature, meta)
    states = wrapper.__dict__.get("_fp32_decode_states", {})
    batch_states = states.get(ctx.batch_descriptor, {})
    if set(batch_states) != set(specifications) or any(
        batch_states[name].signature != specification[0]
        for name, specification in specifications.items()
        if name in batch_states
    ):
        # Never resize a tensor held by a captured graph. Drop graph entries
        # first, then allocate a separate state with the new geometry.
        wrapper.concrete_graph_entries.pop(ctx.batch_descriptor, None)
        states.pop(ctx.batch_descriptor, None)
        states = wrapper.__dict__.setdefault("_fp32_decode_states", {})
        batch_states = {}
        for name, (signature, meta) in specifications.items():
            batch_states[name] = DecodeState(
                signature,
                limit,
                torch.empty_like(
                    meta.seq_lens[:n],
                    device=ctx.no_compile_layers[name].kv_cache.device,
                    dtype=torch.int32,
                ),
                torch.empty_like(
                    meta.slot_mapping[:n],
                    device=ctx.no_compile_layers[name].kv_cache.device,
                    dtype=torch.int64,
                ),
                torch.empty_like(
                    meta.block_tables[:n],
                    device=ctx.no_compile_layers[name].kv_cache.device,
                    dtype=torch.int64,
                ),
            )
        states[ctx.batch_descriptor] = batch_states
    for name, (_, meta) in specifications.items():
        batch_states[name].update(meta)
    return True


def _full_attention(layer, model, query, key, value, cache, meta, output):
    from vllm.forward_context import get_forward_context

    owner = _OWNER.get()
    if owner is None:
        raise RuntimeError("FP32 FULL attention requires a graph owner")
    ctx = get_forward_context()
    state = owner._fp32_decode_states[ctx.batch_descriptor][layer.layer_name]
    n = query.shape[0]
    kc, vc = cache[0], cache[1]
    model.key_cache, model.value_cache = kc, vc
    tensors = (query, key, value, kc, vc, output)
    if any(t.dtype != torch.float32 or t.device != query.device for t in tensors):
        raise ValueError("FP32 FULL attention requires matching FP32 tensors")
    if not kc.is_contiguous() or not vc.is_contiguous():
        raise NotImplementedError("FP32 FULL requires contiguous ND KV cache")
    with torch.autocast(device_type="npu", enabled=False):
        kflat = kc.view(-1, model.num_kv_heads, model.head_size)
        vflat = vc.view_as(kflat)
        slots = state.slots.clamp_min(0)
        kflat.index_copy_(0, slots, key.view(n, model.num_kv_heads, model.head_size))
        vflat.index_copy_(0, slots, value.view(n, model.num_kv_heads, model.head_size))
        positions = torch.arange(state.limit, device=query.device, dtype=torch.int64)
        repeat = model.num_heads // model.num_kv_heads
        for i in range(n):
            physical = (
                state.table[i].index_select(0, positions // kc.shape[1]) * kc.shape[1]
                + positions % kc.shape[1]
            )
            k = (
                kflat.index_select(0, physical)
                .transpose(0, 1)
                .repeat_interleave(repeat, 0)
            )
            v = (
                vflat.index_select(0, physical)
                .transpose(0, 1)
                .repeat_interleave(repeat, 0)
            )
            q = query[i].reshape(model.num_heads, 1, model.head_size)
            scores = torch.matmul(q, k.transpose(-1, -2)) * model.scale
            scores = scores.masked_fill(
                (positions >= state.lengths[i].clamp_min(1))[None, None], -torch.inf
            )
            result = torch.matmul(torch.softmax(scores, -1, dtype=torch.float32), v)
            output[i].copy_(result.reshape_as(output[i]))
    return output


def _paged_attention(
    query,
    key,
    value,
    key_cache,
    value_cache,
    metadata,
    num_heads,
    num_kv_heads,
    head_size,
    scale,
    output,
):
    tensors = [query, key_cache, value_cache, output]
    tensors.extend(t for t in [key, value] if t is not None)
    if any(t.dtype != torch.float32 or t.device.type != "npu" for t in tensors):
        raise RuntimeError("Strict FP32 requires NPU float32 Q/K/V, cache and output")
    if not key_cache.is_contiguous() or not value_cache.is_contiguous():
        raise NotImplementedError("FP32 adapter requires contiguous ND KV layout")
    count = int(metadata.num_actual_tokens)
    ends, lengths = list(metadata.actual_seq_lengths_q), list(metadata.seq_lens_list)
    if not lengths or len(lengths) != len(ends) or ends[-1] != count:
        raise RuntimeError("Invalid packed sequence metadata")
    if any(end <= start for start, end in zip([0] + ends[:-1], ends)):
        raise RuntimeError("Empty or unordered query sequence")
    keys_flat = key_cache.view(-1, num_kv_heads, head_size)
    values_flat = value_cache.view(-1, num_kv_heads, head_size)
    if key is not None:
        slots = metadata.slot_mapping[:count].to(device=query.device, dtype=torch.int64)
        keys_flat.index_copy_(
            0, slots, key.reshape(-1, num_kv_heads, head_size)[:count]
        )
        values_flat.index_copy_(
            0, slots, value.reshape(-1, num_kv_heads, head_size)[:count]
        )
    output.zero_()
    start = 0
    for sequence, (end, length) in enumerate(zip(ends, lengths)):
        end, length = int(end), int(length)
        queries = end - start
        if queries > length:
            raise RuntimeError("Query is longer than its sequence")
        block_size = key_cache.shape[1]
        positions = torch.arange(length, device=query.device, dtype=torch.int64)
        table = metadata.block_tables[sequence].to(torch.int64)
        physical = (
            table.index_select(0, positions // block_size) * block_size
            + positions % block_size
        )
        cached_k = keys_flat.index_select(0, physical)
        cached_v = values_flat.index_select(0, physical)
        repeat = num_heads // num_kv_heads
        q = query[start:end].reshape(queries, num_heads, head_size).transpose(0, 1)
        k = cached_k.transpose(0, 1).repeat_interleave(repeat, dim=0)
        v = cached_v.transpose(0, 1).repeat_interleave(repeat, dim=0)
        query_positions = (
            torch.arange(queries, device=query.device, dtype=torch.int64)
            + length
            - queries
        )
        causal_mask = positions[None, :] > query_positions[:, None]
        with torch.autocast(device_type="npu", enabled=False):
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale
            scores = scores.masked_fill(causal_mask[None], float("-inf"))
            probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
            attended = torch.matmul(probabilities, v)
        output.reshape(-1, num_heads, head_size)[start:end].copy_(
            attended.transpose(0, 1)
        )
        start = end
    return output


class FP32GraphWrapper(GraphWrapper):
    """Use ordinary graph replay with stable FP32 attention metadata."""

    def __init__(self, *args, graph_runtime=None, **kwargs):
        # FP32 attention captures tensor operations directly; it has no native
        # attention tasks to update through the ordinary NPU task controller.
        super().__init__(*args, graph_runtime=None, **kwargs)

    def clear_graphs(self):
        super().clear_graphs()
        self.__dict__.get("_fp32_decode_states", {}).clear()

    def __call__(self, *args, **kwargs):
        if not is_forward_context_available():
            return super().__call__(*args, **kwargs)
        context = get_forward_context()
        if context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            return super().__call__(*args, **kwargs)
        if not prepare_graph(self, context):
            saved_mode = context.cudagraph_runtime_mode
            context.cudagraph_runtime_mode = CUDAGraphMode.NONE
            try:
                return self.runnable(*args, **kwargs)
            finally:
                context.cudagraph_runtime_mode = saved_mode
        with graph_owner(self):
            return super().__call__(*args, **kwargs)


def forward_fp32(model, layer, query, key, value, cache, metadata, output):
    """Keep eager and FULL on the same supported causal FP32 contract."""
    parallel = model.vllm_config.parallel_config
    if (
        parallel.tensor_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or model.sliding_window is not None
    ):
        raise NotImplementedError(
            "FP32 attention requires TP=1, PCP=1, DCP=1 without sliding window"
        )
    if model.attn_type != AttentionType.DECODER:
        raise NotImplementedError("FP32 attention supports causal decoders only")
    if model.alibi_slopes is not None or model.sinks is not None:
        raise NotImplementedError("FP32 attention does not support ALiBi or sinks")
    if metadata is None:
        return output.zero_()
    if not metadata.causal:
        raise NotImplementedError("FP32 attention requires causal metadata")
    if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL:
        return _full_attention(layer, model, query, key, value, cache, metadata, output)
    model.key_cache, model.value_cache = cache[0], cache[1]
    return _paged_attention(
        query,
        key,
        value,
        model.key_cache,
        model.value_cache,
        metadata,
        model.num_heads,
        model.num_kv_heads,
        model.head_size,
        model.scale,
        output,
    )
