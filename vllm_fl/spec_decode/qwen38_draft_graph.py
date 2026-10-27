# SPDX-License-Identifier: Apache-2.0
"""Opt-in full-forward graph adapter for the single QSA-only Qwen MTP layer.

Native proposer token alignment, rejection, sampling and KV ownership remain
unchanged. Metadata and inputs are copied to fixed addresses outside capture.
QSA writes replace the same slots, so capture warmup is idempotent; this must
not be used for recurrent layers or PLE. Larger prefill forwards stay eager.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass

import torch

from vllm.config import CUDAGraphMode
from vllm.distributed import graph_capture
from vllm.forward_context import get_forward_context, set_forward_context
from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import (
    PreparedQSAForwardMetadata, qsa_graph_columns,
)


@dataclass
class DraftGraphEntry:
    inputs: dict
    metadata: dict
    graph: object = None
    outputs: object = None


def fixed_qsa_metadata(source: PreparedQSAForwardMetadata, bucket: int, max_reqs: int):
    """Allocate a stable graph layout; only values may change between replays."""
    result = copy(source)
    device = source.slot_mapping.device
    result.num_actual_tokens = bucket
    result.block_table = torch.zeros(
        (max_reqs, source.block_table.shape[1]),
        dtype=source.block_table.dtype, device=device,
    )
    result.seq_lens = torch.zeros(max_reqs, dtype=source.seq_lens.dtype, device=device)
    result.query_start_loc = torch.zeros(max_reqs + 1, dtype=source.query_start_loc.dtype, device=device)
    result.slot_mapping = torch.full((bucket,), -1, dtype=source.slot_mapping.dtype, device=device)
    result.token_to_req = torch.zeros(bucket, dtype=source.token_to_req.dtype, device=device)
    result.logical_positions = torch.full((bucket,), -1, dtype=source.logical_positions.dtype, device=device)
    return result


def copy_qsa_metadata(target: PreparedQSAForwardMetadata, source: PreparedQSAForwardMetadata):
    if target.score_columns != source.score_columns:
        raise ValueError("Draft QSA score width changed within a graph entry")
    count = source.num_actual_tokens
    requests = source.seq_lens.numel()
    if count > target.num_actual_tokens or requests > target.seq_lens.numel():
        raise ValueError('Draft graph metadata exceeds fixed token/request capacity')
    if source.block_table.shape[1] != target.block_table.shape[1]:
        raise ValueError('Draft graph block-table width changed after capture')
    target.block_table.zero_()
    target.block_table[:source.block_table.shape[0]].copy_(source.block_table)
    target.seq_lens.zero_()
    target.seq_lens[:requests].copy_(source.seq_lens)
    target.query_start_loc.fill_(count)
    target.query_start_loc[:source.query_start_loc.numel()].copy_(source.query_start_loc)
    for name, fill in [('slot_mapping', -1), ('token_to_req', 0), ('logical_positions', -1)]:
        dst, src = getattr(target, name), getattr(source, name)
        dst.fill_(fill)
        dst[:count].copy_(src[:count])


class Qwen38DraftGraph:
    """A model wrapper, scoped to Qwen K1 with independent QSA cache groups."""
    def __init__(self, model, vllm_config):
        self.model = model
        self.vllm_config = vllm_config
        self.max_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_tokens = 2 * self.max_reqs
        sizes = [1]
        while sizes[-1] < self.max_tokens:
            sizes.append(min(2 * sizes[-1], self.max_tokens))
        self.capture_sizes = sizes
        self.entries = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.capture_count = 0
        self.replay_count = 0
        self.eager_count = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def _context(self, entry, bucket):
        return set_forward_context(
            entry.metadata, self.vllm_config, num_tokens=bucket,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping={name: meta.slot_mapping for name, meta in entry.metadata.items()},
        )

    def __call__(self, **kwargs):
        metadata = get_forward_context().attn_metadata
        count = kwargs['positions'].shape[-1]
        valid = (
            isinstance(metadata, dict) and metadata
            and all(isinstance(m, PreparedQSAForwardMetadata) for m in metadata.values())
            and kwargs.get('input_ids') is not None
            and kwargs.get('inputs_embeds') is None
            and kwargs.get('spec_step_idx', 0) == 0
            and 0 < count <= self.max_tokens
        )
        if not valid:
            self.eager_count += int(isinstance(metadata, dict) and bool(metadata))
            return self.model(**kwargs)
        bucket = next(size for size in self.capture_sizes if size >= count)
        graph_key = (bucket, qsa_graph_columns(metadata))
        entry = self.entries.get(graph_key)
        if entry is None:
            inputs = {}
            for name in ('input_ids', 'positions', 'hidden_states'):
                source = kwargs[name]
                if source.ndim == 0 or source.shape[0] != count:
                    raise ValueError('Qwen MTP graph requires flat text token inputs')
                inputs[name] = torch.zeros((bucket, *source.shape[1:]), dtype=source.dtype, device=source.device)
            entry = DraftGraphEntry(inputs, {
                name: fixed_qsa_metadata(meta, bucket, self.max_reqs)
                for name, meta in metadata.items()
            })
            self.entries[graph_key] = entry
        if set(metadata) != set(entry.metadata):
            raise ValueError('Qwen MTP attention-layer set changed after capture')
        for name, buffer in entry.inputs.items():
            buffer.zero_()
            buffer[:count].copy_(kwargs[name])
        for name, meta in metadata.items():
            copy_qsa_metadata(entry.metadata[name], meta)
        if entry.graph is None:
            # The MTP has no recurrent/PLE state. Rewriting its same main/raw/
            # compressed QSA slots during warmup/capture does not advance state.
            current = torch.cuda.current_stream()
            with graph_capture(kwargs['positions'].device) as capture:
                with self._context(entry, bucket):
                    self.model(**entry.inputs)
                capture.stream.synchronize()
                entry.graph = torch.cuda.CUDAGraph()
                with self._context(entry, bucket), torch.cuda.graph(
                    entry.graph, pool=self.pool, stream=capture.stream,
                ):
                    entry.outputs = self.model(**entry.inputs)
            current.wait_stream(capture.stream)
            self.capture_count += 1
        # Capture alone does not produce the real request's outputs. Always
        # replay once, including the first call for a newly captured bucket.
        entry.graph.replay()
        self.replay_count += 1
        return tuple(value[:count] for value in entry.outputs)
