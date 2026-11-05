# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4 INT8 indexer metadata, bounded score dispatch and decode workspaces.

The model adapter owns request/token mapping and graph-local buffers. Scoring
and 32-bit TopK use FlagGems-vllm public APIs under the shared operator policy.
The native vLLM 0.28 split/chunk/decode helpers define request partitioning.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import flaggems_vllm as _hy4_fgv
import torch
from flaggems_vllm import int8_indexer_cache_store
from torch import nn

from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import indexer as native_indexer
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV32IndexerPrefillChunkMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

from ..indexer_policy import _score_capacity
from ..metadata import (
    CPUSequenceLengthsProof,
    PrefillChunkSequenceBounds,
    get_current_cpu_seq_lens_proof,
    validated_prefill_chunk_bound,
)
from vllm_fl.utils import use_flaggems_op

_ATTENTION_CONTEXT = ContextVar("hy4_score_width_attention_context", default=None)


@contextmanager
def score_width_attention_scope(context):
    """Limit PIECEWISE selection to this invocation of attention's eager body."""
    token = _ATTENTION_CONTEXT.set(context)
    try:
        yield
    finally:
        _ATTENTION_CONTEXT.reset(token)


def select_eager_score_capacity(
    indexer, context, q, weights, table, rows, lengths, output, width, page_size
):
    if not _score_context_supported(indexer, context):
        return None
    if q.ndim != 3:
        return None
    m = q.shape[0]
    if m not in (256, 1024, 2048):
        return None
    descriptor = context.batch_descriptor
    if (
        descriptor is None
        or descriptor.num_tokens != m
        or descriptor.num_reqs not in (None, 1)
        or descriptor.uniform
        or descriptor.has_lora
        or descriptor.num_active_loras
    ):
        return None
    metadata = context.attn_metadata
    meta = metadata.get(indexer.cache.prefix) if isinstance(metadata, dict) else None
    if (
        not isinstance(meta, DeepseekV32IndexerMetadata)
        or not m <= meta.max_seq_len <= 32768
        or meta.num_prefills != 1
        or meta.num_decodes != 0
        or meta.num_decode_tokens != 0
        or meta.num_prefill_tokens != m
        or meta.prefill is None
        or len(meta.prefill.chunks) != 1
        or tuple(meta.seq_lens.shape) != (1,)
        or tuple(meta.slot_mapping.shape) != (m,)
        or meta.slot_mapping.dtype != torch.int64
        or meta.slot_mapping.stride() != (1,)
    ):
        return None
    chunk = meta.prefill.chunks[0]
    if (
        not isinstance(chunk, DeepseekV32IndexerPrefillChunkMetadata)
        or chunk.num_reqs != 1
        or chunk.total_seq_lens != meta.max_seq_len
        or chunk.local_total_seq_lens != meta.max_seq_len
        or chunk.token_start != 0
        or chunk.token_end != m
        or chunk.block_table is not table
        or tuple(chunk.local_cu_seq_lens.shape) != (2,)
        or chunk.local_cu_seq_lens.dtype != torch.int32
        or tuple(chunk.cu_seqlen_ks.shape) != (m,)
        or tuple(chunk.cu_seqlen_ke.shape) != (m,)
        or chunk.cu_seqlen_ks.dtype != torch.int32
        or chunk.cu_seqlen_ke.dtype != torch.int32
    ):
        return None
    if not _score_layout_supported(
        indexer,
        q,
        weights,
        table,
        rows,
        lengths,
        output,
        width,
        page_size,
        chunk.num_reqs,
    ):
        return None
    if any(
        x.device != q.device
        for x in (
            meta.seq_lens,
            meta.slot_mapping,
            chunk.local_cu_seq_lens,
            chunk.cu_seqlen_ks,
            chunk.cu_seqlen_ke,
        )
    ):
        return None
    return _score_capacity(meta.max_seq_len, width, indexer.topk)


def select_mixed_score_capacity(
    indexer, context, q, weights, table, rows, lengths, output, width, page_size
):
    """Narrow proven prefill rows in an uncaptured attention invocation.

    Request count and query count are not algorithm qualification keys. A
    current immutable step proof bounds each request-local score row. FULL
    and decode rows retain the original provider/width.
    """
    if not _score_context_supported(indexer, context) or q.ndim != 3:
        return None
    m = q.shape[0]
    desc = context.batch_descriptor
    if (
        m <= 0
        or desc is None
        or desc.num_tokens < m
        or desc.has_lora
        or desc.num_active_loras
    ):
        return None
    metadata = context.attn_metadata
    meta = metadata.get(indexer.cache.prefix) if isinstance(metadata, dict) else None
    if (
        not isinstance(meta, DeepseekV32IndexerMetadata)
        or meta.num_decodes <= 0
        or meta.num_decode_tokens <= 0
        or meta.num_prefills <= 0
        or meta.num_prefill_tokens <= 0
        or meta.num_decode_tokens + meta.num_prefill_tokens > desc.num_tokens
    ):
        return None
    chunks = meta.prefill.chunks if meta.prefill is not None else ()
    matching = [c for c in chunks if c.block_table is table]
    decode = meta.decode
    if len(matching) != 1 or (decode is not None and decode.block_table is table):
        return None
    chunk = matching[0]
    current = get_current_cpu_seq_lens_proof()
    proof = chunk.cpu_seq_len_bound
    if (
        not isinstance(chunk, DeepseekV32IndexerPrefillChunkMetadata)
        or not isinstance(proof, PrefillChunkSequenceBounds)
        or current is None
        or proof.step_proof is not current
        or meta.cpu_seq_lens_proof_generation != current.generation
        or not meta.num_decodes
        <= proof.request_start
        < proof.request_stop
        <= meta.num_decodes + meta.num_prefills
        or chunk.num_reqs != len(proof.seq_lens)
        or chunk.num_reqs != proof.request_stop - proof.request_start
        or (chunk.token_start, chunk.token_end) != (proof.chunk_start, proof.chunk_end)
        or not meta.num_decode_tokens
        <= chunk.token_start
        < chunk.token_end
        <= meta.num_decode_tokens + meta.num_prefill_tokens
        or chunk.token_end - chunk.token_start != m
        or chunk.total_seq_lens != sum(proof.seq_lens)
        or chunk.local_total_seq_lens != chunk.total_seq_lens
        or not proof.seq_lens
        or max(proof.seq_lens) > 32768
    ):
        return None
    if not _score_layout_supported(
        indexer,
        q,
        weights,
        table,
        rows,
        lengths,
        output,
        width,
        page_size,
        chunk.num_reqs,
    ):
        return None
    if (
        tuple(chunk.local_cu_seq_lens.shape) != (chunk.num_reqs + 1,)
        or chunk.local_cu_seq_lens.dtype != torch.int32
        or chunk.local_cu_seq_lens.stride() != (1,)
        or tuple(chunk.cu_seqlen_ks.shape) != (m,)
        or tuple(chunk.cu_seqlen_ke.shape) != (m,)
        or chunk.cu_seqlen_ks.dtype != torch.int32
        or chunk.cu_seqlen_ke.dtype != torch.int32
        or chunk.cu_seqlen_ks.stride() != (1,)
        or chunk.cu_seqlen_ke.stride() != (1,)
        or any(
            x.device != q.device
            for x in (
                chunk.local_cu_seq_lens,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
        )
    ):
        return None
    return _score_capacity(max(proof.seq_lens), width, indexer.topk)


def _score_context_supported(indexer, context):
    mode = context.cudagraph_runtime_mode
    return (
        indexer._hy4_int8_empty_tile_enabled
        and indexer._hy4_int8_prefix_topk_enabled
        and mode in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE)
        and (mode != CUDAGraphMode.PIECEWISE or _ATTENTION_CONTEXT.get() is context)
        and context.ubatch_slices is None
        and context.dp_metadata is None
        and not torch.is_grad_enabled()
    )


def _score_layout_supported(
    indexer, q, weights, table, rows, lengths, output, width, page_size, num_reqs
):
    m = q.shape[0]
    cache = indexer.cache.kv_cache
    return not (
        tuple(q.shape) != (m, 32, 128)
        or q.dtype != torch.int8
        or not q.is_cuda
        or q.stride() != (4096, 128, 1)
        or tuple(weights.shape) != (m, 32)
        or weights.dtype != torch.float32
        or weights.stride() != (32, 1)
        or table.ndim != 2
        or table.shape[0] != num_reqs
        or table.dtype != torch.int32
        or table.stride(1) != 1
        or table.stride(0) < table.shape[1]
        or tuple(rows.shape) != (m,)
        or rows.dtype != torch.int64
        or rows.stride() != (1,)
        or tuple(lengths.shape) != (m,)
        or lengths.dtype != torch.int32
        or lengths.stride() != (1,)
        or tuple(output.shape) != (m, 2048)
        or output.dtype != torch.int32
        or output.stride() != (2048, 1)
        or cache.ndim != 3
        or tuple(cache.shape[1:]) != (64, 132)
        or cache.dtype != torch.uint8
        or cache.stride() != (8448, 132, 1)
        or width <= 16384
        or page_size != 64
        or table.shape[1] * page_size < width
        or indexer.topk != 2048
        or any(
            x.device != q.device for x in (weights, table, rows, lengths, output, cache)
        )
        or torch.cuda.is_current_stream_capturing()
    )


_FULL_WARMED_DESCRIPTORS = set()


def _full_decode_descriptor(counts, seq_lens, padded_tokens, device):
    if type(padded_tokens) is not int or not 0 <= padded_tokens <= 1024:
        return None
    if (
        not counts.is_cuda
        or not seq_lens.is_cuda
        or counts.device != device
        or seq_lens.device != device
    ):
        return None
    return (
        str(counts.device),
        tuple(counts.shape),
        tuple(counts.stride()),
        str(counts.dtype),
        tuple(seq_lens.shape),
        tuple(seq_lens.stride()),
        str(seq_lens.dtype),
        padded_tokens,
    )


def full_decode_metadata_available(counts, seq_lens, padded_tokens, device):
    """Whether native startup completed this exact metadata prewarm descriptor.

    This is a compile/launch capability, not a tensor-value or quality check.
    It does not retain input tensors or graph-owned output buffers. A stock
    native runner never publishes descriptors and retains its original chain.
    """
    key = _full_decode_descriptor(counts, seq_lens, padded_tokens, device)
    return key is not None and key in _FULL_WARMED_DESCRIPTORS


class DecodeMetadataWorkspace:
    def __init__(self, capacity, device):
        from .metadata_kernels import decode_indexer_metadata

        if type(capacity) is not int or capacity < 0:
            raise ValueError(
                "decode metadata workspace capacity must be a nonnegative integer"
            )
        self.dispatch = decode_indexer_metadata
        self.rows = torch.empty((capacity,), dtype=torch.int64, device=device)
        self.lengths_i64 = torch.empty((capacity,), dtype=torch.int64, device=device)
        self.lengths_i32 = torch.empty((capacity,), dtype=torch.int32, device=device)

    def prepare(self, counts, seq_lens, n):
        lengths = (
            self.lengths_i32
            if seq_lens.ndim == 2
            and seq_lens.shape[1] > 1
            and seq_lens.dtype == torch.int32
            else self.lengths_i64
        )
        return self.dispatch(counts, seq_lens, self.rows, lengths, n)


def make_decode_metadata_workspace(capacity, device):
    return DecodeMetadataWorkspace(capacity, device)


def prepare_full_decode_metadata(counts, seq_lens, padded_tokens, device):
    """Graph-local fixed-shape outputs; no state shared across graph instances.

    Warm this exact FULL branch outside capture first. CUDA graph allocation
    owns these intermediate buffers; replay reuses their captured addresses.
    The caller retains current GPU counts/seq inputs and all cache/slot masks.
    """
    workspace = DecodeMetadataWorkspace(padded_tokens, device)
    lengths = (
        workspace.lengths_i32
        if seq_lens.ndim == 2
        and seq_lens.shape[1] > 1
        and seq_lens.dtype == torch.int32
        else workspace.lengths_i64
    )
    return workspace.dispatch(
        counts, seq_lens, workspace.rows, lengths, padded_tokens, padded=True
    )


def warm_full_decode_metadata_before_capture(attn_metadata, device):
    """Warm exact FULL metadata descriptors before the native model capture.

    Native graph_capture's stream context is not an active CUDA capture yet.
    Use a separate stream, wait for metadata H2D/preparation, then finish every
    warm launch before returning to the unchanged native model wrapper.
    This function is startup-only; it is absent from model replay.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FULL metadata prewarm must precede CUDA capture")
    groups = [attn_metadata] if isinstance(attn_metadata, dict) else attn_metadata
    if not isinstance(groups, (list, tuple)) or any(
        not isinstance(group, dict) for group in groups
    ):
        raise TypeError("FULL metadata prewarm requires native per-layer metadata")
    unique, seen = [], set()
    for group in groups:
        for meta in group.values():
            if not isinstance(meta, DeepseekV32IndexerMetadata) or id(meta) in seen:
                continue
            seen.add(id(meta))
            if meta.num_prefills:
                raise ValueError("FULL INT8 metadata prewarm requires pure decode")
            if meta.num_decode_tokens:
                unique.append(meta)
    if not unique:
        raise ValueError("FULL INT8 metadata prewarm found no decode metadata")
    warmed = []
    prep_stream = torch.cuda.current_stream(device)
    warm_stream = torch.cuda.Stream(device=device)
    warm_stream.wait_stream(prep_stream)
    with torch.cuda.stream(warm_stream):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("FULL metadata warm stream must not be capturing")
        try:
            for meta in unique:
                prepare_full_decode_metadata(
                    meta.decode.decode_lens,
                    meta.decode.seq_lens,
                    meta.num_decode_tokens,
                    device,
                )
                key = _full_decode_descriptor(
                    meta.decode.decode_lens,
                    meta.decode.seq_lens,
                    meta.num_decode_tokens,
                    device,
                )
                if key is None:
                    raise ValueError("invalid FULL metadata warm descriptor")
                warmed.append(key)
        finally:
            # Match native warmup ordering, including auxiliary/autotuner work.
            # Preserve failures; never enter model capture after warm failure.
            torch.cuda.synchronize(device)
    _FULL_WARMED_DESCRIPTORS.update(warmed)
    return len(unique)


class HYV4Int8IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """Build INT8 metadata without requesting the native FP8 logits scheduler.

    Request partitioning, chunk construction and decode tensor preparation use
    stock helpers. The Hy4 contract is CP1 with uncompressed KV. Native module
    globals and native builder classes remain unchanged.
    """

    def __init__(
        self,
        kv_cache_spec,
        layer_names,
        vllm_config,
        device,
        *,
        block_table_width,
        **kwargs,
    ):
        parallel = vllm_config.parallel_config
        if (
            parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or kv_cache_spec.compress_ratio != 1
        ):
            raise NotImplementedError(
                "Hy4 INT8 metadata requires CP1 without KV compression"
            )
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            block_table_width=block_table_width,
            **kwargs,
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        from ..runtime import bind_common_runtime_metadata

        common = common_attn_metadata
        bind_common_runtime_metadata(common)
        query_cpu = common.query_start_loc_cpu
        if (
            query_cpu.device.type != "cpu"
            or query_cpu.ndim != 1
            or query_cpu.dtype not in (torch.int32, torch.int64)
            or query_cpu.shape[0] != common.num_reqs + 1
        ):
            raise ValueError("query starts must be a CPU integer vector")
        if int(query_cpu[0].item()) != 0:
            raise ValueError("query starts must begin at zero")
        query_lens_cpu = torch.diff(query_cpu)
        if (common.num_reqs and int(query_lens_cpu.min().item()) < 0) or int(
            query_lens_cpu.sum().item()
        ) > common.num_actual_tokens:
            raise ValueError("decode lengths exceed their token span")
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common,
                decode_threshold=self.decode_threshold,
                require_uniform=not (self.use_flattening or self.supports_varlen),
                treat_short_extends_as_decodes=True,
            )
        )
        assert num_decodes + num_prefills == common.num_reqs
        assert num_decode_tokens + num_prefill_tokens == common.num_actual_tokens
        decode_lens_cpu = None
        max_decode_len = 0
        if num_decodes:
            decode_lens_cpu = query_lens_cpu[:num_decodes]
            max_decode_len = int(decode_lens_cpu.max().item())
        proof = common.cpu_seq_lens_proof
        proof_live = (
            isinstance(proof, CPUSequenceLengthsProof)
            and proof is get_current_cpu_seq_lens_proof()
        )
        prefill_metadata = None
        if num_prefills:
            upper = common.seq_lens_cpu_upper_bound
            assert upper is not None
            query_lens = torch.diff(
                common.query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
            )
            specs = native_indexer.split_indexer_prefill_chunks(
                upper[num_decodes:],
                query_lens,
                self.max_prefill_buffer_size,
                envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024,
                request_offset=num_decodes,
            )
            chunks = []
            for req_slice, query_slice in specs:
                chunk = native_indexer.build_prefill_chunk_metadata(
                    req_slice.start,
                    req_slice.stop,
                    common.query_start_loc,
                    common.query_start_loc_cpu,
                    common.seq_lens,
                    common.seq_lens,
                    upper,
                    common.block_table_tensor,
                    1,
                    query_slice=query_slice,
                    skip_kv_gather=query_slice.start > 0,
                    dcp_rank=0,
                    dcp_world_size=1,
                    cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                )
                if chunk is None:
                    continue
                chunk.cpu_seq_len_bound = (
                    validated_prefill_chunk_bound(
                        common,
                        chunk,
                        req_slice.start,
                        req_slice.stop,
                        num_decodes,
                        self.decode_threshold,
                        1,
                        1,
                        1,
                    )
                    if proof_live
                    else None
                )
                chunks.append(chunk)
            prefill_metadata = native_indexer.DeepseekV32IndexerPrefillMetadata(chunks)

        decode_metadata = None
        if num_decodes:
            torch.diff(
                common.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            next_n = self.num_speculative_tokens + 1
            step_next_n_ok = (
                max_decode_len <= 1
                or native_indexer._supports_native_decode(max_decode_len)
            )
            use_native = (
                not (self.use_flattening or self.supports_varlen)
                and max_decode_len <= next_n
                and step_next_n_ok
                # Keep native FULL-graph views for uniform live requests and
                # zero-length padding. Real nonuniform rows use stock flattening.
                and max_decode_len > 0
                and bool(
                    torch.all(
                        (decode_lens_cpu == max_decode_len) | (decode_lens_cpu == 0)
                    ).item()
                )
                and num_decode_tokens <= num_decodes * max_decode_len
            )
            indices = (
                self._build_varlen_decode_indices(
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    num_decode_tokens=num_decode_tokens,
                )
                if self.supports_varlen
                else None
            )
            seq_lens, table, decode_lens, _, requires_padding = (
                self._prepare_decode_tensors(
                    seq_lens=common.seq_lens[:num_decodes],
                    block_table=common.block_table_tensor[:num_decodes],
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    query_start_loc=common.query_start_loc[:num_decodes],
                    num_decodes=num_decodes,
                    num_decode_tokens=num_decode_tokens,
                    use_native=use_native,
                    next_n=next_n,
                    max_decode_len=max_decode_len,
                )
            )
            if seq_lens.ndim == 1:
                seq_lens = seq_lens.unsqueeze(-1)
            decode_metadata = native_indexer.DeepSeekV32IndexerDecodeMetadata(
                block_table=table,
                seq_lens=seq_lens,
                decode_lens=decode_lens,
                requires_padding=requires_padding,
                schedule_metadata=self.scheduler_metadata_buffer,
                indices=indices,
                global_seq_lens=None,
            )
        metadata = native_indexer.DeepseekV32IndexerMetadata(
            seq_lens=common.seq_lens,
            max_seq_len=common.max_seq_len,
            slot_mapping=common.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )
        metadata.cpu_seq_lens_proof_generation = (
            proof.generation if proof_live else None
        )
        return metadata


class HYV4Int8IndexerBackend(DeepseekV32IndexerBackend):
    # The inherited name preserves native prefill-kernel warmup classification.
    @staticmethod
    def get_builder_cls() -> type[HYV4Int8IndexerMetadataBuilder]:
        return HYV4Int8IndexerMetadataBuilder


class HYV4Int8IndexerCache(DeepseekV32IndexerCache):
    def get_attn_backend(self) -> type[HYV4Int8IndexerBackend]:
        return HYV4Int8IndexerBackend


class HYV4Int8Indexer(nn.Module):
    def __init__(self, cache, topk, max_model_len, indices, parallel_config):
        super().__init__()
        if (
            parallel_config.decode_context_parallel_size != 1
            or parallel_config.prefill_context_parallel_size != 1
        ):
            raise ValueError(
                "HYV4 INT8 indexer currently supports TP/PP without context parallelism"
            )
        self.cache, self.topk = cache, topk
        self.max_model_len, self.indices = max_model_len, indices
        self.dense_mha_metadata_layer_name = ""
        self._configure_providers()

    def _configure_providers(self):
        # Resolve the current shared policy before any KV-cache write.
        empty_tile = use_flaggems_op(
            "int8_indexer_paged_logits_empty_tile", default=False
        )
        prefix_topk = use_flaggems_op("indexer_prefix_topk", default=False)
        sm90 = (
            (empty_tile or prefix_topk)
            and current_platform.is_cuda()
            and current_platform.is_device_capability(90)
        )
        self._hy4_int8_empty_tile_enabled = empty_tile and sm90
        self._hy4_int8_prefix_topk_enabled = prefix_topk and sm90
        self._hy4_int8_paged_logits = _hy4_fgv.int8_indexer_paged_logits
        if self._hy4_int8_empty_tile_enabled:
            self._hy4_int8_paged_logits = _hy4_fgv.int8_indexer_paged_logits_empty_tile
        if self._hy4_int8_prefix_topk_enabled:
            self._hy4_int8_prefix_topk = _hy4_fgv.indexer_prefix_topk
            self._hy4_supports_int8_prefix_topk = _hy4_fgv.supports_indexer_prefix_topk

    @property
    def topk_indices_buffer(self):
        """Expose the buffer name used by native MTP sharing."""
        return self.indices

    @topk_indices_buffer.setter
    def topk_indices_buffer(self, buffer):
        self.indices = buffer

    def _score(self, q, weights, table, rows, lengths, output):
        import flag_gems

        page_size = self.cache.kv_cache.shape[1]
        width = min(self.max_model_len, table.shape[1] * page_size)
        input_rows, input_lengths = rows, lengths
        lengths = lengths.contiguous()
        logits_kwargs = {}
        capacity = select_eager_score_capacity(
            self,
            get_forward_context(),
            q,
            weights,
            table,
            input_rows,
            input_lengths,
            output,
            width,
            page_size,
        )
        pure_capacity = capacity is not None
        if capacity is None:
            capacity = select_mixed_score_capacity(
                self,
                get_forward_context(),
                q,
                weights,
                table,
                input_rows,
                input_lengths,
                output,
                width,
                page_size,
            )
        score_width = width if capacity is None else capacity
        if pure_capacity:
            logits_kwargs = {"pure_prefill": True}
        logits = self._hy4_int8_paged_logits(
            q,
            self.cache.kv_cache,
            weights,
            table,
            rows.contiguous(),
            lengths,
            score_width,
            page_size,
            **logits_kwargs,
        )
        count = min(self.topk, width)
        if count:
            if self._hy4_int8_prefix_topk_enabled:
                topk_output = output[:, :count]
                if self._hy4_supports_int8_prefix_topk(
                    logits, lengths, count, topk_output
                ):
                    self._hy4_int8_prefix_topk(logits, lengths, count, topk_output)
                    return
            values, indices = flag_gems.topk(logits, count, dim=-1)
            output[:, :count] = torch.where(torch.isfinite(values), indices, -1).to(
                torch.int32
            )

    def forward(self, hidden_states, q_quant, k, weights):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return self.indices
        self._configure_providers()
        meta = metadata[self.cache.prefix]
        int8_indexer_cache_store(
            k, self.cache.kv_cache, meta.slot_mapping, self.cache.kv_cache.shape[1]
        )
        self.indices[: hidden_states.shape[0]].fill_(-1)
        if meta.num_prefills:
            for chunk in meta.prefill.chunks:
                start, end = chunk.token_start, chunk.token_end
                # cu_seqlen_ks points to each query's sequence in the flattened
                # prefill key workspace. We score directly from paged INT8 K.
                rows = torch.searchsorted(
                    chunk.local_cu_seq_lens[1:], chunk.cu_seqlen_ks, right=True
                )
                lengths = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks
                self._score(
                    q_quant[start:end],
                    weights[start:end],
                    chunk.block_table,
                    rows,
                    lengths,
                    self.indices[start:end, : self.topk],
                )
        if meta.num_decode_tokens:
            decode = meta.decode
            n = meta.num_decode_tokens
            counts = decode.decode_lens
            prepared_metadata = None
            if (
                get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
                and full_decode_metadata_available(
                    counts, decode.seq_lens, n, q_quant.device
                )
            ):
                if meta.num_prefills != 0 or hidden_states.shape[0] != n:
                    raise ValueError(
                        "FULL metadata requires a complete padded decode batch"
                    )
                prepared_metadata = prepare_full_decode_metadata(
                    counts, decode.seq_lens, n, q_quant.device
                )
            if prepared_metadata is None:
                # Fixed-size integer metadata also covers graph padding where
                # sum(counts) is smaller than the padded query tensor. Invalid
                # tail rows stay masked for the score kernel and cache slots.
                ends = counts.cumsum(0)
                token_ids = torch.arange(n, device=q_quant.device)
                rows = torch.searchsorted(ends, token_ids, right=True)
                live = rows < counts.shape[0]
                safe_rows = rows.clamp(max=counts.shape[0] - 1)
                starts = ends - counts
                steps = token_ids - starts[safe_rows]
                seq_lens = decode.seq_lens
                if seq_lens.ndim == 2 and seq_lens.shape[1] > 1:
                    safe_steps = steps.clamp(min=0, max=seq_lens.shape[1] - 1)
                    lengths = seq_lens[safe_rows, safe_steps]
                else:
                    lengths = (
                        seq_lens.reshape(-1)[safe_rows] - counts[safe_rows] + steps + 1
                    )
                rows = torch.where(live, rows, -1)
                lengths = torch.where(live, lengths, 0)
            else:
                rows, lengths = prepared_metadata
            self._score(
                q_quant[:n],
                weights[:n],
                decode.block_table,
                rows,
                lengths,
                self.indices[:n, : self.topk],
            )
        return self.indices
