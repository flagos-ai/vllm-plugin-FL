# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text runtime for pristine vLLM 0.24.

This combines vLLM 0.24's KDA recurrent layer, DeepSeek-V3.2 sparse MLA,
DeepSeek MoE, and mHC operators. The model-specific bounded KDA gate and the
kpool-compressed index/tail caches are kept plugin-owned.
"""


from collections.abc import Iterable
from types import MethodType

import torch
from einops import rearrange
from torch import nn

from vllm.platforms import current_platform
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.compilation.decorators import support_torch_compile
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MLAAttention,
    DeepseekV2Model,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm_fl.dispatch import CachedOp, prepare_cached_ops



# Sparse indexer implementation.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_fl.dispatch import CachedOp


_indexer_capabilities = CachedOp("sparse_indexer_capabilities")


class SparseIndexerOps:
    """Model-facing calls whose implementations are selected by dispatch."""

    @property
    def compressed_page_size(self):
        return _indexer_capabilities()["compressed_page_size"]

    @property
    def cache_dtype(self):
        return _indexer_capabilities()["cache_dtype"]

    @property
    def query_dtype(self):
        return _indexer_capabilities()["query_dtype"]

    validate_environment = staticmethod(CachedOp("sparse_indexer_validate_environment"))
    rotate_indexer_query = staticmethod(
        CachedOp("sparse_indexer_rotate_indexer_query")
    )
    prepare_query = staticmethod(CachedOp("sparse_indexer_prepare_query"))
    indexer_k_quant_and_cache = staticmethod(
        CachedOp("sparse_indexer_indexer_k_quant_and_cache")
    )
    gather_cache = staticmethod(CachedOp("sparse_indexer_gather_cache"))
    mqa_logits = staticmethod(CachedOp("sparse_indexer_mqa_logits"))
    paged_mqa_logits = staticmethod(CachedOp("sparse_indexer_paged_mqa_logits"))
    topk_prefill = staticmethod(CachedOp("top_k_per_row_prefill"))
    topk_decode = staticmethod(CachedOp("top_k_per_row_decode"))
    pack_seq = staticmethod(CachedOp("sparse_indexer_pack_seq"))
    unpack_seq = staticmethod(CachedOp("sparse_indexer_unpack_seq"))
    persist_prefill_tail = staticmethod(
        CachedOp("sparse_indexer_persist_prefill_tail")
    )
    kpool_compress_and_write_cache = staticmethod(
        CachedOp("sparse_indexer_kpool_compress_and_write_cache")
    )
    kpool_decode_update_and_maybe_write_cache_batched = staticmethod(
        CachedOp("sparse_indexer_kpool_decode_update_and_maybe_write_cache_batched")
    )
    expand_pools_to_tokens = staticmethod(
        CachedOp("sparse_indexer_expand_pools_to_tokens")
    )
    append_tail_to_topk = staticmethod(
        CachedOp("sparse_indexer_append_tail_to_topk")
    )


INDEXER_OPS = SparseIndexerOps()

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32

# kpool write helper: form pools from the current token batch and compress them
# into the index K cache via the fused Triton kernel.


def _kpool_compress_insert(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int,
    round_scale: bool,
) -> None:
    """Pool ``kpool`` consecutive tokens into one fp8 K and write at pool slots.

    ``slot_mapping`` is pool-granular (compress_ratio == kpool on the spec):
    only the *last* token of each complete pool carries a valid (>=0) slot;
    intra-pool tokens are -1. We locate every pool completion, gather its
    ``kpool`` tokens, and run the softmax-weighted-sum + Hadamard + fp8 + write
    kernel. Assumes pool-aligned chunk starts (same invariant as sglang).
    """
    valid = slot_mapping >= 0
    valid_pos = torch.nonzero(valid).squeeze(-1)  # positions of pool completions
    n_pools = valid_pos.numel()
    if n_pools == 0:
        return
    pool_starts = valid_pos - (kpool - 1)  # first token of each pool
    # Drop pools whose start falls before the batch (leading padding); their
    # gate/k data is undefined anyway.
    ok = pool_starts >= 0
    if not bool(ok.all()):
        valid_pos = valid_pos[ok]
        pool_starts = pool_starts[ok]
        n_pools = valid_pos.numel()
        if n_pools == 0:
            return
    offs = torch.arange(kpool, device=k.device)
    idx = pool_starts[:, None] + offs[None, :]  # [n_pools, kpool]
    slot_k = k[idx]  # [n_pools, kpool, head_dim]
    slot_score = gate_score[idx]  # [n_pools, kpool, head_dim]
    loc = slot_mapping[valid_pos].to(torch.int64)  # [n_pools] pool slots
    INDEXER_OPS.kpool_compress_and_write_cache(
        kv_cache,
        slot_k,
        slot_score,
        ape,
        loc,
        pool_size=kpool,
        head_dim=head_dim,
        round_scale=round_scale,
        write_cache=True,
        return_compressed=False,
    )


def _scatter_decode_tokens_by_request(
    tokens: torch.Tensor,
    decode_lens: torch.Tensor,
    num_requests: int,
    lmax: int,
    pad_value,
) -> torch.Tensor:
    """Group ``[N, ...]`` decode tokens into a padded ``[num_requests, lmax, ...]``
    layout: request ``r``'s tokens at row ``r`` in order; short requests padded.

    ``N == decode_lens.sum()``. Unlike ``pack_seq_triton`` this is dtype-agnostic
    (needed for the int32 slot/pos tensors) — it builds the request / intra-index
    on device and scatters. Used only for the rare non-uniform
    (``requires_padding``) decode batch; uniform batches use a zero-copy reshape.
    """
    device = tokens.device
    dl = decode_lens.to(torch.int64)
    req_id = torch.repeat_interleave(
        torch.arange(num_requests, device=device, dtype=torch.int64), dl
    )
    req_starts = torch.cumsum(
        torch.cat([torch.zeros(1, device=device, dtype=torch.int64), dl[:-1]]),
        dim=0,
    )
    # Broadcast the per-request start offsets to per-token (length N == dl.sum())
    # so each token's intra-request index subtracts its own request's start.
    starts = torch.repeat_interleave(req_starts, dl)
    intra = torch.arange(tokens.shape[0], device=device, dtype=torch.int64) - starts
    out = torch.full(
        (num_requests, lmax, *tokens.shape[1:]),
        pad_value,
        dtype=tokens.dtype,
        device=device,
    )
    out[req_id, intra] = tokens
    return out


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. BF16 path: (T, head_dim) bf16 values plus a
    (T, 1) float32 identity scale required by the FlagGems prefill-MQA API.
    MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 plus
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 1), torch.float32),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


@eager_break_during_capture
def _logical_tail_cache_view(
    tail_kv_cache: torch.Tensor,
    pool_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Return the logical tail ring from vLLM's page-size-padded storage.

    The GLM5 cache grouping pads every four-token tail page to the sibling
    compressed-index page size so both groups can use one physical block pool.
    vLLM therefore binds a larger third dimension (for example 768), while the
    tail kernels operate on the first ``pool_size`` slots only.  Narrowing is a
    zero-copy metadata view and is safe during graph capture and replay.
    """
    if (
        tail_kv_cache.ndim != 4
        or tail_kv_cache.shape[1] != 2
        or tail_kv_cache.shape[2] < pool_size
        or tail_kv_cache.shape[3] != head_dim
    ):
        raise ValueError(
            "Expected padded tail cache [pages, 2, >=pool, dim]; got "
            f"shape={tuple(tail_kv_cache.shape)}, pool={pool_size}, dim={head_dim}"
        )
    return tail_kv_cache[:, :, :pool_size, :]


@eager_break_during_capture
def sparse_attn_indexer_kpool(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    # kpool params (Plan-A: gate is consumed at write time and read back at
    # topk time to softmax-weight the pool).
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    # Paged tail cache (in-progress pool's raw K + gate score), replacing the
    # transient _DECODE_TAIL ring. tail_prefix resolves attn_metadata[tail_prefix]
    # for the tail group's token-granular slot_mapping. None on the dummy/profiling
    # path and when the tail cache is disabled.
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = INDEXER_OPS.query_dtype
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Sentinel allocation so the profiler's peak-memory measurement covers
        # the runtime logits tensor. The decode-path fp8_fp4_paged_mqa_logits
        # output is [B*next_n, max_model_len] float32 -- sized by max_model_len,
        # NOT bounded by the prefill chunk cap. This profiling branch returns
        # the fake before ever calling that kernel, so its output tensor is
        # invisible unless we size this sentinel to the real worst-case decode
        # batch; otherwise large max_model_len / max_num_batched_tokens OOMs at
        # warmup (the old fixed VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512MiB was
        # ~10x too small at max_model_len=1M / b8192).
        cfg = get_current_vllm_config_or_none()
        worst_decode_tokens = 0
        if cfg is not None:
            sched = cfg.scheduler_config
            num_spec = (
                cfg.speculative_config.num_speculative_tokens
                if cfg.speculative_config is not None
                else 0
            )
            worst_decode_tokens = min(
                sched.max_num_seqs * (num_spec + 1),
                sched.max_num_batched_tokens,
            )
        # float32 logits -> 4 bytes/element; uint8 sentinel so elems == bytes.
        decode_logits_elems = worst_decode_tokens * max_model_len * 4
        prefill_cap_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        max_logits_elems = max(decode_logits_elems, prefill_cap_elems)
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_kpool_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens
    if tail_kv_cache is not None:
        tail_kv_cache = _logical_tail_cache_view(
            tail_kv_cache, index_kpool, head_dim
        )

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        if index_kpool > 1 and gate_score is not None and compress_ape is not None:
            # kpool prefill write: pool kpool consecutive prefill tokens via
            # softmax(gate+ape)-weighted sum -> Hadamard -> fp8 -> pool slots.
            # Decode tokens (the first num_decode_tokens in the batch) cannot be
            # pooled here — their pool's earlier tokens are not in this batch —
            # so they are deferred to the tail-buffer kernel in has_decode.
            # compress_ratio == index_kpool makes slot_mapping pool-granular.
            n_prefill = num_tokens - num_decode_tokens
            if n_prefill > 0:
                # decode tokens are batched first; prefill tokens follow.
                prefill_slice = slice(num_decode_tokens, num_tokens)
                _kpool_compress_insert(
                    k[prefill_slice],
                    gate_score[prefill_slice],
                    compress_ape,
                    kv_cache,
                    slot_mapping[prefill_slice],
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
                # Persist the prefill tail (trailing incomplete pool's raw K +
                # gate score) into the paged tail cache, so the decode side can
                # compress the boundary pool correctly -- including across PD
                # transfer, where the connector ships this block. Each packed
                # request has its OWN unfinished pool; n_prefill % kpool loses
                # request boundaries and can skip every tail in a full batch.
                if tail_kv_cache is not None and tail_prefix is not None:
                    tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
                    if tail_meta is not None:
                        assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
                        INDEXER_OPS.persist_prefill_tail(
                            k[prefill_slice],
                            gate_score[prefill_slice],
                            tail_meta.slot_mapping[prefill_slice],
                            tail_kv_cache,
                            pool_size=index_kpool,
                        )
        else:
            # standard: per-token fp8 quant + scatter (all tokens).
            assert scale_fmt is not None
            INDEXER_OPS.indexer_k_quant_and_cache(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Short-sequence full-attention fast path (mirrors sglang
        # IndexerKPool._full_topk_for_short_sequence). When every prefill
        # request's full context is <= topk_tokens, sparse selection would
        # pick ALL pools anyway (topk_pool = topk_tokens // index_kpool >=
        # num_pools, plus the always-selected tail == every token), so running
        # the MQA-logits is pointless and (in this port) triggers OOBs. Skip
        # it and attend to every token causally instead. The index-K cache was
        # already written above; this only fills the topk buffer. Real
        # sparsity only kicks in for contexts > topk_tokens.
        n_prefill_sf = num_tokens - num_decode_tokens
        short_prefill = (
            n_prefill_sf > 0
            and positions is not None
            and int(positions[num_decode_tokens:num_tokens].max().item()) + 1
            <= topk_tokens
        )
        if short_prefill:
            # short_prefill is only True when positions is not None (above),
            # but narrow explicitly for the indexer below.
            assert positions is not None
            _arange = torch.arange(
                topk_indices_buffer.shape[1],
                device=topk_indices_buffer.device,
                dtype=torch.int32,
            )
            _pos = positions[num_decode_tokens:num_tokens].to(torch.int32)
            _buf = topk_indices_buffer[num_decode_tokens:num_tokens]
            _buf[:] = _arange[None, :]
            _buf[_arange[None, :] > _pos[:, None]] = -1

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks if not short_prefill else ():
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            # The PPU cache is BF16, but FlagGems fp8_fp4_mqa_logits still
            # multiplies by a per-row scale.  Use identity scales.  Initialize
            # every chunk, including skip_kv_gather, so reused workspace never
            # leaks stale scale values into long-context sparse prefill.
            if not use_fp4_cache:
                k_scale.fill_(1.0)
            if not chunk.skip_kv_gather:
                INDEXER_OPS.gather_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.squeeze(-1)
            logits = INDEXER_OPS.mqa_logits(
                (q_slice_cast, q_scale_slice),
                (k_quant_cast, k_scale_cast),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                clean_logits=False,
            )
            num_rows = logits.shape[0]

            # kpool: logits are pool-granular (compress_ratio == index_kpool),
            # so topk selects pools. We pick topk_tokens // kpool pools then
            # expand each pool back to its kpool constituent tokens.
            select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
            if index_kpool > 1:
                pool_topk = torch.empty(
                    (num_rows, select_k), dtype=torch.int32, device=logits.device
                )
                topk_dst = pool_topk
            else:
                topk_dst = topk_indices_buffer[
                    chunk.token_start : chunk.token_end, :topk_tokens
                ]

            INDEXER_OPS.topk_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_dst,
            )

            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
                valid = pool_ids >= 0
                expanded = INDEXER_OPS.expand_pools_to_tokens(
                    pool_ids, valid, topk_tokens, index_kpool
                )
                # Append the per-query causal tail: tokens beyond the last
                # complete pool within [0, pos]. seq_len_q = pos + 1; the
                # incomplete remainder (seq_len_q % kpool) is not pooled so it
                # is reachable only by direct inclusion.
                if positions is not None:
                    q_pos = positions[chunk.token_start : chunk.token_end].to(
                        torch.int32
                    )
                    q_seq = q_pos + 1
                    pool_lens = (q_seq // index_kpool).to(torch.int32)
                    expanded = INDEXER_OPS.append_tail_to_topk(
                        expanded, q_seq, pool_lens, index_kpool
                    )
                topk_indices_buffer[
                    chunk.token_start : chunk.token_end, : expanded.shape[-1]
                ] = expanded

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache_raw = kv_cache  # raw [num_blocks, block_size, head_dim+4] for writes
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)

        # kpool decode write (must precede the logits read). Append each decode
        # token's k/gate to its REQUEST's tail ring; when a pool fills
        # (pos % kpool == kpool-1) compress + write at the pool slot that
        # compress_ratio hands us via slot_mapping.
        #
        # Spec verify batches next_n (>1) tokens per request. The per-request
        # tail ring must accumulate a request's tokens IN POSITION ORDER, so we
        # group tokens by request ([num_requests, next_n, ...]) and run the
        # per-request kernel once per token-slot — sequential launches keep each
        # request's tokens ordered (token t stashes before token t+1 reads it
        # for pool completion). Mirrors sglang's _forward_cuda_target_verify
        # (per-request kpool write plan, seqlen_per_q = write_start + k + 1).
        # Plain decode (next_n == 1) collapses to a single launch.
        #
        # NOTE: positions must be TOKEN-granular (per-token position, not the
        # pool-granular decode_metadata.seq_lens which is divided by
        # compress_ratio). The kernel derives the pool phase and tail-ring index
        # from pos % kpool, so a pool-granular pos misaligns every pool; a
        # per-request pos under spec is also too short (B entries for B*next_n
        # tokens) and reads out of bounds.
        if (
            index_kpool > 1
            and gate_score is not None
            and compress_ape is not None
            and positions is not None
            and not skip_k_cache_insert
        ):
            num_requests = attn_metadata_narrowed.num_decodes
            # The indexer's flatten decode path rewrites decode_lens to all-1s
            # and reports requires_padding=False even for a variable MTP-verify
            # batch (e.g. one request verifies 3 tokens while the rest verify
            # 4). The logits read is fine with that, but the kpool WRITE must
            # group tokens by their original request. Uniformity and the scatter
            # lmax are precomputed on the host in build()
            # (decode_is_uniform / write_max_decode_len), so this branch needs
            # no runtime .item() -- a .item() under cudagraph capture forces a
            # host sync and invalidates the stream.
            per_req_lens = getattr(decode_metadata, "per_req_decode_lens", None)
            if per_req_lens is not None:
                use_uniform = getattr(
                    decode_metadata, "decode_is_uniform", True
                ) and num_decode_tokens == num_requests * getattr(
                    decode_metadata, "write_max_decode_len", 1
                )
                group_lens = per_req_lens
                lmax = getattr(decode_metadata, "write_max_decode_len", 1)
            else:
                # Legacy metadata without per-request lens: fall back to the
                # host-side requires_padding flag. CUDA-graph dummy metadata
                # in vLLM 0.24 does not carry the GLM-5.3-Flash extension fields,
                # so this path is reached while profiling/capturing. Keep the
                # fallback shape entirely host-static: reading decode_lens via
                # .item() would synchronize the capture stream and invalidate
                # the graph. For a uniform batch the exact width is N/B; for a
                # defensive non-uniform legacy batch, N is a safe upper bound.
                use_uniform = not decode_metadata.requires_padding
                group_lens = decode_metadata.decode_lens
                lmax = (
                    max(1, num_decode_tokens // max(1, num_requests))
                    if use_uniform
                    else max(1, num_decode_tokens)
                )
            if not use_uniform:
                # Non-uniform decode_lens (mixed plain-decode + spec-verify, or
                # a variable MTP-verify batch): scatter actual tokens into a
                # padded [B, lmax] layout. int32 tensors can't go through
                # pack_seq_triton (float/uint8 only).
                dec_k = _scatter_decode_tokens_by_request(
                    k[:num_decode_tokens], group_lens, num_requests, lmax, 0
                )
                dec_gate = _scatter_decode_tokens_by_request(
                    gate_score[:num_decode_tokens],
                    group_lens,
                    num_requests,
                    lmax,
                    0,
                )
                dec_slot = _scatter_decode_tokens_by_request(
                    slot_mapping[:num_decode_tokens],
                    group_lens,
                    num_requests,
                    lmax,
                    -1,
                )
                dec_pos = _scatter_decode_tokens_by_request(
                    positions[:num_decode_tokens].to(torch.int32),
                    group_lens,
                    num_requests,
                    lmax,
                    -1,
                )
            else:
                next_n = num_decode_tokens // num_requests
                shape2 = (num_requests, next_n)
                dec_k = k[:num_decode_tokens].view(*shape2, head_dim)
                dec_gate = gate_score[:num_decode_tokens].view(*shape2, head_dim)
                dec_slot = slot_mapping[:num_decode_tokens].view(shape2)
                dec_pos = positions[:num_decode_tokens].to(torch.int32).view(shape2)
            tail_meta = (
                attn_metadata.get(_resolve_layer_name(tail_prefix))
                if tail_prefix is not None
                else None
            )
            # Paged tail cache replaces the transient _DECODE_TAIL ring. Group
            # the tail group's token-granular slot_mapping per-request, mirroring
            # dec_slot / dec_pos, so the kernel gets each request's current-token
            # tail slot (block * kpool + pos % kpool).
            if tail_meta is not None:
                assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
            if tail_meta is None or tail_kv_cache is None:
                dec_tail_slot = None
            elif not use_uniform:
                dec_tail_slot = _scatter_decode_tokens_by_request(
                    tail_meta.slot_mapping[:num_decode_tokens],
                    group_lens,
                    num_requests,
                    lmax,
                    -1,
                )
            else:
                dec_tail_slot = tail_meta.slot_mapping[:num_decode_tokens].view(shape2)
            # The compress kernel writes the raw fp8 cache (not the quant view);
            # pass the underlying kv_cache, not kv_cache_quant_view.
            if dec_tail_slot is not None:
                # Single batched launch over [num_requests, next_n] replaces the
                # per-token sequential loop. The kernel iterates each request's
                # tokens in position order internally, preserving the
                # pool-completion read-after-stash dependency that the loop
                # provided. Inputs are already grouped per request (uniform:
                # view; non-uniform: _scatter_decode_tokens_by_request padded to
                # [B, lmax]) — no per-token .contiguous() copies needed.
                INDEXER_OPS.kpool_decode_update_and_maybe_write_cache_batched(
                    kv_cache_raw,
                    tail_kv_cache,
                    dec_tail_slot,
                    dec_k,
                    dec_gate,
                    compress_ape,
                    dec_slot,
                    dec_pos,
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = INDEXER_OPS.pack_seq(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = INDEXER_OPS.pack_seq(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = INDEXER_OPS.pack_seq(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        logits = INDEXER_OPS.paged_mqa_logits(
            (padded_q_quant_cast, padded_q_scale),
            kv_cache,
            weights[:num_padded_tokens],
            seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
            clean_logits=False,
        )
        num_rows = logits.shape[0]
        # kpool: logits are pool-granular -> select topk_tokens//kpool pools,
        # then expand each pool back to its kpool tokens.
        select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
        if index_kpool > 1:
            pool_topk = torch.empty(
                (num_rows, select_k), dtype=torch.int32, device=logits.device
            )
            topk_dst = pool_topk
        else:
            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        INDEXER_OPS.topk_decode(
            logits,
            seq_lens,
            topk_dst,
            next_n=next_n,
        )

        # Resolve to token-level indices in the output buffer.
        if index_kpool > 1:
            pool_ids = pool_topk.to(torch.int64)
            valid = pool_ids >= 0
            out = INDEXER_OPS.expand_pools_to_tokens(
                pool_ids, valid, topk_tokens, index_kpool
            )
            # Append the always-selected tail: the request's trailing incomplete
            # pool (seq_len % kpool tokens) is not in the index K cache, so it
            # can only be reached by direct inclusion.
            n = out.shape[0]
            # NOTE: decode_metadata.seq_lens is POOL-granular (divided by
            # compress_ratio in the indexer metadata builder) because it feeds
            # the paged-MQA logits. append_tail_to_topk / pool_lens need
            # TOKEN-granular seq_len, so recover it from the decode tokens'
            # positions (pos == seq_len - 1). Using the compressed seq_lens
            # here yields dec_seq=0 for seq_len<kpool -> empty topk -> the
            # sparse MLA attends to nothing -> decode degradation.
            if positions is not None:
                dec_seq = positions[:n].to(torch.int32) + 1
            else:
                dec_seq = decode_metadata.seq_lens[:n]
                if dec_seq.ndim == 2:
                    dec_seq = dec_seq[:, -1]
                dec_seq = dec_seq.to(torch.int32)
            pool_lens = (dec_seq // index_kpool).to(torch.int32)
            out = INDEXER_OPS.append_tail_to_topk(
                out, dec_seq, pool_lens, index_kpool
            )
        else:
            out = topk_dst

        if decode_metadata.requires_padding:
            # Drop padded query rows introduced by the next_n padding above.
            out = INDEXER_OPS.unpack_seq(
                out.reshape(batch_size, -1, out.shape[-1]), decode_lens
            )
        topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out

    return topk_indices_buffer


def sparse_attn_indexer_kpool_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer_kpool",
    op_func=sparse_attn_indexer_kpool,
    # The indexer writes the index-K cache in place (prefill k-cache insert +
    # kpool decode write), so kv_cache must be declared as mutated — otherwise
    # under full-graph compile dynamo assumes it is unchanged across the
    # indexer→MLA boundary and the MLA reads stale/misaligned KV. The paged tail
    # cache is likewise written in place (prefill tail scatter + decode stash).
    mutates_args=["topk_indices_buffer", "kv_cache", "tail_kv_cache"],
    fake_impl=sparse_attn_indexer_kpool_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        INDEXER_OPS.validate_environment()

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        # The registered operator now dispatches each kernel in the indexer
        # chain independently: NVIDIA keeps the original reference path;
        # PPU uses its registered kernels; other vendors need an implementation.
        return self.forward_cuda(
            hidden_states,
            q_quant,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer_kpool(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            self.tail_cache.kv_cache if self.tail_cache is not None else None,
            self.tail_cache.prefix if self.tail_cache is not None else None,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        raise NotImplementedError("GLM kpool indexer has no registered ROCm backend")

# KPool cache contracts.
"""vLLM 0.24 cache objects for the GLM-5.3-Flash kpool indexer."""


from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


@dataclass(frozen=True, kw_only=True)
class KpoolTailSpec(SlidingWindowSpec):
    """One per-request circular page containing raw K and gate-score tail."""

    def max_admission_blocks_per_request(
        self, max_num_batched_tokens: int, max_model_len: int
    ) -> int:
        del max_num_batched_tokens, max_model_len
        return 1

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(isinstance(spec, KpoolTailSpec) for spec in kv_cache_specs.values())

    @property
    def participates_in_prefix_caching(self) -> bool:
        return False


class KpoolTailManager(FullAttentionManager):
    """Reference no-hit/no-prune fixed-one-block manager for the tail ring."""

    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        del (
            block_hashes,
            max_length,
            block_pool,
            kv_cache_spec,
            drop_eagle_block,
            alignment_tokens,
            dcp_world_size,
            pcp_world_size,
        )
        return tuple([] for _ in kv_cache_group_ids)

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        del request, num_tokens, retention_interval

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        del running_request_id
        return 0

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        del num_computed_tokens
        return 0

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        del request_id, total_computed_tokens

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        del (
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        return max(1 - len(self.req_to_blocks.get(request_id, ())), 0)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        # The base implementation grows with sequence length. The tail instead
        # obtains one page on first admission and circularly reuses it forever.
        del num_tokens, num_tokens_main_model
        req_blocks = self.req_to_blocks[request_id]
        if req_blocks:
            return []
        new_blocks = self.block_pool.get_new_blocks(1)
        req_blocks.extend(new_blocks)
        return new_blocks


class KpoolTailMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.ALWAYS
    supports_update_block_table = False
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        # Storage-only tail: the common builder state is sufficient. In
        # particular, do not allocate the normal indexer's paged-MQA buffers,
        # whose kernel block must be 32/64 rather than kpool (4).
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        del vllm_config, kv_cache_spec
        return AttentionCGSupport.ALWAYS

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        del common_prefix_len, fast_build
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(common_attn_metadata)
        )
        return DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=None,
            decode=None,
        )


class KpoolTailBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "KPOOL_TAIL"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        assert num_kv_heads == 1 and head_size % 2 == 0
        return (num_blocks, 2, block_size, head_size // 2)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        del include_num_layers_dimension
        return (0, 1, 2, 3)

    @staticmethod
    def get_builder_cls():
        return KpoolTailMetadataBuilder


class Glm5IndexerBackend(DeepseekV32IndexerBackend):
    @classmethod
    def indexes_kv_by_block_stride(cls):
        return True


class Glm5NextIndexerCache(DeepseekV32IndexerCache):
    def get_attn_backend(self):
        return Glm5IndexerBackend

    def __init__(self, *, index_kpool: int, **kwargs) -> None:
        super().__init__(**kwargs)
        assert index_kpool > 1
        self.index_kpool = index_kpool

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        spec = super().get_kv_cache_spec(vllm_config)
        assert isinstance(spec, MLAAttentionSpec)
        spec = replace(spec, compress_ratio=self.index_kpool)
        storage_block_size = spec.block_size // self.index_kpool
        assert (
            spec.block_size % self.index_kpool == 0 and storage_block_size % 32 == 0
        ), (
            "GLM-5.3-Flash kpool requires logical block_size to be a multiple of "
            f"index_kpool*32 ({self.index_kpool * 32}); got {spec.block_size}."
        )
        return spec


class Glm5NextTailCache(DeepseekV32IndexerCache):
    def __init__(self, *, index_kpool: int, **kwargs) -> None:
        super().__init__(**kwargs)
        assert index_kpool > 1
        self.index_kpool = index_kpool

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        del vllm_config
        return KpoolTailSpec(
            block_size=self.index_kpool,
            num_kv_heads=1,
            head_size=2 * self.head_dim,
            head_size_v=0,
            dtype=torch.bfloat16,
            sliding_window=self.index_kpool,
            indexes_kv_by_block_stride=True,
        )

    def get_attn_backend(self):
        return KpoolTailBackend

logger = init_logger(__name__)

causal_conv1d_fn = CachedOp("causal_conv1d_fn")
causal_conv1d_update = CachedOp("causal_conv1d_update")
fused_recurrent_kda = CachedOp("fused_recurrent_kda")
chunk_kda_with_safe_gate = CachedOp("chunk_kda_with_safe_gate")
fused_safe_kda_gate = CachedOp("fused_safe_kda_gate")


class MHCPreOp(nn.Module):
    forward = staticmethod(CachedOp("mhc_pre_with_norm"))


class MHCPostOp(nn.Module):
    forward = staticmethod(CachedOp("mhc_post"))


class MHCFusedPostPreOp(nn.Module):
    forward = staticmethod(CachedOp("mhc_fused_post_pre_with_norm"))


class SiluAndMulWithClamp(nn.Module):
    def __init__(self, swiglu_limit, alpha=1.0, beta=0.0, **kwargs):
        super().__init__()
        self.limit, self.alpha, self.beta = swiglu_limit, alpha, beta
        self.op = CachedOp("silu_and_mul_with_clamp")

    def forward(self, x):
        return self.op(x, self.limit, self.alpha, self.beta)


def _hc_expand(x: torch.Tensor, streams: int) -> torch.Tensor:
    return x.unsqueeze(1).expand(-1, streams, -1).contiguous()


def _hc_contract(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1)


class Glm5NextMLP(nn.Module):
    """Reference GLM-5.3-Flash SwiGLU, including the trained clamp."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
        swiglu_limit: float | None = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        if swiglu_limit is None:
            raise ValueError("GLM-5.3-Flash requires a finite swiglu_limit")
        self.act_fn = SiluAndMulWithClamp(swiglu_limit=swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm5NextMoE(nn.Module):
    """Reference GLM-5.3-Flash router and clamped shared/routed experts."""

    def __init__(
        self,
        config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        apply_routed_scale_to_output: bool = False,
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.routed_scaling_factor = float(config.routed_scaling_factor)

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = int(config.n_routed_experts)
        self.n_shared_experts = int(config.n_shared_experts)
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu is supported."
            )

        router_dtype_name = getattr(config, "moe_router_dtype", "float32")
        router_dtype = getattr(torch, str(router_dtype_name))
        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=router_dtype,
            prefix=f"{prefix}.gate",
        )
        if getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None

        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        swiglu_limit = config.swiglu_limit
        if swiglu_limit is None:
            raise ValueError("GLM-5.3-Flash requires a finite swiglu_limit")
        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            self.shared_experts = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=(
                    config.moe_intermediate_size * config.n_shared_experts
                ),
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
                swiglu_limit=swiglu_limit,
            )

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=apply_routed_scale_to_output,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            router_logits_dtype=self.gate.out_dtype,
            swiglu_limit=swiglu_limit,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.is_sequence_parallel and not already_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        if self.is_sequence_parallel and not already_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )[:num_tokens]
        return final_hidden_states.view(num_tokens, hidden_dim)


class Glm5NextLinearAttention(KimiGatedDeltaNetAttention):
    """v0.24 KDA projections with the reference bounded gate on both paths."""

    def __init__(self, config, vllm_config, prefix: str = "") -> None:
        super().__init__(config, vllm_config, prefix)
        kda_config = config.linear_attn_config or {}
        self.kda_lower_bound = float(kda_config.get("gate_lower_bound", -5.0))

        # GLM-5.3-Flash checkpoints store A_log as [num_heads], whereas the v0.24
        # runtime parameter is [1, 1, local_heads, 1].
        old_loader = self.A_log.weight_loader

        def load_a_log(param, loaded_weight):
            if loaded_weight.ndim == 1:
                loaded_weight = loaded_weight.view(1, 1, -1, 1)
            return old_loader(param, loaded_weight)

        self.A_log.weight_loader = load_a_log

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        super().forward(hidden_states, positions, output)
        return output

    @eager_break_during_capture
    def _forward_reference_safe_gate(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            #     # V1 profile run
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = (
            attn_metadata_narrowed.non_spec_state_indices_tensor
        )  # noqa: E501
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        constant_caches = self.kv_cache

        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        (conv_state, recurrent_state) = constant_caches
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        conv_state_q, conv_state_k, conv_state_v = conv_state.chunk(3, dim=-2)

        q_conv_weights = self.q_conv1d.weight.view(
            self.q_conv1d.weight.size(0), self.q_conv1d.weight.size(2)
        )
        k_conv_weights = self.k_conv1d.weight.view(
            self.k_conv1d.weight.size(0), self.k_conv1d.weight.size(2)
        )
        v_conv_weights = self.v_conv1d.weight.view(
            self.v_conv1d.weight.size(0), self.v_conv1d.weight.size(2)
        )
        if attn_metadata_narrowed.num_prefills > 0:
            q_proj_states = q_proj_states.transpose(0, 1)
            k_proj_states = k_proj_states.transpose(0, 1)
            v_proj_states = v_proj_states.transpose(0, 1)
            q = causal_conv1d_fn(
                q_proj_states,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_states=conv_state_q,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k_proj_states,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_states=conv_state_k,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v_proj_states,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_states=conv_state_v,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
        else:
            assert non_spec_state_indices_tensor is not None
            decode_conv_indices = non_spec_state_indices_tensor[
                : attn_metadata_narrowed.num_actual_tokens
            ]
            q = causal_conv1d_update(
                q_proj_states,
                conv_state_q,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )
            k = causal_conv1d_update(
                k_proj_states,
                conv_state_k,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )
            v = causal_conv1d_update(
                v_proj_states,
                conv_state_v,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim), (q, k, v)
        )

        if attn_metadata_narrowed.num_prefills > 0:
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            zero_idx = non_spec_state_indices_tensor[~has_initial_state]
            recurrent_state[zero_idx] = 0
            initial_state = recurrent_state[non_spec_state_indices_tensor].contiguous()
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_kda_with_safe_gate(
                q=q,
                k=k,
                v=v,
                raw_g=g1,
                beta=beta,
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
                lower_bound=self.kda_lower_bound,
            )
            # Init cache
            recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state
        else:
            assert non_spec_query_start_loc is not None
            g1 = fused_safe_kda_gate(
                rearrange(g1, "1 n h d -> n (h d)"),
                self.A_log,
                self.head_dim,
                g_bias=self.dt_bias,
                lower_bound=self.kda_lower_bound,
            ).unsqueeze(0)
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g1,
                beta=beta,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc[
                    : attn_metadata_narrowed.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
            )
        core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
            0, :num_actual_tokens
        ]

    _forward = _forward_reference_safe_gate


def _indexer_forward_nope(
    self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
) -> torch.Tensor:
    """DSA indexer projection when the checkpoint has zero RoPE channels."""
    del rotary_emb
    q, _ = self.wq_b(qr)
    q = q.view(-1, self.n_head, self.head_dim)
    kw, _ = self.wk_weights_proj(hidden_states)
    k = self.k_norm(kw[:, : self.head_dim])
    weights = kw[:, self.head_dim :]

    # Kpool stores H(k); score in the same orthonormal basis, H(q).H(k).
    # Omitting this rotation silently corrupts sparse selection beyond top-k.
    q = INDEXER_OPS.rotate_indexer_query(q)
    q = q.view(-1, self.head_dim)
    q_fp8, q_scale = INDEXER_OPS.prepare_query(
        q,
        self.quant_block_size,
        output_dtype=INDEXER_OPS.query_dtype,
        column_major_scales=False,
        use_ue8m0=self.scale_fmt is not None,
    )
    q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
    q_scale = q_scale.view(-1, self.n_head, 1)
    weights = (
        weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
    ).squeeze(-1)
    gate_score = torch.nn.functional.linear(
        hidden_states, self.index_kpool_compress_gate
    )
    return self.indexer_op(
        hidden_states,
        q_fp8,
        k,
        weights,
        gate_score=gate_score,
        compress_ape=self.index_kpool_compress_ape,
        index_kpool=self.index_kpool,
        positions=positions,
    )


def _ensure_thead_runtime(vllm_config) -> None:
    """Configure shared prefill and GLM cache handling in each spawned worker."""
    if getattr(current_platform, "vendor_name", None) != "thead":
        return
    from vllm_fl.attention.mla_prefill import configure_mla_prefill
    from vllm_fl.patches.glm5next import install_glm5_runner_adapter

    from vllm_fl.patches.glm5next import install_glm5next_kpool

    install_glm5next_kpool()
    configure_mla_prefill(vllm_config)
    install_glm5_runner_adapter()


class Glm5NextMLAAttention(DeepseekV2MLAAttention):
    """DeepSeek sparse MLA with an explicit zero-RoPE fast path."""

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs["vllm_config"]
        _ensure_thead_runtime(vllm_config)
        config = kwargs["config"]
        cache_config = kwargs["cache_config"]
        super().__init__(*args, **kwargs)
        if self.indexer is not None and config.index_kpool_compress:
            indexer = self.indexer
            kpool = int(config.index_kpool)

            # Replace the stock per-token cache object registered by the base
            # constructor with the kpool-compressed cache at the same prefix.
            static_ctx = vllm_config.compilation_config.static_forward_context
            old_cache = indexer.k_cache
            assert static_ctx.get(old_cache.prefix) is old_cache
            del static_ctx[old_cache.prefix]

            indexer.index_kpool = kpool
            indexer.index_kpool_compress_ape = nn.Parameter(
                torch.zeros(kpool, indexer.head_dim, dtype=torch.float32)
            )
            indexer.index_kpool_compress_gate = nn.Parameter(
                torch.empty(
                    indexer.head_dim,
                    config.hidden_size,
                    dtype=torch.bfloat16,
                )
            )
            indexer.k_cache = Glm5NextIndexerCache(
                head_dim=indexer.head_dim,
                dtype=INDEXER_OPS.cache_dtype,
                prefix=old_cache.prefix,
                cache_config=cache_config,
                index_kpool=kpool,
            )
            indexer.tail_cache = Glm5NextTailCache(
                head_dim=indexer.head_dim,
                dtype=torch.bfloat16,
                prefix=f"{indexer.prefix}.tail_cache",
                cache_config=cache_config,
                index_kpool=kpool,
            )
            indexer.indexer_op = SparseAttnIndexerKpool(
                indexer.k_cache,
                indexer.quant_block_size,
                indexer.scale_fmt,
                indexer.topk_tokens,
                indexer.head_dim,
                indexer.max_model_len,
                indexer.max_total_seq_len,
                indexer.topk_indices_buffer,
                tail_cache=indexer.tail_cache,
            )

        if self.qk_rope_head_dim == 0:
            self.mla_attn.rotary_emb = None
            self.mla_attn.indexer_rope_emb = None
            if self.indexer is not None:
                self.indexer.forward = MethodType(_indexer_forward_nope, self.indexer)


class Glm5NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.layer_idx = layer_idx
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.mhc = bool(config.mhc)

        if config.is_kda_layer(layer_idx):
            self.self_attn = Glm5NextLinearAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = Glm5NextMLAAttention(
                vllm_config=vllm_config,
                config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                cache_config=cache_config,
                # This verified BF16 checkpoint keeps MLA projections in BF16.
                quant_config=None,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
            )

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = Glm5NextMoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                swiglu_limit=config.swiglu_limit,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.mhc:
            self.n = config.mhc_num_residual_streams
            d_model = self.n * config.hidden_size
            mix_hc = (2 + self.n) * self.n
            self.hc_eps = config.hc_eps
            self.mhc_sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.mhc_post_mult_value = config.mhc_post_mult_value

            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            self.mhc_pre_op = MHCPreOp()
            self.mhc_post_op = MHCPostOp()
            self.mhc_fused_post_pre_op = MHCFusedPostPreOp()

    def _attention(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self.self_attn, DeepseekV2MLAAttention):
            return self.self_attn(positions, hidden_states, None)
        return self.self_attn(hidden_states, positions)

    def _hc_pre(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm: RMSNorm,
    ):
        return self.mhc_pre_op(
            residual=residual,
            fn=fn,
            hc_scale=scale,
            hc_base=base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
            norm_weight=norm.weight.data,
            norm_eps=norm.variance_epsilon,
        )

    def _hc_fused_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm: RMSNorm,
    ):
        return self.mhc_fused_post_pre_op(
            x=x,
            residual=residual,
            post_layer_mix=post,
            comb_res_mix=comb,
            fn=fn,
            hc_scale=scale,
            hc_base=base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.mhc_post_mult_value,
            sinkhorn_repeat=self.mhc_sinkhorn_iterations,
            n_splits=1,
            tile_n=1,
            norm_weight=norm.weight.data,
            norm_eps=norm.variance_epsilon,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        post: torch.Tensor | None,
        comb: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if not self.mhc:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states = self._attention(positions, hidden_states)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual=residual
            )
            hidden_states = residual + self.mlp(hidden_states)
            return hidden_states, residual, None, None

        x = hidden_states
        if post is None:
            if self.layer_idx == 0:
                x = _hc_expand(x, self.n)
            residual = x
            post, comb, x = self._hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.input_layernorm,
            )
        else:
            assert residual is not None and comb is not None
            residual, post, comb, x = self._hc_fused_post_pre(
                x,
                residual,
                post,
                comb,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.input_layernorm,
            )

        x = self._attention(positions, x)
        assert residual is not None and post is not None and comb is not None
        residual, post, comb, x = self._hc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.post_attention_layernorm,
        )
        x = self.mlp(x)

        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.mhc_post_op(x, residual, post, comb)
            return _hc_contract(x), None, None, None
        return x, residual, post, comb


@support_torch_compile
class Glm5NextModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size

        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            # kpool expands 512 pool ids to 2048 tokens, appends <=3 tail
            # tokens, and sparse MLA requires a 128-column aligned width.
            ((config.index_topk + config.index_kpool - 1 + 127) // 128 * 128),
            dtype=torch.int32,
            device=current_platform.device_type,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return Glm5NextDecoderLayer(
                vllm_config,
                layer_idx,
                prefix,
                topk_indices_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.use_mha = False
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        world_size = get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            residual = post = comb = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            post = comb = None

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        return self.norm(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The v0.24 DeepSeek loader handles fused MLA/indexer projections,
        # dense SwiGLU stacking, expert tensors, and direct KDA/mHC parameters.
        return DeepseekV2Model.load_weights(self, weights)


class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_text_config
        self.model = Glm5NextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        # FULL decode tracing starts during memory profiling, before these
        # Python call sites necessarily receive an eager example input.
        prepare_cached_ops()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        config = vllm_config.model_config.hf_text_config
        speculative_config = vllm_config.speculative_config
        num_spec = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            config.linear_num_heads,
            config.linear_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
            ignore_unexpected_prefixes=["model.visual."],
        )
        loaded = loader.load_weights(weights)

        # AutoWeightsLoader accepts a checkpoint as soon as every *present*
        # key has a destination; it does not verify the inverse condition that
        # every runtime parameter received a checkpoint tensor.  That is too
        # weak for a newly adapted architecture: one missed packed projection
        # otherwise remains torch.empty() and only appears later as meaningless
        # logits.  Audit the text model at the innermost CausalLM boundary so
        # vision-only parameters and outer HF prefix mapping cannot obscure the
        # result.
        expected = {name for name, _ in self.named_parameters()}
        missing = sorted(expected - loaded)
        unexpected = sorted(loaded - expected)
        logger.info(
            "GLM-5.3-Flash strict text weight audit: loaded=%d expected=%d "
            "missing=%d unexpected=%d",
            len(loaded),
            len(expected),
            len(missing),
            len(unexpected),
        )
        if unexpected:
            logger.warning(
                "GLM-5.3-Flash weight audit returned unexpected names: %s",
                unexpected[:32],
            )
        if missing:
            raise RuntimeError(
                "GLM-5.3-Flash checkpoint did not initialize all text parameters; "
                f"first missing names: {missing[:64]}"
            )
        return loaded

# Conditional-generation wrapper required by the checkpoint.
"""GLM-5.3-Flash multimodal wrapper for pristine vLLM 0.24.

The language runtime remains plugin-owned while the vision tower and the
batch-level ViT data-parallel transport reuse vLLM's GLM-OCR/GLM-4V support.
"""

from typing import ClassVar, Literal, Mapping

from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
    Glm4vMultiModalProcessor,
    Glm4vProcessingInfo,
)
from vllm.model_executor.models.glm_ocr import (
    GlmOcrPatchMerger,
    GlmOcrVisionTransformer,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


class Glm5NextVisionPatchMerger(GlmOcrPatchMerger):
    pass


class Glm5NextVisionTransformer(GlmOcrVisionTransformer):
    """GLM-OCR tower with GLM-5.3-Flash's wider projection bottleneck."""

    def __init__(
        self,
        text_config,
        vision_config,
        norm_eps: float = 1e-5,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            text_config,
            vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.merger = Glm5NextVisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.projection_intermediate_size,
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.merger",
        )


class Glm5NextProcessingInfo(Glm4vProcessingInfo):
    """Build the checkpoint's custom image/video processor locally."""

    def get_hf_processor(self, **kwargs: object):
        del kwargs
        processor = getattr(self, "_glm5_hf_processor", None)
        if processor is None:
            from vllm_fl.transformers_utils.processors.glm5next import (
                Glm5NextProcessor,
            )

            processor = Glm5NextProcessor.from_pretrained(self.ctx.model_config.model)
            self._glm5_hf_processor = processor
        return processor

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        """Per-item multimodal budget for the MultiModalBudget startup check.

        Mirrors ``Glm4vProcessingInfo.get_mm_max_tokens_per_item`` but reads the
        video pixel budget from ``_smart_resize_max``: the GLM-5.3-Flash video
        processor normalizes ``size`` into a height/width form for the generic
        resize path and keeps the real pixel budget (``longest_edge``) in that
        attribute, so the stock ``size["longest_edge"]`` lookup would return
        None and crash with ``None // int`` during engine init.
        """
        result: dict[str, int] = {}

        if mm_counts.get("image", 0) > 0:
            result["image"] = self.get_max_image_tokens()

        if mm_counts.get("video", 0) > 0:
            video_processor = self.get_video_processor()
            max_pixels = video_processor._smart_resize_max

            vision_config = self.get_hf_config().vision_config
            temporal_patch_size = vision_config.temporal_patch_size
            patch_size = vision_config.patch_size
            merge_size = vision_config.spatial_merge_size

            max_vision_tokens = max_pixels // (
                temporal_patch_size * patch_size**2 * merge_size**2
            )

            # GLMGA supports up to 640 frames (max_frames).
            max_grid_t = 640 // temporal_patch_size

            tokenizer = self.get_tokenizer()
            max_ts_tokens = max(
                len(tokenizer.encode(f"{t:.1f} seconds", add_special_tokens=False))
                for t in range(min(max_grid_t, 300))
            )

            result["video"] = max_vision_tokens + max_grid_t * (2 + max_ts_tokens) + 2

        return result


@MULTIMODAL_REGISTRY.register_processor(
    Glm4vMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid
):
    """GLM-5.3-Flash VLM: ViT-DP plus TP language layers and EP experts."""

    has_inner_state: ClassVar[Literal[True]] = True
    is_hybrid: ClassVar[Literal[True]] = True
    supports_encoder_tp_data = True

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):

        return Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):

        return Glm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):

        return Glm5NextForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Bypass Glm4vForConditionalGeneration.__init__: its language-model
        # architecture selection does not know GLM-5.3-Flash.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionTransformer(
                config.text_config,
                config.vision_config,
                norm_eps=config.vision_config.rms_norm_eps,
                # The vision checkpoint is BF16 even when the language model
                # comes from the separately published FP8 directory.
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Glm5NextForCausalLM"],
            )

__all__ = [
    "Glm5NextForCausalLM",
    "Glm5NextForConditionalGeneration",
    "Glm5NextProcessingInfo",
    "Glm5NextVisionTransformer",
]
