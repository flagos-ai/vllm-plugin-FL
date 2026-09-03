# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers (BF16 path)."""

import torch
from flag_gems.fused import bf16_paged_mqa_logits

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import has_deep_gemm
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

from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm_fl.dispatch import CachedOp

_cp_gather_indexer_k_bf16_cache = CachedOp("gather_bf16_kv_from_pages")
_bf16_mqa_logits = CachedOp("bf16_mqa_logits")
_top_k_per_row_prefill = CachedOp("top_k_per_row_prefill")
_pack_seq_triton = CachedOp("pack_seq_triton")
_top_k_per_row_decode = CachedOp("top_k_per_row_decode")
_unpack_seq_triton = CachedOp("unpack_seq_triton")

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
) -> tuple[tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype),) for
    the BF16 K-gather workspace. BF16 path only needs a contiguous bf16 buffer,
    no separate scale tensor."""
    return (
        ((total_seq_lens, head_dim), torch.bfloat16),
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


# Maximum number of heads per chunk for bf16_paged_mqa_logits.
# The DeepGEMM kernel only supports num_heads == 32 or 64.  When the actual
# head count exceeds this or is not one of those values, we shard the heads
# into chunks of at most this size and accumulate the logits.
_BF16_HEADS_PER_CHUNK = 32


def _bf16_paged_mqa_logits_head_sharded(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Head-sharded wrapper around bf16_paged_mqa_logits.

    The DeepGEMM bf16_paged_mqa_logits kernel requires num_heads to be
    exactly 32 or 64. When the model's head count doesn't match (e.g. with
    tensor parallelism) or when we want to reduce shared-memory pressure,
    this function splits the heads into chunks, calls the kernel per chunk,
    and sums the results.

    The logits computation is:
        logits[b, pos] = sum_h(relu(q[b, :, h, :] @ K[pos, :]) * w[b, h])
    which is linear in the head dimension, so chunking + summing is exact.

    Args:
        q: [B, next_n, H, D] bf16
        kv_cache: [num_blocks, block_size, 1, D] bf16
        weights: [B * next_n, H] float32
        context_lens: [B, next_n] int32
        block_table: [B, max_blocks] int32
        schedule_metadata: scheduling tensor from get_paged_mqa_logits_metadata
        max_model_len: max sequence length for output
        clean_logits: whether to clean unfilled positions to -inf

    Returns:
        logits: [B * next_n, max_model_len] float32
    """
    num_heads = q.shape[2]

    # If heads fit a supported kernel size directly, no sharding needed
    if num_heads in (32, 64):
        return bf16_paged_mqa_logits(
            q, kv_cache, weights, context_lens, block_table,
            schedule_metadata, max_model_len, clean_logits=clean_logits,
        )

    # Determine chunk size: pick the largest supported size that divides evenly,
    # or fall back to 32 (padding the last chunk if needed).
    if num_heads % 64 == 0:
        chunk_size = 64
    elif num_heads % 32 == 0:
        chunk_size = 32
    else:
        # num_heads not divisible by 32 — pad to next multiple of 32
        chunk_size = 32

    num_chunks = (num_heads + chunk_size - 1) // chunk_size
    logits_accum = None

    for i in range(num_chunks):
        h_start = i * chunk_size
        h_end = min(h_start + chunk_size, num_heads)
        actual_chunk_heads = h_end - h_start

        q_chunk = q[:, :, h_start:h_end, :].contiguous()
        w_chunk = weights[:, h_start:h_end].contiguous()

        # Pad to a supported size (32 or 64) if the chunk is smaller
        kernel_heads = actual_chunk_heads
        if actual_chunk_heads not in (32, 64):
            kernel_heads = 32 if actual_chunk_heads <= 32 else 64
            if actual_chunk_heads < kernel_heads:
                # Pad q and weights with zeros (zero-padded heads contribute
                # nothing to logits since relu(0 * K) * 0 = 0)
                pad_h = kernel_heads - actual_chunk_heads
                q_chunk = torch.nn.functional.pad(
                    q_chunk, (0, 0, 0, pad_h), value=0.0
                )
                w_chunk = torch.nn.functional.pad(
                    w_chunk, (0, pad_h), value=0.0
                )

        chunk_logits = bf16_paged_mqa_logits(
            q_chunk, kv_cache, w_chunk, context_lens, block_table,
            schedule_metadata, max_model_len, clean_logits=False,
        )

        if logits_accum is None:
            logits_accum = chunk_logits
        else:
            logits_accum += chunk_logits

    return logits_accum


def sparse_attn_indexer_bf16(
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
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        (values_spec,) = _gather_workspace_shapes(
            total_seq_lens, head_dim
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_bf16_fake(
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
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # For BF16 cache, insert K directly without quantization.
        # kv_cache: [num_blocks, block_size, head_dim], k: [num_tokens, head_dim]
        # Vectorized scatter instead of Python for-loop.
        block_size_val = kv_cache.shape[1]
        valid_mask = slot_mapping >= 0
        valid_slots = slot_mapping[valid_mask]
        valid_k = k[valid_mask]
        blk_indices = valid_slots // block_size_val
        pos_indices = valid_slots % block_size_val
        kv_cache[blk_indices, pos_indices] = valid_k

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffer once (will allocate on first use).
        workspace_manager = current_workspace_manager()
        (values_spec,) = _gather_workspace_shapes(
            total_seq_lens, head_dim
        )
        (kv_buf_full,) = workspace_manager.get_simultaneous(
            values_spec,
        )

        kv_cache_4d = kv_cache_as_quant_view(
            kv_cache, head_dim, use_fp4_cache=False
        )

        for chunk in prefill_metadata.chunks:
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            w_slice = weights[chunk.token_start : chunk.token_end]
            num_tokens = q_slice.shape[0]
            schedule_metadata = getattr(chunk, "schedule_metadata", None)
            query_token_to_req = getattr(chunk, "query_token_to_req", None)

            if (
                current_platform.is_cuda()
                and has_deep_gemm()
                and schedule_metadata is not None
                and query_token_to_req is not None
            ):
                # Paged path: directly compute logits on paged KV cache
                # without gather. ~3-11x faster than gather + non-paged.

                # Expand per-request block_table to per-token
                per_token_block_table = chunk.block_table[
                    query_token_to_req
                ]

                # Per-token context lens [num_tokens, 1] for DeepGEMM
                per_token_ctx_lens = (
                    chunk.cu_seqlen_ke - chunk.cu_seqlen_ks
                ).unsqueeze(-1)

                # q: [num_tokens, H, D] -> [num_tokens, 1, H, D]
                q_4d = q_slice.unsqueeze(1)

                logits = _bf16_paged_mqa_logits_head_sharded(
                    q_4d,
                    kv_cache_4d,
                    w_slice,
                    per_token_ctx_lens,
                    per_token_block_table,
                    schedule_metadata,
                    max_model_len=int(per_token_ctx_lens.max().item()),
                    clean_logits=False,
                )

                # Paged kernel outputs [num_tokens, max_model_len] with
                # logits from 0..ctx_len per row. topk_per_row_prefill
                # expects absolute cu_seqlen_ks/ke, so use 0-based range.
                num_rows = logits.shape[0]
                cu_ks_zero = torch.zeros_like(chunk.cu_seqlen_ks)
                cu_ke_local = per_token_ctx_lens.squeeze(-1)

                topk_indices = topk_indices_buffer[
                    chunk.token_start : chunk.token_end, :topk_tokens
                ]

                _top_k_per_row_prefill(
                    logits,
                    cu_ks_zero,
                    cu_ke_local,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                # Fallback: gather + non-paged kernel (for non-CUDA or
                # when schedule_metadata is unavailable)
                kv_buf = kv_buf_full[: chunk.total_seq_lens]

                if not chunk.skip_kv_gather:
                    _cp_gather_indexer_k_bf16_cache(
                        kv_cache,
                        chunk.block_table,
                        chunk.cu_seq_lens,
                        chunk.token_to_seq,
                        chunk.total_seq_lens,
                        dst=kv_buf,
                    )

                logits = _bf16_mqa_logits(
                    q_slice,
                    kv_buf,
                    w_slice,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
                num_rows = logits.shape[0]

                topk_indices = topk_indices_buffer[
                    chunk.token_start : chunk.token_end, :topk_tokens
                ]

                _top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache=False)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            if q_scale is not None:
                padded_q_quant_decode_tokens = _pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = _pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = _pack_seq_triton(
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
        # otherwise. deep_gemm bf16_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = padded_q_quant_decode_tokens
        logits = _bf16_paged_mqa_logits_head_sharded(
            padded_q_quant_cast,
            kv_cache,
            weights[:num_padded_tokens],
            seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
            clean_logits=False,
        )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        _top_k_per_row_decode(
            logits,
            next_n,
            seq_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = _unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_bf16_fake(
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
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer_bf16",
    op_func=sparse_attn_indexer_bf16,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_bf16_fake,
    dispatch_key=current_platform.dispatch_key,
)


class SparseAttnIndexerBF16(SparseAttnIndexer):
    """Sparse Attention Indexer for BF16 KV cache path.

    Inherits from the base SparseAttnIndexer and overrides forward via
    forward_oot to dispatch through the unified CachedOp mechanism,
    eliminating platform-specific branching.
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
    ):
        super().__init__(
            k_cache=k_cache,
            quant_block_size=quant_block_size,
            scale_fmt=scale_fmt,
            topk_tokens=topk_tokens,
            head_dim=head_dim,
            max_model_len=max_model_len,
            max_total_seq_len=max_total_seq_len,
            topk_indices_buffer=topk_indices_buffer,
            skip_k_cache_insert=skip_k_cache_insert,
            use_fp4_cache=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_oot(hidden_states, q_quant, k, weights)

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer_bf16(
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
        )
