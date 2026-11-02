# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM v0.24.0 sparse_attn_indexer.py; metadata orchestration only.
"""Portable FP8 sparse indexer with explicit providers and real CUDA identity."""

import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import (
    _gather_workspace_shapes,
    kv_cache_as_quant_view,
    sparse_attn_indexer_fake,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
_LAYER_OPERATIONS = {}


@eager_break_during_capture
def portable_sparse_attn_indexer(
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
) -> torch.Tensor:
    if use_fp4_cache:
        raise ValueError("Portable sparse indexer currently requires FP8 cache")
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)
    implementations = _LAYER_OPERATIONS[k_cache_prefix]
    if not isinstance(attn_metadata, dict):
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec, scales_spec, ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8)
        )
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )
        return sparse_attn_indexer_fake(
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
    assert q_scale is None, "q_scale must be None when use_fp4_cache=False"
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]
    if not skip_k_cache_insert:
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        implementations["indexer_k_quant_and_cache"](
            k, kv_cache, slot_mapping, quant_block_size, scale_fmt
        )
    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec, scales_spec
        )
        for chunk in prefill_metadata.chunks:
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
            if not chunk.skip_kv_gather:
                implementations["cp_gather_indexer_k_quant_cache"](
                    kv_cache, k_quant, k_scale, chunk.block_table, chunk.cu_seq_lens
                )
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            logits = implementations["fp8_fp4_mqa_logits"](
                (q_slice, None),
                (k_quant, k_scale_cast),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                clean_logits=False,
            )
            num_rows = logits.shape[0]
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            implementations["top_k_per_row_prefill"](
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
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            padded_q_quant_decode_tokens = pack_seq_triton(
                q_quant[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        logits = implementations["fp8_fp4_paged_mqa_logits"](
            (padded_q_quant_decode_tokens, None),
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
        implementations["top_k_per_row_decode"](
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
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )
    return topk_indices_buffer


direct_register_custom_op(
    op_name="portable_sparse_attn_indexer",
    op_func=portable_sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


class PortableSparseAttnIndexer(nn.Module):
    """Model-owned entry point; neither DeepGEMM nor native top-k is implied by CUDA."""

    def __init__(
        self,
        k_cache,
        quant_block_size,
        scale_fmt,
        topk_tokens,
        head_dim,
        max_model_len,
        max_total_seq_len,
        topk_indices_buffer,
        *,
        operations,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        _LAYER_OPERATIONS[k_cache.prefix] = operations

    def forward(self, hidden_states, q_quant, k, weights):
        return torch.ops.vllm.portable_sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_quant,
            None,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            False,
            False,
        )
