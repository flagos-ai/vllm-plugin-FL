# SPDX-License-Identifier: Apache-2.0
"""Ascend adapters for MiniMax BF16 metadata, cache and dispatch."""

import inspect

import torch

from vllm import ir
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerTritonImpl,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseTritonImpl,
)

from vllm_fl.dispatch import resolve_op


def resolve_paged_cache_writer():
    from flaggems_vllm import paged_scatter_triton

    inspect.signature(paged_scatter_triton).bind(None, None, None, 1)
    return paged_scatter_triton


def gemma_norm(x, weight, eps):
    # Keep Gemma's 1+w and reduction in FP32 before the final output cast.
    return ir.ops.rms_norm(x, weight.float() + 1, eps)


def partial_neox_rope(x, positions, cos_sin_cache, rotary_dim):
    if rotary_dim % 2 or rotary_dim > x.shape[-1]:
        raise ValueError("Invalid MiniMax partial rotary dimension")
    cos, sin = cos_sin_cache[positions.long(), :rotary_dim].chunk(2, dim=-1)
    rotated = ApplyRotaryEmb.forward_static(
        x[..., :rotary_dim], cos, sin, enable_fp32_compute=True
    )
    return torch.cat((rotated, x[..., rotary_dim:]), -1)


def norm_rope(x, norm, positions, rotary_emb):
    shape = x.shape
    dtype = x.dtype
    x = gemma_norm(
        x.float().reshape(shape[0], -1, norm.weight.numel()),
        norm.weight,
        norm.variance_epsilon,
    )
    return (
        partial_neox_rope(x, positions, rotary_emb.cos_sin_cache, rotary_emb.rotary_dim)
        .to(dtype)
        .reshape(shape)
        .contiguous()
    )


def _update_graph_params(update_stream, forward_context, num_tokens, vllm_config):
    # Upstream builders update stable seq-len/block-table storage in place.
    pass


class AscendMiniMaxSparseImpl(MiniMaxM3SparseTritonImpl):
    update_graph_params = staticmethod(_update_graph_params)

    def forward(self, layer, query, kv_cache, output):
        from vllm.forward_context import get_forward_context

        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            output.zero_()
            return output
        md = metadata[layer.layer_name]
        nd, total = md.num_decode_tokens, md.num_actual_tokens
        q = query[:total].view(-1, self.num_heads, self.head_size)
        out = output[:total].view(-1, self.num_heads, self.head_size)
        selected = layer.topk_indices_buffer
        sparse_attention = layer._fl_sparse_attention
        if md.num_decodes:
            d = md.decode
            batch = d.seq_lens.shape[0]
            cu = (
                torch.arange(batch + 1, device=q.device, dtype=torch.int32)
                * d.decode_query_len
            )
            prefix = (d.seq_lens - d.decode_query_len).clamp(min=0)
            sparse_attention(
                q[:nd],
                (kv_cache[:, 0], kv_cache[:, 1]),
                selected[:, :nd],
                d.block_table,
                cu,
                d.seq_lens,
                prefix,
                d.decode_query_len,
                self.num_kv_heads,
                self.scale,
                out[:nd],
            )
        if md.num_prefills:
            p = md.prefill
            sparse_attention(
                q[nd:],
                (kv_cache[:, 0], kv_cache[:, 1]),
                selected[:, nd:total],
                p.block_table,
                p.cu_seqlens_q,
                p.seq_lens,
                p.context_lens,
                p.max_query_len,
                self.num_kv_heads,
                self.scale,
                out[nd:],
            )
        output[total:].zero_()
        return output


class AscendMiniMaxIndexerImpl(MiniMaxM3IndexerTritonImpl):
    update_graph_params = staticmethod(_update_graph_params)

    def forward(self, index_query):
        from .minimax_m3_indexer import index_forward

        return index_forward(self, index_query)


class AscendMiniMaxSparseBackend(MiniMaxM3SparseBackend):
    @staticmethod
    def get_impl_cls():
        return AscendMiniMaxSparseImpl

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        if include_num_layers_dimension:
            raise NotImplementedError("MiniMax does not stack caches across layers")
        return (0, 1, 2, 3, 4)


class AscendMiniMaxIndexerBackend(MiniMaxM3IndexerBackend):
    @staticmethod
    def get_impl_cls():
        return AscendMiniMaxIndexerImpl


class MiniMaxM3Kernels:
    """One selected provider owns norm, cache layout and sparse execution."""

    cache_contract = "minimax_m3_block_major_bf16"
    sparse_backend = AscendMiniMaxSparseBackend
    indexer_backend = AscendMiniMaxIndexerBackend
    gemma_norm = staticmethod(gemma_norm)
    norm_rope = staticmethod(norm_rope)

    @staticmethod
    def prepare_sparse(layer):
        if layer.kv_cache_torch_dtype != torch.bfloat16:
            raise ValueError("MiniMax Ascend currently requires BF16 KV cache")
        if layer.indexer.index_cache.dtype != torch.bfloat16:
            raise ValueError("MiniMax Ascend currently requires BF16 index cache")
        if not isinstance(layer.impl, MiniMaxM3SparseTritonImpl):
            raise TypeError("MiniMax Ascend requires upstream sparse Triton")
        if not isinstance(layer.indexer.impl, MiniMaxM3IndexerTritonImpl):
            raise TypeError("MiniMax Ascend requires upstream indexer Triton")
        sparse_attention = resolve_op("minimax_m3_sparse_attention")
        paged_cache_writer = resolve_paged_cache_writer()
        layer._fl_sparse_attention = sparse_attention
        layer._fl_cache_writer = paged_cache_writer
        layer.impl.__class__ = AscendMiniMaxSparseImpl
        layer.indexer.impl.__class__ = AscendMiniMaxIndexerImpl
        layer.attn_backend = AscendMiniMaxSparseBackend
        layer.indexer.index_cache.backend_cls = AscendMiniMaxIndexerBackend

    @staticmethod
    def insert_caches(layer, key, value, index_key, main_slots, index_slots):
        cache = layer.kv_cache
        if cache.ndim != 5 or cache.shape[1] != 2 or not cache.is_contiguous():
            raise ValueError("Expected block-major [blocks,2,block,heads,dim] cache")
        pages, _, block_size, _, _ = cache.shape
        key_cache = cache[:, 0].view(pages, block_size, layer.kv_size)
        value_cache = cache[:, 1].view(pages, block_size, layer.kv_size)
        index_cache = layer.indexer.index_cache.kv_cache
        plans = (
            (key_cache, main_slots[: key.shape[0]], key.contiguous()),
            (value_cache, main_slots[: value.shape[0]], value.contiguous()),
            (
                index_cache,
                index_slots[: index_key.shape[0]],
                index_key.reshape(-1, layer.idx_head_dim).contiguous(),
            ),
        )
        # Restrict the public writer to disjoint contiguous rows so every
        # capability check can be completed before the first cache mutation.
        for target, slots, values in plans:
            if target.ndim != 3 or target.shape[1] <= 1:
                raise ValueError(
                    "Expected paged cache with block size greater than one"
                )
            pages, block, width = target.shape
            if (
                values.ndim != 2
                or values.shape != (slots.numel(), width)
                or slots.ndim != 1
            ):
                raise ValueError("Cache values and slots have incompatible shapes")
            if (
                target.device.type != "npu"
                or values.device != target.device
                or slots.device != target.device
                or target.layout != torch.strided
                or values.layout != torch.strided
                or slots.layout != torch.strided
                or target.dtype != torch.bfloat16
                or values.dtype != target.dtype
                or slots.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("Unsupported paged cache tensor contract")
            if pages * block > 2**31 - 1:
                raise ValueError("Paged cache exceeds the scatter slot limit")
            if slots.numel() and width:
                s0, s1, s2 = target.stride()
                if (
                    not pages
                    or any(
                        s <= 0
                        for s in target.stride() + values.stride() + slots.stride()
                    )
                    or s2 != 1
                    or s1 < width
                    or s0 < (block - 1) * s1 + width
                ):
                    raise ValueError("Unsupported paged cache layout")
                storage = target.untyped_storage()
                limit = (
                    storage.nbytes() // target.element_size() - target.storage_offset()
                )
                end = (pages - 1) * s0 + (block - 1) * s1 + width
                if (
                    end > limit
                    or end > 2**63 - 1
                    or any(
                        storage.data_ptr() == tensor.untyped_storage().data_ptr()
                        for tensor in (slots, values)
                    )
                ):
                    raise ValueError("Unsupported paged cache storage")
        for target, slots, values in plans:
            if not layer._fl_cache_writer(target, slots, values, target.shape[1]):
                raise RuntimeError("FlagGems-vLLM rejected paged cache layout")

    @staticmethod
    def attend(layer, query, index_query, output):
        # Run through the bound providers inside the captured attention region.
        layer.indexer(index_query)
        layer.impl.forward(layer, query, layer.kv_cache, output)
