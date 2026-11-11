# SPDX-License-Identifier: Apache-2.0
"""MiniMax BF16 adapters over torch_npu and upstream sparse Triton kernels."""

import torch

from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerTritonImpl,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseTritonImpl,
)


def gemma_norm(x, weight, eps):
    # Keep Gemma's 1+w and reduction in FP32 before the final output cast.
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return (value * (weight.float() + 1)).to(x.dtype)


def partial_neox_rope(x, positions, cos_sin_cache, rotary_dim):
    if rotary_dim % 2 or rotary_dim > x.shape[-1]:
        raise ValueError("Invalid MiniMax partial rotary dimension")
    cs = cos_sin_cache[positions.long()].float()
    half = rotary_dim // 2
    cos, sin = cs[:, None, :half], cs[:, None, half:rotary_dim]
    left, right = x[..., :half].float(), x[..., half:rotary_dim].float()
    rotated = torch.cat((left * cos - right * sin, right * cos + left * sin), -1)
    return torch.cat((rotated.to(x.dtype), x[..., rotary_dim:]), -1)


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

        from .minimax_m3_attention import minimax_m3_sparse_attn

        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            output.zero_()
            return output
        md = metadata[layer.layer_name]
        nd, total = md.num_decode_tokens, md.num_actual_tokens
        q = query[:total].view(-1, self.num_heads, self.head_size)
        out = output[:total].view(-1, self.num_heads, self.head_size)
        selected = layer.topk_indices_buffer
        if md.num_decodes:
            d = md.decode
            batch = d.seq_lens.shape[0]
            cu = (
                torch.arange(batch + 1, device=q.device, dtype=torch.int32)
                * d.decode_query_len
            )
            prefix = (d.seq_lens - d.decode_query_len).clamp(min=0)
            minimax_m3_sparse_attn(
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
            # Empty padded requests have no softmax domain; zero only those
            # rows, keeping nonempty-request NaNs visible to correctness gates.
            valid = (
                (d.seq_lens > 0)[:, None].expand(batch, d.decode_query_len).reshape(-1)
            )
            out[:nd].copy_(torch.where(valid[:, None, None], out[:nd], 0))
        if md.num_prefills:
            p = md.prefill
            minimax_m3_sparse_attn(
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
        layer.impl.__class__ = AscendMiniMaxSparseImpl
        layer.indexer.impl.__class__ = AscendMiniMaxIndexerImpl
        layer.attn_backend = AscendMiniMaxSparseBackend
        layer.indexer.index_cache.backend_cls = AscendMiniMaxIndexerBackend

    @staticmethod
    def insert_caches(layer, key, value, index_key, main_slots, index_slots):
        from .scatter_nd_update_v2 import _launch, _plan

        cache = layer.kv_cache
        if cache.ndim != 5 or cache.shape[1] != 2 or not cache.is_contiguous():
            raise ValueError("Expected block-major [blocks,2,block,heads,dim] cache")
        pages, _, block_size, _, _ = cache.shape
        key_cache = cache[:, 0].view(pages, block_size, layer.kv_size)
        value_cache = cache[:, 1].view(pages, block_size, layer.kv_size)
        index_cache = layer.indexer.index_cache.kv_cache
        # Validate every layout before the first write. Reuse the existing
        # padding-safe writer byte-for-byte; no fallback after any mutation.
        plans = (
            _plan(key.contiguous(), main_slots[: key.shape[0]], key_cache),
            _plan(value.contiguous(), main_slots[: value.shape[0]], value_cache),
            _plan(
                index_key.reshape(-1, layer.idx_head_dim).contiguous(),
                index_slots[: index_key.shape[0]],
                index_cache,
            ),
        )
        for plan in plans:
            _launch(plan)

    @staticmethod
    def attend(layer, query, index_query, output):
        # Bypass upstream's eager_break wrapper: capture the Triton launches.
        layer.indexer(index_query)
        layer.impl.forward(layer, query, layer.kv_cache, output)
