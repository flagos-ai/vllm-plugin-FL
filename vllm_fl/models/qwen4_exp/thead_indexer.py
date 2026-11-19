# Copyright (c) 2026 BAAI. All rights reserved.
"""FlagGems-vllm QSA scoring and cache operations for the T-Head model."""

import torch
from vllm.models.qwen4_exp.common.qsa_cache import canonical_qsa_rope_positions
from vllm.models.qwen4_exp.nvidia.indexer_qsa import (
    QSAIndexer as VllmQSAIndexer,
)
from vllm.models.qwen4_exp.nvidia.indexer_qsa import apply_qsa_rope as _apply_mrope
from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_compress_groups_with_ratio
from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import expand_qsa_block_indices


def apply_qsa_rope(
    rotary_emb: torch.nn.Module, positions: torch.Tensor, tensor: torch.Tensor
) -> torch.Tensor:
    if positions.ndim == 2:
        return _apply_mrope(rotary_emb, positions, tensor)
    # The stock 1D helper calls the NVIDIA FlashAttention rotary extension.
    rotary_dim = rotary_emb.rotary_dim
    cos, sin = rotary_emb._match_cos_sin_cache_dtype(tensor)[positions].chunk(2, dim=-1)
    rotary = rotary_emb.apply_rotary_emb.forward_native(
        tensor[..., :rotary_dim], cos, sin
    )
    return torch.cat((rotary, tensor[..., rotary_dim:]), dim=-1)


class QSAIndexer(VllmQSAIndexer):
    def forward(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        from flaggems_vllm.ops.qwen4 import qsa_mqa_paged, qsa_store_cache_rows

        metadata = self._metadata()
        if metadata is None:
            return out.fill_(-1)
        raw, compressed = metadata
        tokens = raw.num_actual_tokens
        if tokens == 0:
            return out
        projected_q, raw_keys = projected_qk[:tokens].split(
            (self.index_n_heads * self.index_head_dim, self.index_head_dim), dim=-1
        )
        positions = positions[..., :tokens]
        q = self.q_layernorm.forward_native(
            projected_q.reshape(-1, self.index_head_dim)
        ).reshape(tokens, self.index_n_heads, self.index_head_dim)
        q = apply_qsa_rope(self.rotary_emb, positions, q)
        raw_keys = raw_keys.reshape(tokens, 1, self.index_head_dim)
        raw_key_cache = self.raw_key_cache.key_cache
        rope_cache = self.raw_key_cache.rope_position_cache
        if rope_cache is None:
            position_rows = raw.logical_positions.view(-1, 1, 1).expand(-1, 1, 3)
        else:
            position_rows = canonical_qsa_rope_positions(positions)

        # Retain the vLLM 0.32 compressor ring ABI. The v0.24 FlagGems-vllm
        # compressor reads a full history cache, which uses different addressing.
        pooled, first_positions = qsa_compress_groups_with_ratio(
            raw_keys,
            position_rows,
            raw_key_cache,
            raw.block_table,
            raw.token_to_req,
            raw.query_start_loc,
            raw.logical_positions,
            compressed.slot_mapping,
            self.compress_ratio,
            rope_cache,
        )
        keys = self.k_layernorm.forward_native(
            pooled.reshape(-1, self.index_head_dim)
        ).reshape(-1, 1, self.index_head_dim)
        if getattr(self.rotary_emb, "mrope_section", None):
            first_positions = first_positions.transpose(0, 1)
        else:
            first_positions = first_positions[:, 0]
        keys = apply_qsa_rope(self.rotary_emb, first_positions, keys)
        key_cache = self.compressed_key_cache.kv_cache
        qsa_store_cache_rows(key_cache, compressed.slot_mapping, keys)
        qsa_store_cache_rows(raw_key_cache, raw.slot_mapping, raw_keys)
        if rope_cache is not None:
            qsa_store_cache_rows(rope_cache, raw.slot_mapping, position_rows)

        if self.skip_topk:
            return out

        # Bound the score and stable sort scratch space for long prefills.
        columns = compressed.block_table.shape[1] * key_cache.shape[1]
        rows_per_chunk = max(1, (128 << 20) // max(columns * 32, 1))
        block_topk = self.token_topk // self.compress_ratio
        for start in range(0, tokens, rows_per_chunk):
            rows = slice(start, min(start + rows_per_chunk, tokens))
            scores, visible = qsa_mqa_paged(
                q[rows],
                key_cache,
                compressed.block_table,
                compressed.token_to_req[rows],
                compressed.logical_positions[rows],
                compressed.seq_lens,
                self.compress_ratio,
            )
            width = min(scores.shape[1], block_topk)
            ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)[
                :, :width
            ]
            ranked = ranked.masked_fill(ranked >= visible[:, None], columns)
            ranked = ranked.sort(dim=-1).values
            blocks = torch.full(
                (ranked.shape[0], block_topk), -1, dtype=torch.int32, device=q.device
            )
            blocks[:, :width] = ranked.masked_fill(ranked == columns, -1)
            expand_qsa_block_indices(
                blocks,
                compressed.logical_positions[rows],
                visible,
                self.compress_ratio,
                self.token_topk,
                out[rows],
            )
        return out
