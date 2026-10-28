# SPDX-License-Identifier: Apache-2.0
"""vLLM metadata adapter for the FlagGems INT8 sparse indexer."""

import torch
from torch import nn

from flag_gems.fused.int8_sparse_indexer import (
    int8_indexer_cache_store,
    int8_indexer_paged_logits,
)
from vllm.forward_context import get_forward_context


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
        lengths = lengths.contiguous()
        logits = int8_indexer_paged_logits(
            q,
            self.cache.kv_cache,
            weights,
            table,
            rows.contiguous(),
            lengths,
            width,
            page_size,
        )
        count = min(self.topk, width)
        if count:
            values, indices = flag_gems.topk(logits, count, dim=-1)
            output[:, :count] = torch.where(torch.isfinite(values), indices, -1).to(
                torch.int32
            )

    def forward(self, hidden_states, q_quant, k, weights):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return self.indices
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
            rows = torch.repeat_interleave(
                torch.arange(counts.shape[0], device=q_quant.device),
                counts,
                output_size=n,
            )
            starts = counts.cumsum(0) - counts
            steps = torch.arange(n, device=q_quant.device) - starts[rows]
            seq_lens = decode.seq_lens
            if seq_lens.ndim == 2 and seq_lens.shape[1] > 1:
                lengths = seq_lens[rows, steps]
            else:
                lengths = seq_lens.reshape(-1)[rows] - counts[rows] + steps + 1
            self._score(
                q_quant[:n],
                weights[:n],
                decode.block_table,
                rows,
                lengths,
                self.indices[:n, : self.topk],
            )
        return self.indices
