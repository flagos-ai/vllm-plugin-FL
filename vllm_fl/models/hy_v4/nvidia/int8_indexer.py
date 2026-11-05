# SPDX-License-Identifier: Apache-2.0
"""vLLM metadata adapter for the FlagGems INT8 sparse indexer."""

import flaggems_vllm as _hy4_fgv
import torch
from flaggems_vllm import (
    int8_indexer_cache_store,
)
from torch import nn

from vllm.forward_context import get_forward_context

from vllm_fl.utils import use_flaggems_op


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
        # Resolve before cache writes; numerical execution errors propagate.
        self._hy4_int8_paged_logits_module = _hy4_fgv
        self._hy4_int8_paged_logits_symbol = "int8_indexer_paged_logits"
        empty_tile = use_flaggems_op(
            "int8_indexer_paged_logits_empty_tile", default=False
        )
        prefix_topk = use_flaggems_op("indexer_prefix_topk", default=False)
        sm90 = False
        if empty_tile or prefix_topk:
            from vllm.platforms import current_platform

            sm90 = current_platform.is_cuda() and current_platform.is_device_capability(
                90
            )
        self._hy4_int8_empty_tile_enabled = empty_tile and sm90
        if self._hy4_int8_empty_tile_enabled:
            self._hy4_int8_paged_logits_symbol = "int8_indexer_paged_logits_empty_tile"
        # Resolve the callable before this forward can mutate the KV cache.
        getattr(self._hy4_int8_paged_logits_module, self._hy4_int8_paged_logits_symbol)
        # Resolve existing operator policy before preparing this forward.
        self._hy4_int8_prefix_topk_enabled = prefix_topk and sm90
        self._hy4_int8_prefix_topk = None
        self._hy4_supports_int8_prefix_topk = None
        if self._hy4_int8_prefix_topk_enabled:
            from flaggems_vllm import (
                indexer_prefix_topk,
                supports_indexer_prefix_topk,
            )

            self._hy4_int8_prefix_topk = indexer_prefix_topk
            self._hy4_supports_int8_prefix_topk = supports_indexer_prefix_topk
        self._hy4_int8_bounded_score_enabled = False
        self._hy4_int8_bounded_score_metadata_types = None
        self._hy4_int8_cpu_bound_proof_type = None
        self._hy4_int8_current_cpu_proof = None
        if self._hy4_int8_empty_tile_enabled and self._hy4_int8_prefix_topk_enabled:
            from vllm.platforms import current_platform

            self._hy4_int8_bounded_score_enabled = (
                current_platform.is_device_capability(90)
            )
            if self._hy4_int8_bounded_score_enabled:
                from vllm.v1.attention.backends.mla.indexer import (
                    DeepseekV32IndexerMetadata,
                    DeepseekV32IndexerPrefillChunkMetadata,
                )

                self._hy4_int8_bounded_score_metadata_types = (
                    DeepseekV32IndexerMetadata,
                    DeepseekV32IndexerPrefillChunkMetadata,
                )
                from vllm.v1.attention import backend as attention_backend

                self._hy4_int8_cpu_bound_proof_type = getattr(
                    attention_backend, "PrefillChunkSequenceBounds", None
                )
                self._hy4_int8_current_cpu_proof = getattr(
                    attention_backend, "get_current_cpu_seq_lens_proof", None
                )

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
        from .score_width_dispatch import (
            select_eager_score_capacity,
            select_mixed_score_capacity,
        )

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
        logits = getattr(
            self._hy4_int8_paged_logits_module, self._hy4_int8_paged_logits_symbol
        )(
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
            from vllm.config import CUDAGraphMode

            if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL:
                from .decode_metadata import (
                    full_decode_metadata_available,
                    prepare_full_decode_metadata,
                )

                if full_decode_metadata_available(
                    counts, decode.seq_lens, n, q_quant.device
                ):
                    if meta.num_prefills != 0 or hidden_states.shape[0] != n:
                        raise ValueError(
                            "FULL metadata requires a complete padded decode batch"
                        )
                    prepared_metadata = prepare_full_decode_metadata(
                        counts, decode.seq_lens, n, q_quant.device
                    )
            if prepared_metadata is None:
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
