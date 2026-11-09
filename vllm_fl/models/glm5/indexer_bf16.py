# SPDX-License-Identifier: Apache-2.0
"""BF16 indexer adapters composed from existing PyTorch and public library ops.

K-pool math, page/slot addressing and tail ordering match the FP8 path. Only
quantization is omitted. No device kernels or checkpoint conversions live here.
"""

import torch


class Bf16Indexer:
    def __init__(self, backend):
        self.backend = backend

    def query(self, q):
        values = self.backend._kpool("hadamard128", q)
        return values, q.new_ones((q.shape[0], 1), dtype=torch.float32)

    def store(self, cache, rows, slots):
        self.backend._kpool(
            "qsa_store_cache_rows", cache.unsqueeze(2), slots.reshape(-1), rows
        )

    def insert(self, k, cache, slots, quant_block_size, scale_fmt):
        del quant_block_size, scale_fmt
        self.store(cache, k.to(torch.bfloat16), slots)

    def compress(
        self,
        cache,
        keys,
        scores,
        ape,
        slots,
        pool_size,
        head_dim=128,
        write_mask=None,
        round_scale=True,
        return_compressed=False,
        write_cache=True,
    ):
        del round_scale
        assert keys.shape[1:] == (pool_size, head_dim)
        # Preserve the BF16 materialization before and after the rotation.
        gates = torch.softmax(scores.float() + ape.float(), dim=1)
        pooled = (keys.float() * gates).sum(dim=1).to(torch.bfloat16)
        rotated = self.backend._kpool("hadamard128", pooled)
        if write_cache:
            if write_mask is not None:
                slots = torch.where(write_mask, slots, -1)
            self.store(cache, rotated, slots)
        if return_compressed:
            return rotated, rotated.new_ones(rotated.shape[0], dtype=torch.float32)

    def decode(
        self,
        cache,
        tail,
        tail_slots,
        keys,
        scores,
        ape,
        slots,
        positions,
        pool_size,
        head_dim=128,
        round_scale=True,
    ):
        # Iterate the host-known next_n dimension in order: a later token must
        # see earlier tail writes when it completes a pool. No device sync.
        for step in range(keys.shape[1]):
            tail_slot = tail_slots[:, step]
            valid = (tail_slot >= 0) & (positions[:, step] >= 0)
            tail_slot = torch.where(valid, tail_slot, -1)
            self.store(tail[:, 0], keys[:, step], tail_slot)
            self.store(tail[:, 1], scores[:, step], tail_slot)
            blocks = (tail_slot.clamp_min(0) // pool_size).long()
            pool_keys = tail[:, 0].index_select(0, blocks)
            pool_scores = tail[:, 1].index_select(0, blocks)
            complete = valid & (positions[:, step] % pool_size == pool_size - 1)
            self.compress(
                cache,
                pool_keys,
                pool_scores,
                ape,
                slots[:, step],
                pool_size,
                head_dim,
                write_mask=complete,
                round_scale=round_scale,
            )

    def gather(self, cache, values, scales, block_table, cu_seq_lens):
        positions = torch.arange(values.shape[0], device=cache.device)
        requests = torch.searchsorted(cu_seq_lens[1:], positions, right=True)
        offsets = positions - cu_seq_lens[requests]
        pages = block_table[requests, offsets // cache.shape[1]].long()
        values.copy_(cache[pages, offsets % cache.shape[1]])
        # Retain the existing workspace ABI; BF16 values have unit scale.
        scales.view(torch.float32).fill_(1)

    def logits(self, q, kv, weights, starts, ends, clean_logits=True):
        query, query_scale = q
        keys, _ = kv
        assert query.dtype == keys.dtype == torch.bfloat16
        assert query_scale is None
        rows, heads, _ = query.shape
        columns = keys.shape[0]
        result = query.new_empty((rows, columns), dtype=torch.float32)
        # Bound intermediates independently of context length/head count. FP32
        # accumulation preserves signed per-head weights after ReLU.
        for row in range(0, rows, 32):
            query_chunk = query[row : row + 32]
            weight_chunk = weights[row : row + 32].float().unsqueeze(-1)
            for col in range(0, columns, 512):
                # Existing BF16 x BF16 -> FP32 GEMM; its public router name
                # describes the usual caller, not a restriction on the matrix.
                dots = self.backend._kpool(
                    "router_gemm",
                    query_chunk.reshape(-1, query.shape[-1]),
                    keys[col : col + 512],
                ).view(query_chunk.shape[0], heads, -1)
                result[row : row + 32, col : col + 512] = (
                    dots.relu() * weight_chunk
                ).sum(dim=1)
        if clean_logits:
            offsets = torch.arange(columns, device=query.device)
            result.masked_fill_(
                (offsets[None] < starts[:, None]) | (offsets[None] >= ends[:, None]),
                -torch.inf,
            )
        return result

    def paged_logits(
        self,
        q,
        cache,
        weights,
        lengths,
        block_table,
        schedule_metadata,
        max_model_len,
        clean_logits=False,
    ):
        query, query_scale = q
        assert query_scale is None
        batch, next_n, heads, dim = query.shape
        cache = cache.squeeze(2)
        del schedule_metadata
        # Reuse BF16 GEMM/ReLU/reduction for arbitrary physical page strides and
        # head counts; the specialized BF16 API only handles dense 64-entry pages
        # with head count 32/64 and does not accept physical cache strides.
        result = query.new_empty((batch * next_n, max_model_len), dtype=torch.float32)
        offsets = torch.arange(max_model_len, device=query.device)
        page_indices = offsets // cache.shape[1]
        for request in range(batch):
            visible = offsets < lengths[request].reshape(-1).max()
            in_table = page_indices < block_table.shape[1]
            pages = block_table[
                request, page_indices.clamp_max(block_table.shape[1] - 1)
            ]
            valid = visible & in_table & (pages >= 0) & (pages < cache.shape[0])
            # Padded page-table entries must not address out-of-bounds storage.
            safe_pages = torch.where(valid, pages, 0).long()
            keys = cache[safe_pages, offsets % cache.shape[1]]
            keys = torch.where(valid[:, None], keys, 0)
            start = request * next_n
            result[start : start + next_n] = self.logits(
                (query[request], None),
                (keys, None),
                weights[start : start + next_n],
                torch.zeros(next_n, device=query.device, dtype=torch.int32),
                lengths[request].reshape(-1),
                clean_logits=clean_logits,
            )
        return result
