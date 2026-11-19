# SPDX-License-Identifier: Apache-2.0
"""Token layout and causal metadata for collocated CP (also testable on CPU)."""

from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache
from typing import Callable

import torch


@dataclass(frozen=True)
class PagedKVPlan:
    """Compact only referenced native pages; do not touch unused cache blocks."""

    block_ids: tuple[int, ...]
    block_table: tuple[tuple[int, ...], ...]
    _tensor_cache: dict[tuple[object, ...], torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def _cached_tensor(self, name: str, values, device,
                       dtype: torch.dtype) -> torch.Tensor:
        key = (name, str(device))
        tensor = self._tensor_cache.get(key)
        if tensor is None:
            tensor = torch.tensor(values, dtype=dtype, device=device)
            self._tensor_cache[key] = tensor
        return tensor

    @classmethod
    def build(cls, block_table, seq_lens, block_size):
        if block_size <= 0 or len(block_table) != len(seq_lens):
            raise ValueError("Invalid KV page metadata")
        counts = [(int(n) + block_size - 1) // block_size for n in seq_lens]
        if any(n < 0 for n in seq_lens):
            raise ValueError("Negative KV length")
        pages, remap, table = [], {}, []
        width = max(counts, default=0)
        for row, count in zip(block_table, counts):
            if len(row) < count:
                raise ValueError("Block table is shorter than the KV sequence")
            indices = []
            for block in row[:count]:
                block = int(block)
                if block < 0:
                    raise ValueError("Negative active KV block ID")
                if block not in remap:
                    remap[block] = len(pages)
                    pages.append(block)
                indices.append(remap[block])
            table.append(tuple(indices + [0] * (width - count)))
        return cls(tuple(pages), tuple(table))

    def table(self, device):
        return self._cached_tensor("table", self.block_table, device,
                                   torch.int32)

    @classmethod
    @lru_cache(maxsize=8)
    def contiguous_prefill(cls, num_tokens, block_size):
        """Virtual pages for a complete, page-aligned single request."""
        return cls.contiguous_prefills((num_tokens,), block_size)

    @classmethod
    @lru_cache(maxsize=8)
    def contiguous_prefills(cls, seq_lens: tuple[int, ...], block_size,
                           allow_partial=False):
        """Virtual pages for complete requests with separate page ranges.

        Request i starts after all pages of requests before it. Padding table
        columns are ignored by attention via each virtual request's seq_len.
        The cache key includes every request boundary, not just total tokens.
        """
        if (block_size <= 0 or not seq_lens
                or any(n <= 0 or (not allow_partial and n % block_size)
                       for n in seq_lens)):
            raise ValueError("Full prefill must contain complete KV pages")
        counts = tuple((n + block_size - 1) // block_size for n in seq_lens)
        width = max(counts)
        offset, table = 0, []
        for count in counts:
            table.append(tuple(range(offset, offset + count))
                         + (0,) * (width - count))
            offset += count
        return cls(tuple(range(offset)), tuple(table))

    def prefill_kv(self, k, v, seq_lens, block_size):
        """Page each complete request independently; mask every padding row.

        Native cache writes use the original unpadded tensors. These virtual
        pages are only consumed by attention with the true per-request lengths.
        """
        shape = (-1, block_size, *k.shape[1:])
        if all(n % block_size == 0 for n in seq_lens):
            return k.view(shape), v.view(shape)
        key = ("prefill_sources", str(k.device))
        sources = self._tensor_cache.get(key)
        if sources is None:
            values, start = [], 0
            for length in seq_lens:
                count = (length + block_size - 1) // block_size * block_size
                values.extend(range(start, start + length))
                values.extend([-1] * (count - length))
                start += length
            sources = self._cached_tensor("prefill_sources", values, k.device,
                                          torch.long)
        if k.is_cuda:
            from .layout_kernels import pad_prefill_kv

            pk, pv = pad_prefill_kv(k, v, sources)
        else:
            valid = sources >= 0
            pk = k.index_select(0, sources.clamp_min(0))
            pv = v.index_select(0, sources.clamp_min(0))
            pk[~valid] = 0
            pv[~valid] = 0
        return pk.view(shape), pv.view(shape)

    def gather(self, cache, total_kv_heads, world_size, all_gather):
        if cache.ndim != 5 or cache.shape[0] != 2:
            raise ValueError("Expected native FlashAttention KV cache shape")
        local_heads = max(1, total_kv_heads // world_size)
        if (max(total_kv_heads, world_size) % min(total_kv_heads, world_size)
                or cache.shape[3] != local_heads):
            raise ValueError("KV cache does not match native TP head layout")
        blocks = self._cached_tensor("block_ids", self.block_ids, cache.device,
                                     torch.long)
        local = cache.index_select(1, blocks).contiguous()
        gathered = all_gather(local, dim=3)
        if world_size > total_kv_heads:
            # Native QKVParallelLinear replicates a head on consecutive ranks:
            # KV=2, TP=4 -> [h0, h0, h1, h1], not [h0, h1, h0, h1].
            replicas = world_size // total_kv_heads
            key = ("replica_heads", str(cache.device), world_size,
                   total_kv_heads)
            heads = self._tensor_cache.get(key)
            if heads is None:
                heads = torch.arange(0, world_size, replicas,
                                     device=cache.device)
                self._tensor_cache[key] = heads
            gathered = gathered.index_select(3, heads)
        return gathered


@dataclass(frozen=True)
class ZigZagPlan:
    """Two blocks per request per rank; no padding enters attention or GDN.

    A local block [a, b) in a request with an existing prefix of length p
    attends to p+b cached tokens. Bottom-right causal alignment then gives
    query row i exactly p+a+i+1 visible keys, including during chunked prefill.
    """

    world_size: int
    rank: int
    num_tokens: int
    padded_rows: int
    uniform_rows: bool
    rows: tuple[int, ...]
    restore_rows: tuple[int, ...]
    request_ids: tuple[int, ...]
    query_start_loc: tuple[int, ...]
    seq_lens: tuple[int, ...]
    single_full_prefill: bool = False
    full_prefill_lengths: tuple[int, ...] = ()
    _tensor_cache: dict[tuple[object, ...], torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def _cached_tensor(self, name: str, values, device,
                       dtype: torch.dtype) -> torch.Tensor:
        """Materialize a layout index once per device and reuse it by layer."""
        key = (name, str(device))
        tensor = self._tensor_cache.get(key)
        if tensor is None:
            tensor = torch.tensor(values, dtype=dtype, device=device)
            self._tensor_cache[key] = tensor
        return tensor

    def rows_tensor(self, device) -> torch.Tensor:
        return self._cached_tensor("rows", self.rows, device, torch.long)

    def restore_rows_tensor(self, device) -> torch.Tensor:
        return self._cached_tensor("restore_rows", self.restore_rows, device,
                                   torch.long)

    def request_ids_tensor(self, device) -> torch.Tensor:
        return self._cached_tensor("request_ids", self.request_ids, device,
                                   torch.long)

    def query_start_loc_tensor(self, device) -> torch.Tensor:
        return self._cached_tensor("query_start_loc", self.query_start_loc,
                                   device, torch.int32)

    def seq_lens_tensor(self, device) -> torch.Tensor:
        return self._cached_tensor("seq_lens", self.seq_lens, device,
                                   torch.int32)

    def rank_major_rows_tensor(self, plans: tuple["ZigZagPlan", ...],
                               device) -> torch.Tensor:
        """Return source rows in the order required by reduce-scatter packing."""
        cached = self._tensor_cache.get(("rank_major_rows", str(device)))
        if cached is not None:
            return cached
        values = tuple(row for plan in plans for row in plan.rows)
        return self._cached_tensor("rank_major_rows", values, device,
                                   torch.long)

    def rank_major_dest_tensor(self, plans: tuple["ZigZagPlan", ...],
                               device) -> torch.Tensor:
        """Return packed destinations for ragged rank-major rows."""
        cached = self._tensor_cache.get(("rank_major_dest", str(device)))
        if cached is not None:
            return cached
        values = tuple(
            owner * self.padded_rows + offset
            for owner, plan in enumerate(plans)
            for offset in range(len(plan.rows))
        )
        return self._cached_tensor("rank_major_dest", values, device,
                                   torch.long)

    def padded_rank_major_rows_tensor(self, plans: tuple["ZigZagPlan", ...],
                                     device) -> torch.Tensor:
        """Output-only permutation; -1 denotes a disposable padding row."""
        if self.uniform_rows:
            return self.rank_major_rows_tensor(plans, device)
        key = ("padded_rank_major_rows", str(device))
        cached = self._tensor_cache.get(key)
        if cached is not None:
            return cached
        values = tuple(row for plan in plans for row in
                       (*plan.rows, *([-1] * (self.padded_rows - len(plan.rows)))))
        return self._cached_tensor("padded_rank_major_rows", values, device,
                                   torch.long)

    @classmethod
    def build(cls, query_start_loc, seq_lens, world_size: int, rank: int):
        if not 0 <= rank < world_size:
            raise ValueError("Invalid CP world size / rank")
        return cls.build_all(query_start_loc, seq_lens, world_size)[rank]

    @classmethod
    def build_all(cls, query_start_loc, seq_lens, world_size: int):
        """Build the shared row permutation once for all owners."""
        starts = [int(x) for x in query_start_loc]
        lengths = [int(x) for x in seq_lens]
        if world_size < 1:
            raise ValueError("Invalid CP world size / rank")
        if not starts or starts[0] != 0 or len(starts) != len(lengths) + 1:
            raise ValueError("Invalid packed query boundaries")
        qlens = [b - a for a, b in zip(starts, starts[1:])]
        if any(q < 0 or s < q for q, s in zip(qlens, lengths)):
            raise ValueError("Query lengths must be nonnegative and <= KV lengths")

        rank_rows = []
        rank_metadata = []
        for owner in range(world_size):
            rows = []
            local_requests, local_starts, local_lens = [], [0], []
            for block in (owner, 2 * world_size - 1 - owner):
                for req, qlen in enumerate(qlens):
                    a = qlen * block // (2 * world_size)
                    b = qlen * (block + 1) // (2 * world_size)
                    if a == b:
                        continue
                    rows.extend(range(starts[req] + a, starts[req] + b))
                    local_requests.append(req)
                    local_starts.append(local_starts[-1] + b - a)
                    local_lens.append(lengths[req] - qlen + b)
            rank_rows.append(rows)
            rank_metadata.append((tuple(local_requests), tuple(local_starts),
                                  tuple(local_lens)))

        padded_rows = max(1, max(map(len, rank_rows)))
        restore = [0] * starts[-1]
        for owner, rows in enumerate(rank_rows):
            for i, row in enumerate(rows):
                restore[row] = owner * padded_rows + i
        uniform_rows = all(len(rows) == padded_rows for rows in rank_rows)
        restore = tuple(restore)
        return tuple(
            cls(world_size, rank, starts[-1], padded_rows, uniform_rows,
                tuple(rows), restore, *rank_metadata[rank],
                single_full_prefill=(len(lengths) == 1
                                     and lengths[0] == starts[-1]),
                full_prefill_lengths=(tuple(lengths) if qlens == lengths else ()))
            for rank, rows in enumerate(rank_rows)
        )

    @cached_property
    def _row_bounds(self):
        return (min(self.rows), max(self.rows)) if self.rows else (0, -1)

    @cached_property
    def _restore_max(self):
        return max(self.restore_rows, default=-1)

    def select(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        rows = self.rows_tensor(x.device)
        size = x.shape[dim]
        low, high = self._row_bounds
        if self.rows and (low < 0 or high >= size):
            raise RuntimeError(
                f"ZigZagPlan select out of bounds: dim={dim} size={size} "
                f"rows=({low}, {high}) "
                f"rank={self.rank} num_tokens={self.num_tokens}"
            )
        return x.index_select(dim, rows)

    def gather_rank_major(self, local: torch.Tensor,
                          all_gather: Callable) -> torch.Tensor:
        """Gather directly in collective order, including any padded rows."""
        if local.shape[0] != len(self.rows):
            raise ValueError("Local tensor does not match the CP token layout")
        if self.uniform_rows:
            # Single-request prefill is already evenly partitioned. Avoid a
            # device memset and copy before every sequence all-gather.
            gathered = all_gather(
                local if local.is_contiguous() else local.contiguous(), dim=0
            )
        else:
            padded = local.new_zeros((self.padded_rows, *local.shape[1:]))
            padded[:len(self.rows)].copy_(local)
            gathered = all_gather(padded, dim=0)
        return gathered

    def gather(self, local: torch.Tensor, all_gather: Callable) -> torch.Tensor:
        """Equal-size collective even for empty ranks and ragged requests."""
        gathered = self.gather_rank_major(local, all_gather)
        restore = self.restore_rows_tensor(local.device)
        if self._restore_max >= gathered.shape[0]:
            raise RuntimeError(
                f"ZigZagPlan gather out of bounds: gathered={gathered.shape[0]} "
                f"restore=({min(self.restore_rows)}, {max(self.restore_rows)})"
            )
        return gathered.index_select(0, restore)

    def gather_native_kv(self, k, v, head_start, num_heads, all_gather):
        """Fuse K/V packing, then restore only this rank's native KV heads."""
        if k.shape != v.shape or k.ndim != 3 or k.shape[0] != len(self.rows):
            raise ValueError("K/V tensors do not match the CP token layout")
        if k.device != v.device or k.dtype != v.dtype:
            raise ValueError("K/V tensors must have matching device and dtype")
        if not 0 <= head_start < head_start + num_heads <= k.shape[1]:
            raise ValueError("Invalid native KV head range")
        restore = self.restore_rows_tensor(k.device)
        if k.is_cuda:
            from .layout_kernels import pack_kv, restore_native_kv

            packed = pack_kv(k, v, self.padded_rows)
            gathered = all_gather(packed, dim=0)
            return restore_native_kv(gathered, restore, head_start, num_heads)
        packed = k.new_zeros((self.padded_rows, 2, *k.shape[1:]))
        packed[:len(self.rows), 0].copy_(k)
        packed[:len(self.rows), 1].copy_(v)
        gathered = all_gather(packed, dim=0)
        native = gathered.index_select(0, restore)[
            :, :, head_start:head_start + num_heads
        ]
        return native[:, 0].contiguous(), native[:, 1].contiguous()

    def metadata(self, original):
        """Build virtual requests for paged FlashAttention without changing KV slots."""
        device = original.query_start_loc.device
        requests = self.request_ids_tensor(original.block_table.device)
        return replace(
            original,
            num_actual_tokens=len(self.rows),
            query_start_loc=self.query_start_loc_tensor(device),
            seq_lens=self.seq_lens_tensor(device),
            max_query_len=max((b - a for a, b in zip(
                self.query_start_loc, self.query_start_loc[1:])), default=0),
            max_seq_len=max(self.seq_lens, default=0),
            block_table=original.block_table.index_select(0, requests),
            # Original metadata was scheduled for TP head counts / full rows.
            scheduler_metadata=None,
            prefix_scheduler_metadata=None,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
        )
