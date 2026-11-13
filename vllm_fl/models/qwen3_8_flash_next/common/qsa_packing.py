# SPDX-License-Identifier: Apache-2.0
"""Request-aligned QSA packing metadata; cache lifetime is one builder invocation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

import torch


def qsa_pack_row_groups(starts, seq_lens, num_tokens, alignment=4):
    """Group same-request queries without crossing an incomplete KV group.

    Tail columns2048..2050 contain raw tokens from one compression group.
    Chunk starts may be unaligned, so grouping every four query rows is unsafe.
    """
    groups = []
    for request, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        start, end = int(start), int(end)
        position = int(seq_lens[request]) - (end - start)
        first = start
        while first < end:
            take = min(alignment - position % alignment, end - first)
            group = list(range(first, first + take))
            groups.append(group + [-1] * (4 - take))
            first += take
            position += take
    for first in range(int(starts[-1]), num_tokens, 4):
        group = list(range(first, min(first + 4, num_tokens)))
        groups.append(group + [-1] * (4 - len(group)))
    return groups


_QSA_PACK_ROWS_CACHE: ContextVar[dict | None] = ContextVar(
    "qwen38_pack_rows_cache", default=None
)


@contextmanager
def qsa_metadata_build_scope():
    """Share immutable packing rows only within one metadata-build invocation."""
    token = _QSA_PACK_ROWS_CACHE.set({})
    try:
        yield
    finally:
        _QSA_PACK_ROWS_CACHE.reset(token)


def _cached_pack_rows(starts_cpu, seq_cpu, num_tokens, device):
    starts, lengths = starts_cpu.tolist(), seq_cpu.tolist()
    key = (tuple(starts), tuple(lengths), num_tokens, str(device))
    cache = _QSA_PACK_ROWS_CACHE.get()
    if cache is not None and key in cache:
        return cache[key]
    groups = qsa_pack_row_groups(starts, lengths, num_tokens)
    rows = None
    if groups:
        cpu_rows = torch.tensor(groups, dtype=torch.int32, pin_memory=True)
        rows = cpu_rows.to(device, non_blocking=True)
    if cache is not None:
        cache[key] = rows
    return rows


def _cached_indexer_pack_rows(starts_cpu, num_tokens, device, pack=16):
    """Group queries by request, with no GPU readback or position alignment requirement."""
    if starts_cpu.device.type != "cpu":
        raise ValueError("QSA query boundaries must already be on CPU")
    if starts_cpu.ndim != 1 or starts_cpu.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "QSA request boundaries must be a one-dimensional integer tensor"
        )
    if pack != 16:
        raise ValueError("QSA indexer packing requires groups of 16 rows")
    starts = tuple(int(x) for x in starts_cpu.tolist())
    if (
        not starts
        or starts[0] != 0
        or starts[-1] > num_tokens
        or any(a > b for a, b in zip(starts[:-1], starts[1:]))
    ):
        raise ValueError(
            "QSA request boundaries must be ordered within the token batch"
        )
    key = ("indexer", starts, num_tokens, str(device), pack)
    cache = _QSA_PACK_ROWS_CACHE.get()
    if cache is not None and key in cache:
        return cache[key]
    groups = []
    for start, end in zip(starts[:-1], starts[1:]):
        for first in range(start, end, pack):
            row = list(range(first, min(first + pack, end)))
            groups.append(row + [-1] * (pack - len(row)))
    for first in range(starts[-1], num_tokens, pack):
        row = list(range(first, min(first + pack, num_tokens)))
        groups.append(row + [-1] * (pack - len(row)))
    rows = None
    if groups:
        cpu_rows = torch.tensor(
            groups, dtype=torch.int32, pin_memory=device.type == "cuda"
        )
        rows = cpu_rows.to(device, non_blocking=True)
    if cache is not None:
        cache[key] = rows
    return rows
