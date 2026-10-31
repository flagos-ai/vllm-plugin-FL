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
