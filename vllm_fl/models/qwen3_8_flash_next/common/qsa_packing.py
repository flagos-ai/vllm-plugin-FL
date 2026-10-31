# SPDX-License-Identifier: Apache-2.0
"""Request-aligned QSA packing metadata; cache lifetime is one builder invocation."""

from __future__ import annotations


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
