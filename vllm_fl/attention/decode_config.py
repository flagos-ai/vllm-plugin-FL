"""Static split-decode kernel configuration shared by attention backends."""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class DecodeConfig:
    """Kernel launch parameters and measured (batch size, split count) pairs."""

    variant: int
    tile: int
    num_warps: int
    num_stages: int
    table: tuple[tuple[int, int], ...]

    def lookup(self, num_seqs: int) -> int:
        """Use the closest profiled batch size on a logarithmic scale."""
        log_batch = math.log2(max(num_seqs, 1))
        _, splits = min(
            self.table, key=lambda entry: abs(math.log2(entry[0]) - log_batch)
        )
        return splits
