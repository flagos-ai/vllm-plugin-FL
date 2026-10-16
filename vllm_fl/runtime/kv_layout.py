# SPDX-License-Identifier: Apache-2.0
"""Optional backend contract for caches whose physical pages differ from tokens."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalCacheLayout:
    logical_block_size: int
    storage_block_size: int
    kernel_block_size: int

    def __post_init__(self):
        if (
            min(
                self.logical_block_size, self.storage_block_size, self.kernel_block_size
            )
            <= 0
            or self.logical_block_size % self.storage_block_size
            or self.storage_block_size % self.kernel_block_size
        ):
            raise ValueError(f"Invalid physical cache layout: {self}")

    @property
    def pages_per_block(self):
        return self.storage_block_size // self.kernel_block_size

    @property
    def metadata_block_size(self):
        return (
            self.logical_block_size // self.storage_block_size * self.kernel_block_size
        )


def get_physical_cache_layout(backend, spec):
    """None preserves the framework's path for backends without this contract."""
    capability = getattr(backend, "get_physical_cache_layout", None)
    return capability(spec) if capability is not None else None
