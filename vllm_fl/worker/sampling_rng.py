# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-local RNG rollback for sampled tokens discarded during prefill."""

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Protocol


class OffsetGenerator(Protocol):
    def get_offset(self) -> int: ...

    def set_offset(self, offset: int) -> None: ...


@contextmanager
def preserve_discarded_generator_offsets(
    generators: Mapping[int, OffsetGenerator], discarded_indices: Iterable[int]
) -> Iterator[None]:
    """Restore only discarded requests, independent of sampler RNG consumption.

    Keep original generator objects and entry offsets in this call's scope.
    Valid requests and requests using the global generator are left untouched.
    Sampler failures propagate after restoring discarded-request offsets.
    """
    saved = []
    for index in discarded_indices:
        generator = generators.get(int(index))
        if generator is not None:
            saved.append((generator, generator.get_offset()))
    try:
        yield
    finally:
        for generator, offset in saved:
            generator.set_offset(offset)
