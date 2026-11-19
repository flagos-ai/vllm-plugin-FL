# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-local RNG rollback for samples discarded during chunked prefill."""

from contextlib import contextmanager


@contextmanager
def preserve_discarded_generator_offsets(generators, discarded_indices):
    """Restore discarded rows on success or failure; valid rows advance normally."""
    saved = []
    for index in discarded_indices:
        generator = generators.get(int(index))
        if generator is None:
            continue
        if generator.device.type == "cpu":
            saved.append((generator.set_state, generator.get_state()))
        else:
            saved.append((generator.set_offset, generator.get_offset()))
    try:
        yield
    finally:
        for restore, state in saved:
            restore(state)
