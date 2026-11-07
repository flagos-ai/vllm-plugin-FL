# Copyright (c) 2026 BAAI. All rights reserved.
"""Do not keep inference tensors alive in generated FlagGems autotuners."""
from functools import wraps


def _release_benchmark_tensors(run):
    if getattr(run, '_iluvatar_release_benchmark_tensors', False):
        return run

    @wraps(run)
    def wrapped(self, *args, **kwargs):
        try:
            return run(self, *args, **kwargs)
        finally:
            # These are diagnostic replay arguments, not kernel/config caches.
            # Generated per-shape tuners otherwise retain whole activations.
            self._last_benchmark_args = ()
            self._last_benchmark_meta = {}
            self.nargs = None

    wrapped._iluvatar_release_benchmark_tensors = True
    return wrapped


def patch_autotune_memory() -> None:
    from flag_gems.utils.libentry import LibTuner

    LibTuner.run = _release_benchmark_tensors(LibTuner.run)
