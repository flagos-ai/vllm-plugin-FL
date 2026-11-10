# Copyright (c) 2026 BAAI. All rights reserved.

from functools import wraps

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _release_benchmark_tensors(run):
    if getattr(run, "_iluvatar_release_benchmark_tensors", False):
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


def _is_uva_available() -> bool:
    return False


def patch_v2_uva() -> None:
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", "") != "iluvatar":
        return

    from vllm.v1.worker.gpu import buffer_utils

    if buffer_utils.is_uva_available is _is_uva_available:
        return
    buffer_utils.is_uva_available = _is_uva_available
    logger.info("Iluvatar V2 buffers use explicit host-to-device copies instead of UVA")


def apply_worker_patches() -> None:
    """Apply patches that must execute inside every worker process."""
    if getattr(current_platform, "vendor_name", "") != "iluvatar":
        return
    from vllm_fl.dispatch.backends.vendor.iluvatar.moe_router import patch_moe_router

    patch_moe_router()
    from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
        patch_vision_precision,
    )

    patch_vision_precision()
    patch_autotune_memory()
    patch_v2_uva()
    from vllm_fl.dispatch.backends.vendor.iluvatar.mrope import patch_mrope

    patch_mrope()
    from vllm_fl.dispatch.backends.vendor.iluvatar.iluvatar import (
        patch_sampler_compile_for_iluvatar,
        patch_torch_inductor_for_iluvatar,
    )

    # Required worker patches must fail startup, not silently degrade execution.
    patch_torch_inductor_for_iluvatar(strict=True)
    patch_sampler_compile_for_iluvatar(strict=True)
