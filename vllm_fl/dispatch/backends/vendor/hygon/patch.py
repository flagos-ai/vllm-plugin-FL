# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon (HCU/DCU) patches for vLLM 0.32 compatibility."""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def apply_hygon_patches():
    """Apply Hygon patches that must run before model construction.

    Cached so the patches run once per process. ``lru_cache`` does not cache
    exceptions, so a failed call can be retried.
    """
    patch_accelerator_missing_attrs()
    patch_uva_unavailable()


def patch_uva_unavailable() -> bool:
    """Report UVA as unavailable so Model Runner V2 uses its copy fallback.

    vLLM 0.32 derives UVA support from pinned memory alone
    (``vllm.utils.platform_utils.is_uva_available``), but ``UvaBuffer`` also
    needs ``get_accelerator_view_from_cpu_tensor``, which only handles XPU and
    CUDA-alike platforms. HCU is not CUDA-alike and this DTK build ships no
    ``torch.ops._C.get_cuda_view_from_cpu_tensor``, so ``UvaBuffer`` aborts
    Model Runner V2 construction. Reporting no UVA makes ``StagedWriteTensor``
    and ``UvaBufferPool`` take upstream's explicit-copy ``NonUvaBuffer`` path.

    Remove once DTK provides an HCU ``get_cuda_view_from_cpu_tensor``.
    """
    try:
        import sys

        import vllm.utils.platform_utils as platform_utils

        if getattr(platform_utils, "_hcu_uva_patched", False):
            return True

        def _uva_unavailable() -> bool:
            return False

        platform_utils.is_uva_available = _uva_unavailable
        # ``from ... import is_uva_available`` copies the callable, so rebind it
        # in vLLM modules that were imported before this patch ran.
        for name, module in list(sys.modules.items()):
            if not name.startswith("vllm.") or module is None:
                continue
            if getattr(module, "is_uva_available", None) is not None:
                module.is_uva_available = _uva_unavailable

        platform_utils._hcu_uva_patched = True
        logger.info(
            "Reported UVA as unavailable for Hygon: HCU has no "
            "get_cuda_view_from_cpu_tensor, Model Runner V2 uses the "
            "explicit-copy buffer fallback"
        )
        return True
    except Exception as exc:  # pragma: no cover - never block registration
        logger.warning("Failed to disable UVA for Hygon: %s", exc)
        return False


def patch_accelerator_missing_attrs() -> bool:
    """Fill in ``torch.accelerator.get_memory_info``, which DTK's Torch 2.9 lacks.

    vLLM 0.32 targets Torch 2.13, whose ``torch.accelerator`` namespace exposes
    ``get_memory_info``; 0.32 calls it on the worker startup path
    (``vllm.utils.mem_utils.MemorySnapshot``) and whenever the KV-cache memory
    budget is computed. The FlagOS Hygon image pins ``torch 2.9.0+das``
    (DTK 26.04), whose ``torch.accelerator`` has no ``get_memory_info``, so it
    is bound to ``torch.cuda.mem_get_info`` (HCU devices go through the
    CUDA-shaped API).

    Nothing else is filled in: DTK already ships the other ``torch.accelerator``
    entry points 0.32 uses (``current_device_index`` / ``set_device_index`` /
    ``device_count``), and 0.32 guards its only ``empty_host_cache`` call site
    with ``try/except AttributeError``.

    Return True when the accelerator namespace is complete afterwards.
    """
    try:
        import torch

        if getattr(torch.accelerator, "_hcu_attrs_patched", False):
            return True

        # Only fill in the attribute if the installed Torch does not provide
        # it, so a newer DTK that ships the real implementation keeps it
        # instead of being shadowed by the CUDA-shaped fallback below.
        bindings = {}
        if not hasattr(torch.accelerator, "get_memory_info"):
            bindings["get_memory_info"] = (
                lambda device_index=None: torch.cuda.mem_get_info(device_index)
            )

        for name, fn in bindings.items():
            setattr(torch.accelerator, name, fn)

        torch.accelerator._hcu_attrs_patched = True
        logger.info(
            "Patched torch.accelerator APIs for Hygon DTK Torch: %s",
            sorted(bindings),
        )
        return True
    except Exception as exc:  # pragma: no cover - never block registration
        logger.warning("Failed to patch torch.accelerator attrs for Hygon: %s", exc)
        return False
