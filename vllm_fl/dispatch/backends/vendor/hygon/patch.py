# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon (HCU/DCU) patches for vLLM 0.32 compatibility."""

import logging
from collections.abc import Callable
from functools import lru_cache

logger = logging.getLogger(__name__)

_MISSING = object()


def _noop() -> None:
    """Rollback for a patch that found its work already done."""


def _snapshot(owner, name: str):
    """Record the value of ``owner.name`` before this patch changes it."""
    return getattr(owner, name, _MISSING)


def _apply(owner, name: str, value, applied: list) -> None:
    """Snapshot ``owner.name`` and then set it, recording the undo entry."""
    applied.append((owner, name, _snapshot(owner, name)))
    setattr(owner, name, value)


def _restore_all(applied: list) -> None:
    """Undo every recorded entry in reverse order."""
    for owner, name, original in reversed(applied):
        try:
            if original is _MISSING:
                if hasattr(owner, name):
                    delattr(owner, name)
            else:
                setattr(owner, name, original)
        except Exception as exc:  # pragma: no cover - rollback is best effort
            logger.warning(
                "Failed to roll back %s.%s: %s",
                getattr(owner, "__name__", owner),
                name,
                exc,
            )


@lru_cache(maxsize=1)
def apply_hygon_patches():
    """Apply Hygon patches that must run before model construction.

    Cached so the patches run once per process. Every patch snapshots what it
    is about to change and returns a rollback callable; when one fails, the
    changes already applied are reverted before the error is raised, so the
    process is never left with a half-installed patch set. ``lru_cache`` does
    not cache exceptions, so the next call retries from that clean state.
    """
    applied: list[Callable[[], None]] = []
    for name, install in (
        ("torch.accelerator attrs", _install_accelerator_missing_attrs),
        ("UVA unavailable", _install_uva_unavailable),
    ):
        try:
            applied.append(install())
        except Exception as exc:
            logger.warning("Hygon patch %r failed: %s", name, exc)
            for rollback in reversed(applied):
                rollback()
            raise RuntimeError(
                "Hygon patches failed and will be retried on the next call: " + name
            ) from exc


def _install_uva_unavailable() -> Callable[[], None]:
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
    import sys

    import vllm.utils.platform_utils as platform_utils

    if getattr(platform_utils, "_hcu_uva_patched", False):
        return _noop

    def _uva_unavailable() -> bool:
        return False

    # Collect every binding before changing any of them: ``from ... import
    # is_uva_available`` copies the callable, so modules imported before this
    # patch ran keep the original name too.
    targets = [platform_utils]
    for module_name, module in list(sys.modules.items()):
        if (
            not module_name.startswith("vllm.")
            or module is None
            or module is platform_utils
        ):
            continue
        if getattr(module, "is_uva_available", None) is not None:
            targets.append(module)

    applied: list = []
    try:
        for owner in targets:
            _apply(owner, "is_uva_available", _uva_unavailable, applied)
        _apply(platform_utils, "_hcu_uva_patched", True, applied)
    except Exception:
        _restore_all(applied)
        raise

    logger.info(
        "Reported UVA as unavailable for Hygon: HCU has no "
        "get_cuda_view_from_cpu_tensor, Model Runner V2 uses the "
        "explicit-copy buffer fallback"
    )
    return lambda: _restore_all(applied)


def _install_accelerator_missing_attrs() -> Callable[[], None]:
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

    Returns a rollback callable that restores the namespace as it was.
    """
    import torch

    if getattr(torch.accelerator, "_hcu_attrs_patched", False):
        return _noop

    # Only fill in the attribute if the installed Torch does not provide it, so
    # a newer DTK that ships the real implementation keeps it instead of being
    # shadowed by the CUDA-shaped fallback below.
    bindings = {}
    if not hasattr(torch.accelerator, "get_memory_info"):

        def get_memory_info(device_index=None):
            return torch.cuda.mem_get_info(device_index)

        bindings["get_memory_info"] = get_memory_info

    applied: list = []
    try:
        for name, fn in bindings.items():
            _apply(torch.accelerator, name, fn, applied)
        _apply(torch.accelerator, "_hcu_attrs_patched", True, applied)
    except Exception:
        _restore_all(applied)
        raise

    logger.info(
        "Patched torch.accelerator APIs for Hygon DTK Torch: %s",
        sorted(bindings),
    )
    return lambda: _restore_all(applied)
