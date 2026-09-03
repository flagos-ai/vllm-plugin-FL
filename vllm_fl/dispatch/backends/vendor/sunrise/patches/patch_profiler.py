# Copyright (c) 2026 BAAI. All rights reserved.

"""Map torch profiler CUDA activity to PrivateUse1 on PTPU."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_patched = False


def install() -> None:
    """Alias ``TorchProfilerActivityMap['CUDA']`` to ``PrivateUse1`` on PTPU.

    Idempotent. Silent no-op when:
      * not on PTPU,
      * ``vllm.profiler.wrapper`` cannot be imported,
      * ``torch.profiler.ProfilerActivity.PrivateUse1`` is unavailable,
      * the alias has already been installed (by this function or any
        other patch).
    """
    global _patched
    if _patched:
        return

    pid = os.getpid()

    try:
        from vllm.platforms import current_platform
    except Exception as exc:
        logger.info(
            "[FL_INIT] profile.install skipped: cannot import current_platform "
            "(%s) pid=%d",
            exc,
            pid,
        )
        return

    device_type = getattr(current_platform, "device_type", None)
    if device_type != "ptpu":
        logger.info(
            "[FL_INIT] profile.install skipped: device_type=%r != 'ptpu' pid=%d",
            device_type,
            pid,
        )
        return

    try:
        import torch.profiler
    except Exception as exc:
        logger.info(
            "[FL_INIT] profile.install skipped: torch.profiler unavailable "
            "(%s) pid=%d",
            exc,
            pid,
        )
        return

    private_use1 = getattr(
        torch.profiler.ProfilerActivity, "PrivateUse1", None
    )
    if private_use1 is None:
        logger.info(
            "[FL_INIT] profile.install skipped: "
            "torch.profiler.ProfilerActivity.PrivateUse1 not present "
            "(torch=%s) pid=%d",
            getattr(__import__("torch"), "__version__", "?"),
            pid,
        )
        return

    try:
        from vllm.profiler import wrapper as _vllm_profiler_wrapper
    except Exception as exc:
        logger.info(
            "[FL_INIT] profile.install skipped: cannot import "
            "vllm.profiler.wrapper (%s) pid=%d",
            exc,
            pid,
        )
        return

    activity_map = getattr(
        _vllm_profiler_wrapper, "TorchProfilerActivityMap", None
    )
    if activity_map is None or "CUDA" not in activity_map:
        logger.info(
            "[FL_INIT] profile.install skipped: "
            "vllm.profiler.wrapper.TorchProfilerActivityMap missing or has no "
            "'CUDA' entry pid=%d",
            pid,
        )
        return

    prev = activity_map.get("CUDA")
    if prev is private_use1:
        _patched = True
        logger.info(
            "[FL_INIT] profile.install installed (no-op): "
            "TorchProfilerActivityMap['CUDA'] is already PrivateUse1 pid=%d",
            pid,
        )
        return

    activity_map["CUDA"] = private_use1
    _patched = True
    logger.info(
        "[FL_INIT] profile.install installed: "
        "TorchProfilerActivityMap['CUDA'] -> ProfilerActivity.PrivateUse1 "
        "(was %s) pid=%d",
        prev,
        pid,
    )
