# Copyright (c) 2025 BAAI. All rights reserved.

"""
Platform-aware device cleanup between E2E test cases.

Kills stale serving processes and logs device memory state to prevent
GPU/NPU memory leaks from causing cascading test failures.

Usage::

    from tests.utils.cleanup import device_cleanup

    # Between test cases:
    device_cleanup("cuda")  # NVIDIA GPU cleanup
    device_cleanup("ascend")  # Huawei NPU cleanup
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time


def device_cleanup(platform: str, wait: float = 3.0) -> None:
    """Run platform-specific cleanup between E2E test cases.

    1. Kill stale vllm/model-serving processes
    2. Wait briefly for resources to be released
    3. Log device memory state

    Args:
        platform: Platform name (e.g. ``"cuda"``, ``"ascend"``).
        wait: Seconds to wait after killing processes.
    """
    _kill_stale_processes()

    if wait > 0:
        time.sleep(wait)

    cleanup_fn = _PLATFORM_CLEANUP.get(platform, _cleanup_noop)
    cleanup_fn()


# ---------------------------------------------------------------------------
# Stale process cleanup (platform-agnostic)
# ---------------------------------------------------------------------------

# Process name patterns that indicate a stale serving process
_STALE_PATTERNS = ["vllm serve", "vllm.entrypoints"]


def _kill_stale_processes() -> None:
    """Kill any leftover vllm serving processes."""
    for pattern in _STALE_PATTERNS:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True,
            )
            pids = result.stdout.strip().split("\n")
            pids = [p for p in pids if p and p != str(os.getpid())]

            if pids:
                print(f"[cleanup] Killing stale processes matching '{pattern}': {pids}")
                for pid in pids:
                    with contextlib.suppress(ProcessLookupError, ValueError):
                        os.kill(int(pid), signal.SIGKILL)
        except FileNotFoundError:
            # pgrep not available
            pass


# ---------------------------------------------------------------------------
# Platform-specific cleanup
# ---------------------------------------------------------------------------


def _cleanup_cuda() -> None:
    """Log NVIDIA GPU memory state."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            print("[cleanup] GPU memory (MiB):")
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 4:
                    idx, used, free, total = parts
                    print(f"  GPU {idx}: {used}/{total} MiB used, {free} MiB free")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # nvidia-smi not available or timed out — skip memory logging
        pass


def _cleanup_ascend() -> None:
    """Log Huawei Ascend NPU memory state."""
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            print("[cleanup] NPU state:")
            # Print a condensed summary — npu-smi output is verbose
            for line in result.stdout.strip().split("\n"):
                if "HBM" in line or "Aicore" in line or "NPU" in line:
                    print(f"  {line.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # npu-smi not available or timed out — skip memory logging
        pass


def _cleanup_noop() -> None:
    """No-op cleanup for unknown platforms."""
    pass


# Registry: platform name → cleanup function
_PLATFORM_CLEANUP = {
    "cuda": _cleanup_cuda,
    "ascend": _cleanup_ascend,
}
