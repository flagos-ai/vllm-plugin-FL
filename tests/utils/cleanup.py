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
from collections.abc import Callable


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

    # Clear framework cache to reclaim memory held by PyTorch allocator
    cache_fn = _PLATFORM_CACHE_CLEAR.get(platform, _cache_clear_noop)
    cache_fn()

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


# ---------------------------------------------------------------------------
# Memory info (platform-specific)
# ---------------------------------------------------------------------------


def _mem_info_cuda() -> list[tuple[int, int]]:
    """Return [(free_bytes, total_bytes), ...] for each CUDA device."""
    import torch

    result = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        result.append((free, total))
    return result


def _mem_info_ascend() -> list[tuple[int, int]]:
    """Return [(free_bytes, total_bytes), ...] for each Ascend NPU."""
    import torch

    try:
        import torch_npu  # noqa: F401

        result = []
        for i in range(torch.npu.device_count()):
            free, total = torch.npu.mem_get_info(i)
            result.append((free, total))
        return result
    except (ImportError, AttributeError):
        return []


def _mem_info_noop() -> list[tuple[int, int]]:
    return []


_PLATFORM_MEMORY_INFO: dict[str, Callable[[], list[tuple[int, int]]]] = {
    "cuda": _mem_info_cuda,
    "ascend": _mem_info_ascend,
}


# ---------------------------------------------------------------------------
# Cache clear (platform-specific)
# ---------------------------------------------------------------------------


def _cache_clear_cuda() -> None:
    """Clear PyTorch CUDA cache."""
    import torch

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _cache_clear_ascend() -> None:
    """Clear PyTorch NPU cache."""
    try:
        import torch
        import torch_npu  # noqa: F401

        torch.npu.empty_cache()
    except (ImportError, AttributeError):
        pass


def _cache_clear_noop() -> None:
    pass


_PLATFORM_CACHE_CLEAR: dict[str, Callable[[], None]] = {
    "cuda": _cache_clear_cuda,
    "ascend": _cache_clear_ascend,
}


# ---------------------------------------------------------------------------
# Public memory API
# ---------------------------------------------------------------------------


def get_device_memory(platform: str) -> list[tuple[float, float]]:
    """Return [(free_mb, total_mb), ...] for each device on the platform."""
    mem_fn = _PLATFORM_MEMORY_INFO.get(platform, _mem_info_noop)
    return [(free / (1024 * 1024), total / (1024 * 1024)) for free, total in mem_fn()]


def wait_for_memory(
    platform: str,
    gpu_memory_utilization: float = 0.9,
    timeout: int = 1800,
    interval: int = 30,
) -> tuple[bool, str]:
    """Wait until devices have enough free memory for the given utilization.

    Args:
        platform: Platform name (e.g. ``"cuda"``, ``"ascend"``).
        gpu_memory_utilization: Fraction of total memory the model needs.
        timeout: Maximum seconds to wait (default 30 min).
        interval: Seconds between polls.

    Returns:
        ``(True, info)`` if memory is available, ``(False, info)`` on timeout.
    """
    mem_fn = _PLATFORM_MEMORY_INFO.get(platform, _mem_info_noop)
    cache_fn = _PLATFORM_CACHE_CLEAR.get(platform, _cache_clear_noop)

    deadline = time.time() + timeout
    attempt = 0

    while True:
        attempt += 1

        # Kill stale vllm processes from previous e2e tests
        _kill_stale_processes()
        # Clear framework cache
        cache_fn()
        # Brief pause for resources to be released
        time.sleep(1)

        mem_info = mem_fn()
        if not mem_info:
            return (True, "no devices detected, skipping memory check")

        # Check each device
        all_ok = True
        lines = []
        for i, (free, total) in enumerate(mem_info):
            required = total * gpu_memory_utilization
            free_mb = free / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            required_mb = required / (1024 * 1024)
            ok = free >= required
            status = "OK" if ok else "WAIT"
            lines.append(
                f"  Device {i}: {free_mb:.0f}/{total_mb:.0f} MiB free, "
                f"need {required_mb:.0f} MiB ({gpu_memory_utilization:.0%}) [{status}]"
            )
            if not ok:
                all_ok = False

        info = "\n".join(lines)
        print(
            f"[memory] Attempt {attempt} (util={gpu_memory_utilization:.0%}):\n{info}"
        )

        if all_ok:
            return (True, info)

        if time.time() >= deadline:
            return (False, info)

        print(f"[memory] Waiting {interval}s for memory to free up...")
        time.sleep(interval)
