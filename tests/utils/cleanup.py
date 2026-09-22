# Copyright (c) 2025 BAAI. All rights reserved.

"""
Platform-aware device cleanup between E2E test cases.

Kills stale serving processes and logs device memory state to prevent
GPU/NPU memory leaks from causing cascading test failures.

Usage::

    from tests.utils.cleanup import device_cleanup

    # Between test cases:
    device_cleanup("cuda")  # NVIDIA GPU cleanup
    device_cleanup("ascend")  # Huawei Ascend NPU cleanup
    device_cleanup("hygon")  # Hygon DCU cleanup
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Callable


def device_cleanup(
    platform: str,
    wait: float = 3.0,
    slots: list[int] | None = None,
) -> None:
    """Run platform-specific cleanup between E2E test cases.

    1. Kill stale vllm/model-serving processes
    2. Clear framework allocator cache to reclaim held memory
    3. Wait briefly for device resources to be released
    4. Log device memory state

    Args:
        platform: Platform name (e.g. ``"cuda"``, ``"ascend"``, ``"hygon"``).
        wait: Seconds to wait after killing processes before logging memory.
        slots: Device slot indices this case owns. When provided, kill is
            restricted to processes whose VISIBLE_DEVICES env matches these
            slots, preventing accidental kill of concurrently running cases.
            ``None`` (serial path) retains the original global-kill behaviour.
    """
    _kill_stale_processes(slots=slots)

    # On PPU, driver crashes can leave processes holding device handles that
    # pgrep-based cleanup misses. fuser on the device files catches them.
    if platform == "thead":
        _kill_stale_ppu_processes(slots=slots)

    # Clear framework cache to reclaim memory held by the PyTorch allocator
    cache_fn = _PLATFORM_CACHE_CLEAR.get(platform, _cache_clear_noop)
    cache_fn()

    if wait > 0:
        time.sleep(wait)

    # Log current device memory state for diagnostic purposes
    log_fn = _PLATFORM_MEMORY_LOG.get(platform, _log_memory_noop)
    log_fn()


# ---------------------------------------------------------------------------
# Stale process cleanup (platform-agnostic)
# ---------------------------------------------------------------------------

# Process name patterns that indicate a stale vllm process.
# Includes both serving processes (vllm.entrypoints) and inference worker
# processes that rename themselves via prctl (VLLM::Worker, VLLM::EngineCore).
_STALE_PATTERNS = [
    "vllm serve",
    "vllm.entrypoints",
    "VLLM::Worker",
    "VLLM::EngineCore",
]

# All env vars that different platforms use to restrict device visibility.
_VISIBLE_DEVICE_VARS = [
    "CUDA_VISIBLE_DEVICES",  # cuda, hygon, kunlunxin, thead, iluvatar
    "ASCEND_RT_VISIBLE_DEVICES",  # ascend
    "MACA_VISIBLE_DEVICES",  # metax
    "TOPS_VISIBLE_DEVICES",  # enflame
    "MTHREADS_VISIBLE_DEVICES",  # musa
]


def _pid_owns_slots(pid: str, slots: list[int]) -> bool:
    """Return True if process *pid* was started with VISIBLE_DEVICES that
    overlaps *slots*.

    Reads ``/proc/{pid}/environ`` (Linux-only) to inspect the process's
    original environment.  Returns False if the file is unreadable (process
    already gone or permission denied) — caller treats unreadable as
    "do not kill" to avoid accidental kills.
    """
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            env_raw = f.read()
        env = dict(
            e.split("=", 1)
            for e in env_raw.decode(errors="replace").split("\0")
            if "=" in e
        )
        slot_set = set(slots)
        for var in _VISIBLE_DEVICE_VARS:
            val = env.get(var, "")
            if not val:
                continue
            pid_slots = {
                int(x) for x in val.split(",") if x.strip().lstrip("-").isdigit()
            }
            if pid_slots & slot_set:
                return True
    except (OSError, ValueError):
        pass
    return False


def _kill_stale_processes(slots: list[int] | None = None) -> None:
    """Kill any leftover vllm serving or inference worker processes.

    Args:
        slots: When provided, only kill processes whose VISIBLE_DEVICES env
            overlaps *slots*.  ``None`` kills all matching processes (serial
            path — original behaviour).
    """
    for pattern in _STALE_PATTERNS:
        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True,
            )
            pids = result.stdout.strip().split("\n")
            pids = [p for p in pids if p and p != str(os.getpid())]

            if slots is not None:
                pids = [p for p in pids if _pid_owns_slots(p, slots)]

            if pids:
                print(f"[cleanup] Killing stale processes matching '{pattern}': {pids}")
                for pid in pids:
                    with contextlib.suppress(ProcessLookupError, ValueError):
                        os.kill(int(pid), signal.SIGKILL)
        except FileNotFoundError:
            # pgrep not available
            pass


def _kill_stale_ppu_processes(slots: list[int] | None = None) -> None:
    """Kill processes holding T-Head PPU device handles via fuser.

    PPU driver crashes (e.g. Hggc failure) can leave worker processes in
    uninterruptible sleep. Pattern-based pgrep may miss them. fuser on the
    PPU device files gives a definitive list of processes with open handles.

    Args:
        slots: When provided, only kill processes whose VISIBLE_DEVICES env
            overlaps *slots*.  ``None`` kills all matching processes.
    """
    import glob

    ppu_devs = sorted(glob.glob("/dev/alixpu_ppu*"))
    if not ppu_devs:
        return

    pids: set[str] = set()
    for dev in ppu_devs:
        try:
            result = subprocess.run(
                ["fuser", dev],
                capture_output=True,
                text=True,
            )
            for pid in result.stdout.split():
                pid = pid.strip()
                if pid and pid != str(os.getpid()):
                    pids.add(pid)
        except FileNotFoundError:
            # fuser not available — fall back to pattern-based kill only
            return

    if slots is not None:
        pids = {p for p in pids if _pid_owns_slots(p, slots)}

    if pids:
        print(f"[cleanup] Killing stale PPU processes (fuser): {sorted(pids)}")
        for pid in pids:
            with contextlib.suppress(ProcessLookupError, ValueError):
                os.kill(int(pid), signal.SIGKILL)


# ---------------------------------------------------------------------------
# Platform-specific memory logging
# ---------------------------------------------------------------------------


def _log_memory_cuda() -> None:
    """Log NVIDIA GPU memory state via nvidia-smi."""
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


def _log_memory_ascend() -> None:
    """Log Huawei Ascend NPU memory state via npu-smi."""
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


def _log_memory_hygon() -> None:
    """Log Hygon DCU memory state via hy-smi."""
    try:
        result = subprocess.run(
            ["hy-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            print("[cleanup] DCU memory (MiB):")
            for line in result.stdout.strip().split("\n"):
                # Lines look like: HCU[0]  : vram Total Memory (MiB): 65520
                #                  HCU[0]  : vram Total Used Memory (MiB): 51920
                if "Total Memory" in line or "Total Used Memory" in line:
                    print(f"  {line.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # hy-smi not available or timed out — skip memory logging
        pass


def _log_memory_thead() -> None:
    """Log T-Head PPU memory state via ppu-smi (symlinked as nvidia-smi)."""
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
            print("[cleanup] PPU memory (MiB):")
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 4:
                    idx, used, free, total = parts
                    print(f"  PPU {idx}: {used}/{total} MiB used, {free} MiB free")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _log_memory_noop() -> None:
    """No-op memory log for unknown platforms."""
    pass


# Registry: platform name → memory logging function
_PLATFORM_MEMORY_LOG: dict[str, Callable[[], None]] = {
    "cuda": _log_memory_cuda,
    "ascend": _log_memory_ascend,
    "hygon": _log_memory_hygon,
    "thead": _log_memory_thead,
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


def _mem_info_thead() -> list[tuple[int, int]]:
    """Return [(free_bytes, total_bytes), ...] for each T-Head PPU.

    Uses ppu-smi (symlinked as nvidia-smi) rather than torch.cuda.mem_get_info
    because the PPU CUDA-compat layer only reports the current process's
    allocations. ppu-smi queries the device driver directly and sees memory
    held by ALL processes, which is what we need for the memory guard.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        entries = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                free_mb, total_mb = int(parts[0]), int(parts[1])
                entries.append((free_mb * 1024 * 1024, total_mb * 1024 * 1024))
        return entries
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return []


_PLATFORM_MEMORY_INFO: dict[str, Callable[[], list[tuple[int, int]]]] = {
    "cuda": _mem_info_cuda,
    "ascend": _mem_info_ascend,
    # Hygon DCUs are exposed to PyTorch as CUDA devices.
    "hygon": _mem_info_cuda,
    # T-Head PPU: use ppu-smi for device-wide visibility across all processes.
    "thead": _mem_info_thead,
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
    # Hygon DCUs and T-Head PPUs are exposed to PyTorch as CUDA devices.
    "hygon": _cache_clear_cuda,
    "thead": _cache_clear_cuda,
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
    device_indices: list[int] | None = None,
) -> tuple[bool, str]:
    """Wait until devices have enough free memory for the given utilization.

    Args:
        platform: Platform name (e.g. ``"cuda"``, ``"ascend"``).
        gpu_memory_utilization: Fraction of total memory the model needs.
        timeout: Maximum seconds to wait (default 30 min).
        interval: Seconds between polls.
        device_indices: When provided, only check and kill for these device
            indices.  Used in parallel e2e runs so each case only inspects
            its own allocated slots and does not interfere with other cases.
            ``None`` checks all devices (serial path).

    Returns:
        ``(True, info)`` if memory is available, ``(False, info)`` on timeout.
    """
    mem_fn = _PLATFORM_MEMORY_INFO.get(platform, _mem_info_noop)
    cache_fn = _PLATFORM_CACHE_CLEAR.get(platform, _cache_clear_noop)
    kill_slots = device_indices  # same list used for both kill and mem filter

    deadline = time.time() + timeout
    attempt = 0

    while True:
        attempt += 1

        # Kill stale vllm processes from previous e2e tests.
        # In parallel mode, restrict to slots owned by this case so we
        # don't accidentally kill workers belonging to other running cases.
        _kill_stale_processes(slots=kill_slots)
        if platform == "thead":
            _kill_stale_ppu_processes(slots=kill_slots)
        # Clear framework cache
        cache_fn()
        # Brief pause for resources to be released
        time.sleep(5)

        mem_info = mem_fn()
        if not mem_info:
            return (True, "no devices detected, skipping memory check")

        # In parallel mode restrict the check to the assigned device slots.
        if device_indices is not None:
            mem_info = [mem_info[i] for i in device_indices if i < len(mem_info)]
            if not mem_info:
                return (True, "assigned device indices out of range, skipping check")

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
            dev_label = device_indices[i] if device_indices else i
            lines.append(
                f"  Device {dev_label}: {free_mb:.0f}/{total_mb:.0f} MiB free, "
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
