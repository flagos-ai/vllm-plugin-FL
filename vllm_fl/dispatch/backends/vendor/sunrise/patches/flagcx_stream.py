# Copyright (c) 2025 BAAI. All rights reserved.
"""Registry and stream sync helpers for PTPU cudagraph + FlagCX."""

from __future__ import annotations

from typing import Any

_ptpu_cudagraph_ar_stream: Any | None = None


def set_ptpu_cudagraph_ar_stream(stream: Any) -> None:
    global _ptpu_cudagraph_ar_stream
    _ptpu_cudagraph_ar_stream = stream


def get_ptpu_cudagraph_ar_stream() -> Any | None:
    return _ptpu_cudagraph_ar_stream


def sync_capture_stream_before_replay() -> None:
    """Ensure input updates on the compute stream finish before graph replay."""
    stream = _ptpu_cudagraph_ar_stream
    if stream is None:
        return
    from vllm.platforms import current_platform

    if current_platform.device_type != "ptpu":
        return
    compute_stream = current_platform.torch_device_fn.current_stream()
    if compute_stream is not stream:
        stream.wait_stream(compute_stream)


def sync_compute_stream_after_replay() -> None:
    """Ensure post-replay work on the compute stream sees graph results."""
    stream = _ptpu_cudagraph_ar_stream
    if stream is None:
        return
    from vllm.platforms import current_platform

    if current_platform.device_type != "ptpu":
        return
    compute_stream = current_platform.torch_device_fn.current_stream()
    if compute_stream is not stream:
        compute_stream.wait_stream(stream)
