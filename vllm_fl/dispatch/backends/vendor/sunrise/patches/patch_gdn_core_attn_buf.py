# Copyright (c) 2026 BAAI. All rights reserved.

"""Reuse GDN core_attn_out buffer to avoid per-iter torch.zeros under cudagraph."""

from __future__ import annotations

import contextvars
import logging
import os
from typing import Any, Optional, Tuple

import torch

_log = logging.getLogger(__name__)

_PATCH_APPLIED = False

# Cached buffer sizes for cudagraph capture vs eager prefill.
_CAPTURE_SIZES: "Optional[frozenset]" = None


def _capture_sizes() -> "frozenset":
    """Cudagraph capture batch sizes for this worker (empty if none/eager)."""
    global _CAPTURE_SIZES
    if _CAPTURE_SIZES is not None:
        return _CAPTURE_SIZES
    sizes: set = set()
    env = os.environ.get("VLLM_FL_GDN_CORE_ATTN_PERSIST_SIZES")
    if env:
        try:
            sizes = {int(x) for x in env.replace(",", " ").split()}
        except ValueError:
            sizes = set()
    else:
        try:
            from vllm.config import get_current_vllm_config

            cc = get_current_vllm_config().compilation_config
            cs = getattr(cc, "cudagraph_capture_sizes", None) or []
            sizes = {int(s) for s in cs}
        except Exception:
            sizes = set()
    _CAPTURE_SIZES = frozenset(sizes)
    return _CAPTURE_SIZES


def _in_cudagraph_capture() -> bool:
    """True while a cudagraph is being captured (forward_cuda Python runs during
    capture, but not during replay), so we know this call's buffer must persist.

    Best-effort fallback for the rare case the capture-size set can't be
    resolved; a false result only means we treat the call as eager (shared
    buffer), which is safe unless we are genuinely mid-capture.
    """
    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return False
        mode = get_forward_context().cudagraph_runtime_mode
        return mode is not None and mode != CUDAGraphMode.NONE
    except Exception:
        return False

# When ``forward_cuda`` is active, the wrapper publishes the expected
# ``(shape, dtype, device, cached_buffer)`` tuple here. ``_intercepted_zeros``
# checks this slot; if the next ``torch.zeros`` call originating from
# ``gdn_linear_attn`` matches the published signature, it returns
# ``cached_buffer`` and clears the slot so any other ``torch.zeros``
# inside the same forward pass goes through normally.
_TARGET: "contextvars.ContextVar[Optional[Tuple[Tuple[int, ...], torch.dtype, torch.device, torch.Tensor]]]" = (
    contextvars.ContextVar("_sunrise_gdn_core_attn_target", default=None)
)


def _is_disabled() -> bool:
    val = os.environ.get("VLLM_FL_SUNRISE_BENCH_BASELINE_GDN_CORE_ATTN_ZEROS")
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}


class _GDNTorchProxy:
    """Per-module proxy that forwards every attribute to ``torch`` except
    ``zeros``, which is routed through :func:`_intercepted_zeros`.

    Installed on ``vllm.model_executor.layers.mamba.gdn_linear_attn.torch``
    so the global ``torch`` module is left untouched.
    """

    __slots__ = ("_real_torch",)

    def __init__(self, real_torch: Any) -> None:
        object.__setattr__(self, "_real_torch", real_torch)

    def __getattr__(self, name: str) -> Any:
        if name == "zeros":
            return _intercepted_zeros
        return getattr(self._real_torch, name)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        # GDN module code never reassigns ``torch.<x>``; reject mutations
        # so a future upstream refactor that does this fails loud.
        raise AttributeError(
            f"_GDNTorchProxy: refuse to set attribute {name!r}; the proxy is "
            "read-only by design."
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<_GDNTorchProxy real={self._real_torch!r}>"


def _intercepted_zeros(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Replacement for ``torch.zeros`` (visible only inside
    ``gdn_linear_attn``) that returns a cached buffer when the active
    ``forward_cuda`` published a matching target.

    Falls through to the real ``torch.zeros`` for every other call --
    including any ``torch.zeros`` invocation in ``gdn_linear_attn`` whose
    shape/dtype/device do not match the published target (e.g. the
    ``ChunkGatedDeltaRule`` allocations on the prefill path).
    """
    target = _TARGET.get()
    if target is not None:
        target_shape, target_dtype, target_device, target_buf = target

        if args:
            size_arg = args[0]
        else:
            size_arg = kwargs.get("size")

        if size_arg is not None:
            if isinstance(size_arg, torch.Size):
                shape_tup: Optional[Tuple[int, ...]] = tuple(size_arg)
            elif isinstance(size_arg, (list, tuple)):
                shape_tup = tuple(size_arg)
            else:
                shape_tup = None

            if shape_tup is not None:
                dtype = kwargs.get("dtype")
                device_arg = kwargs.get("device")
                if device_arg is None:
                    # ``torch.zeros`` with no device kwarg uses the
                    # default device. We deliberately only match when the
                    # caller explicitly passes ``device=...`` so we never
                    # capture a CPU-default call by accident.
                    device = None
                else:
                    device = torch.device(device_arg)

                if (
                    shape_tup == target_shape
                    and dtype == target_dtype
                    and device == target_device
                ):
                    # Clear the slot so a second ``torch.zeros`` inside
                    # the same forward (if any) does NOT also grab the
                    # cached buffer.
                    _TARGET.set(None)
                    return target_buf

    return torch.zeros(*args, **kwargs)


def _make_forward_cuda_wrapper(orig_forward_cuda):
    """Return a wrapped ``forward_cuda`` that publishes the target slot
    for :func:`_intercepted_zeros` to consume.

    The wrapper is per-class; ``orig_forward_cuda`` is bound at patch
    time and captured here so we don't recurse via the rebound method.
    """

    def _patched_forward_cuda(self, hidden_states: torch.Tensor, output: torch.Tensor):
        # Per-call kill-switch (cheap env read; cached lookups are fine
        # because env vars are stable for the lifetime of the worker).
        if _is_disabled():
            return orig_forward_cuda(self, hidden_states, output)

        num_tokens = hidden_states.size(0)
        tail_shape: Tuple[int, ...] = (
            self.num_v_heads // self.tp_size,
            self.head_v_dim,
        )
        expected_shape: Tuple[int, ...] = (num_tokens, *tail_shape)
        expected_dtype = hidden_states.dtype
        expected_device = hidden_states.device

        # Keying the cache on ``num_tokens`` (the original design) leaks
        # unboundedly: every distinct eager prefill / mixed batch size adds a
        # buffer that is never released, so a workload with many request
        # lengths (e.g. an accuracy sweep) grows this dict by tens of GiB
        # across the GDN layers. But we cannot simply share one grow-to-max
        # buffer either: under ``FULL_DECODE_ONLY`` the decode path is captured
        # into a cudagraph that bakes in the buffer's address, so a
        # captured-size buffer must never be freed/reallocated.
        #
        # So we persist a per-size buffer ONLY for cudagraph-captured sizes
        # (a small, fixed set); every other call -- all eager steps of
        # arbitrary length -- shares a single grow-to-max buffer and gets a
        # ``[:num_tokens]`` view. The cache is thus bounded regardless of how
        # many distinct request lengths appear.
        must_persist = (num_tokens in _capture_sizes()) or _in_cudagraph_capture()

        cache = self.__dict__.setdefault("_sunrise_core_attn_out_cache", {})
        if must_persist:
            cache_key = (num_tokens, expected_dtype, expected_device)
            buf = cache.get(cache_key)
            if buf is None:
                # Allocate via the **real** ``torch.zeros`` (NOT the intercept)
                # so we don't accidentally publish-and-consume the slot during
                # bootstrap.
                buf = torch.zeros(
                    expected_shape, dtype=expected_dtype, device=expected_device
                )
                cache[cache_key] = buf
        else:
            big_key = ("_shared", expected_dtype, expected_device)
            base = cache.get(big_key)
            if base is None or base.size(0) < num_tokens:
                base = torch.zeros(
                    (num_tokens, *tail_shape),
                    dtype=expected_dtype,
                    device=expected_device,
                )
                cache[big_key] = base
            buf = base[:num_tokens]

        token = _TARGET.set((expected_shape, expected_dtype, expected_device, buf))
        try:
            return orig_forward_cuda(self, hidden_states, output)
        finally:
            _TARGET.reset(token)

    _patched_forward_cuda.__wrapped__ = orig_forward_cuda  # type: ignore[attr-defined]
    _patched_forward_cuda.__name__ = "forward_cuda"
    _patched_forward_cuda.__qualname__ = "GatedDeltaNetAttention.forward_cuda"
    return _patched_forward_cuda


def apply_patch() -> bool:
    """Install the GDN ``core_attn_out`` zero-elision patch.

    Idempotent; a silent no-op on subsequent calls. Returns ``True`` when
    the patch was applied in this call, ``False`` if it was already
    applied, the upstream class was missing, or the kill-switch is on.
    """
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return False

    if _is_disabled():
        _log.info(
            "Sunrise GDN core_attn_out patch skipped: "
            "VLLM_FL_SUNRISE_BENCH_BASELINE_GDN_CORE_ATTN_ZEROS is set."
        )
        return False

    try:
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_mod
    except Exception as exc:  # pragma: no cover - upstream rename safety net
        _log.warning(
            "Sunrise GDN core_attn_out patch skipped: failed to import "
            "vllm.model_executor.layers.mamba.gdn_linear_attn (%s). Patch "
            "had no effect.",
            exc,
        )
        return False

    GatedDeltaNetAttention = getattr(_gdn_mod, "GatedDeltaNetAttention", None)
    if GatedDeltaNetAttention is None:  # pragma: no cover
        _log.warning(
            "Sunrise GDN core_attn_out patch skipped: "
            "GatedDeltaNetAttention not found in gdn_linear_attn module. "
            "Patch had no effect."
        )
        return False

    orig_forward_cuda = GatedDeltaNetAttention.forward_cuda
    if getattr(orig_forward_cuda, "_sunrise_core_attn_buf_patched", False):
        _PATCH_APPLIED = True
        return False

    # 1. Wrap forward_cuda so the contextvar is published per call.
    wrapped = _make_forward_cuda_wrapper(orig_forward_cuda)
    wrapped._sunrise_core_attn_buf_patched = True  # type: ignore[attr-defined]
    GatedDeltaNetAttention.forward_cuda = wrapped

    # 2. Install the per-module torch proxy. Idempotent: if a previous
    #    apply_patch already installed a proxy, reusing it is safe.
    current_torch = getattr(_gdn_mod, "torch", None)
    if not isinstance(current_torch, _GDNTorchProxy):
        if current_torch is None:  # pragma: no cover - unexpected
            current_torch = torch
        _gdn_mod.torch = _GDNTorchProxy(current_torch)

    _PATCH_APPLIED = True
    _log.info(
        "Applied PTPU GDN core_attn_out zero-elision patch: "
        "GatedDeltaNetAttention.forward_cuda reuses a per-instance zero "
        "buffer instead of allocating + zeroing a fresh torch.zeros every "
        "call. Disable with "
        "VLLM_FL_SUNRISE_BENCH_BASELINE_GDN_CORE_ATTN_ZEROS=1."
    )
    return True
