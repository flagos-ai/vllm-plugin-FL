# Copyright (c) 2026 BAAI. All rights reserved.
"""Optional shape-aware dispatch for FlagGems' ``aten.mm`` kernel.

FlagGems' generic ``aten.mm`` implementation can be a good fit for large-M
prefill GEMMs, while the native CUDA implementation can be better for tiny-M
decode GEMVs. This module installs a *process-local* CUDA dispatch wrapper
only when explicitly requested by policy.

The wrapper is deliberately small and uses tensor metadata only.  In
particular, it does not call ``item()``, synchronize, or inspect device data,
so it is safe to use from CUDA-graph/static-shape paths.

The common worker policy is disabled by default. Model integrations may choose
their own default after model-specific validation. An explicit
``VLLM_FL_FLAGOS_MM_SHAPE_AWARE`` setting always wins, and ``mm`` can still be
excluded from FlagGems with the existing platform-specific list.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

ENABLE_ENV = "VLLM_FL_FLAGOS_MM_SHAPE_AWARE"
THRESHOLD_ENV = "VLLM_FL_FLAGOS_MM_DECODE_MAX_M"
DEFAULT_DECODE_MAX_M = 64


@dataclass(frozen=True)
class ShapeAwareMMState:
    """Handles retained for the lifetime of the process.

    ``torch.library.Library`` registrations are revoked when the Library
    object is garbage-collected, so the handle must be kept alive globally.
    The original kernels are ``torch.library.get_kernel``'s
    ``SafeKernelFunction`` handles.  They must be retained and called through
    ``call_boxed(dispatch_keys, ...)`` after the override is installed; a
    normal Python call is not supported by torch 2.11 and a redispatch using a
    guessed keyset can recurse or bypass the intended backend.
    """

    threshold: int
    native_mm: Any
    flaggems_mm: Any
    library: Any


_STATE: ShapeAwareMMState | None = None


def _parse_bool_env(name: str, *, default: bool = False) -> bool:
    """Parse an override without accepting ambiguous values."""

    value = os.environ.get(name)
    if value is None:
        return default
    if value == "1" or value == "true":
        return True
    if value == "0" or value == "false":
        return False
    raise ValueError(
        f"{name} must be exactly one of 0, 1, false, true; got {value!r}"
    )


def _parse_threshold_env(name: str = THRESHOLD_ENV) -> int:
    """Parse the decode-M threshold strictly as a positive decimal integer."""

    value = os.environ.get(name)
    if value is None:
        return DEFAULT_DECODE_MAX_M
    # Do not accept whitespace, signs, floating point, or a suffix.  This
    # prevents a typo in a launch command from silently changing the policy.
    if re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError(f"{name} must be a positive decimal integer; got {value!r}")
    threshold = int(value, 10)
    if threshold < 1:
        raise ValueError(f"{name} must be >= 1; got {value!r}")
    return threshold


def _is_native_candidate(a: Any, b: Any, threshold: int) -> bool:
    """Return whether this metadata-only call should use native CUDA ``mm``."""

    # ``aten.mm`` is 2-D by contract.  Keeping this check explicit also makes
    # the wrapper robust when called through a tracing/decomposition path.
    a_device = getattr(a, "device", None)
    b_device = getattr(b, "device", None)
    if getattr(a_device, "type", None) != "cuda":
        return False
    if getattr(b_device, "type", None) != "cuda":
        return False
    if getattr(a, "ndim", None) != 2 or getattr(b, "ndim", None) != 2:
        return False

    a_shape = getattr(a, "shape", ())
    b_shape = getattr(b, "shape", ())
    if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] < 1:
        return False
    if a_shape[0] > threshold:
        return False

    # BF16 is the primary serving dtype. FP16/FP32 are included because they
    # are supported by native CUDA and useful for unit/integration probes. Do
    # not alter mixed-dtype or integer/FP8 dispatch semantics here.
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if getattr(a, "dtype", None) not in supported_dtypes:
        return False
    if getattr(b, "dtype", None) != getattr(a, "dtype", None):
        return False

    # Native CUDA mm handles the usual row-major activation and row/column
    # major weight layouts.  Reject unusual strided views so that this small
    # prototype never changes FlagGems' existing stride-normalization path.
    a_stride = a.stride()
    b_stride = b.stride()
    if len(a_stride) != 2 or len(b_stride) != 2:
        return False
    if a_stride[1] != 1 or a_stride[0] < 1:
        return False
    if not (b_stride[1] == 1 or b_stride[0] == 1):
        return False
    return not (b_stride[0] < 1 or b_stride[1] < 1)


def _get_registered_mm_kernel() -> Any:
    """Capture a CUDA ``aten::mm`` kernel as a boxed-safe handle.

    ``torch.library.get_kernel`` returns a ``SafeKernelFunction`` on the
    PyTorch build used by FlagOS.  Its ``call_boxed`` method is the supported
    way to invoke a retained dispatcher kernel from a ``with_keyset``
    implementation.  Refuse raw Python callables: accepting one would make
    it too easy to install a wrapper which recurses into itself.
    """

    get_kernel = getattr(torch.library, "get_kernel", None)
    if not callable(get_kernel):
        raise RuntimeError(
            "Shape-aware FlagGems mm requires torch.library.get_kernel to "
            "capture the pre-existing CUDA kernel safely"
        )
    try:
        kernel = get_kernel("aten::mm", "CUDA")
    except Exception as exc:  # pragma: no cover - backend/version dependent
        raise RuntimeError(
            "Unable to capture the CUDA aten::mm kernel; refusing a "
            "potentially recursive override"
        ) from exc
    call_boxed = getattr(kernel, "call_boxed", None)
    if not callable(call_boxed):
        raise RuntimeError(
            "torch.library.get_kernel('aten::mm', 'CUDA') did not return "
            "a SafeKernelFunction with call_boxed; this torch build cannot "
            "safely retain both native and FlagGems kernels"
        )
    return kernel


def capture_native_mm_kernel() -> Any:
    """Capture native CUDA ``aten.mm`` before ``flag_gems.enable``.

    The worker calls this before FlagGems changes the CUDA registration.  It
    is intentionally a separate operation so that looking up the kernel after
    the override cannot accidentally capture the wrapper itself.
    """

    return _get_registered_mm_kernel()


def is_shape_aware_mm_enabled(*, default: bool = False) -> bool:
    """Return the policy switch without touching dispatch state."""

    return _parse_bool_env(ENABLE_ENV, default=default)


def is_mm_dispatch_enabled(
    whitelist: list[str] | None,
    blacklist: list[str] | None,
) -> bool:
    """Return whether FlagGems owns ``aten.mm`` for this worker.

    A shape-aware override needs both the captured native kernel and the
    post-``flag_gems.enable`` FlagGems kernel. Installing it while ``mm`` is
    excluded would otherwise capture the native kernel twice and silently
    defeat the explicit rollback switch. Whitelists take precedence in the
    same way as :func:`get_flag_gems_whitelist_blacklist`.
    """

    if whitelist is not None:
        return "mm" in whitelist
    return "mm" not in (blacklist or ())


def _register_override(library: Any, wrapper: Callable[..., Any]) -> None:
    """Install the wrapper only on torch versions supporting safe override."""

    impl = library.impl
    try:
        signature = inspect.signature(impl)
    except (TypeError, ValueError) as exc:  # pragma: no cover - C API variant
        raise RuntimeError(
            "Cannot inspect torch.library.Library.impl; refusing to override "
            "aten::mm without an explicit with_keyset/allow_override API"
        ) from exc
    missing = {
        name
        for name in ("with_keyset", "allow_override")
        if name not in signature.parameters
    }
    if missing:
        raise RuntimeError(
            "torch.library.Library.impl lacks "
            f"{', '.join(sorted(missing))}; this torch build cannot safely "
            "replace the FlagGems CUDA aten::mm kernel"
        )
    try:
        impl("mm", wrapper, "CUDA", with_keyset=True, allow_override=True)
    except TypeError as exc:  # pragma: no cover - incompatible torch ABI
        raise RuntimeError(
            "torch.library.Library.impl rejected the safe with_keyset/"
            "allow_override registration; refusing a potentially recursive "
            "aten::mm override"
        ) from exc


def apply_shape_aware_mm(
    *,
    native_mm_kernel: Any | None = None,
    default_enabled: bool = False,
) -> bool:
    """Install the opt-in shape-aware CUDA ``aten.mm`` wrapper.

    Returns ``True`` when a new override is installed and ``False`` when the
    feature is disabled or was already installed.  Invalid configuration and
    unsupported torch registration APIs raise immediately with an actionable
    error; silently falling back would make a benchmark claim ambiguous.
    """

    global _STATE

    if not _parse_bool_env(ENABLE_ENV, default=default_enabled):
        return False
    threshold = _parse_threshold_env()
    if _STATE is not None:
        if _STATE.threshold != threshold:
            raise RuntimeError(
                "Shape-aware FlagGems mm is already installed with threshold "
                f"{_STATE.threshold}, cannot change it to {threshold} in-process"
            )
        return False

    if native_mm_kernel is None:
        raise RuntimeError(
            "Shape-aware FlagGems mm must capture native CUDA aten::mm "
            "before flag_gems.enable; call capture_native_mm_kernel() first"
        )
    if not callable(getattr(native_mm_kernel, "call_boxed", None)):
        raise RuntimeError(
            "native_mm_kernel is not a SafeKernelFunction with call_boxed; "
            "refusing a recursive or ambiguous aten::mm override"
        )

    flaggems_mm = _get_registered_mm_kernel()

    def shape_aware_mm(dispatch_keys: Any, a: Any, b: Any) -> Any:
        # ``dispatch_keys`` is supplied by torch.library's with_keyset=True
        # wrapper.  Passing it to the retained SafeKernelFunction preserves
        # the original dispatcher context without a host sync or guessed
        # redispatch keyset.
        if _is_native_candidate(a, b, threshold):
            return native_mm_kernel.call_boxed(dispatch_keys, a, b)
        return flaggems_mm.call_boxed(dispatch_keys, a, b)

    shape_aware_mm.__name__ = "shape_aware_mm"
    shape_aware_mm._vllm_fl_shape_aware_mm = True
    shape_aware_mm._vllm_fl_original_flaggems_mm = flaggems_mm
    shape_aware_mm._vllm_fl_native_mm = native_mm_kernel

    library = torch.library.Library("aten", "IMPL")
    _register_override(library, shape_aware_mm)
    _STATE = ShapeAwareMMState(
        threshold=threshold,
        native_mm=native_mm_kernel,
        flaggems_mm=flaggems_mm,
        library=library,
    )
    logger.info(
        "Enabled shape-aware FlagGems aten.mm: native CUDA for M <= %d, "
        "FlagGems for larger/unsupported shapes",
        threshold,
    )
    return True


__all__ = [
    "DEFAULT_DECODE_MAX_M",
    "ENABLE_ENV",
    "ShapeAwareMMState",
    "THRESHOLD_ENV",
    "_is_native_candidate",
    "_parse_bool_env",
    "_parse_threshold_env",
    "apply_shape_aware_mm",
    "capture_native_mm_kernel",
    "is_mm_dispatch_enabled",
    "is_shape_aware_mm_enabled",
]
