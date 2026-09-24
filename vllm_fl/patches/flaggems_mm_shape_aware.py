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
class MMStatus:
    status: str
    reason: str
    native: str | None = None
    flaggems: str | None = None


@dataclass(frozen=True)
class ShapeAwareMMState:
    """Retain the MM handles and Libraries for the process lifetime."""

    result: MMStatus
    native_mm: Any
    flaggems_mm: Any
    flaggems_library: Any
    library: Any
    registration: tuple[str, ...] | None


def _parse_bool_env(name: str, *, default: bool = False) -> bool:
    """Parse an override without accepting ambiguous values."""

    value = os.environ.get(name)
    if value is None:
        return default
    if value == "1" or value == "true":
        return True
    if value == "0" or value == "false":
        return False
    raise ValueError(f"{name} must be exactly one of 0, 1, false, true; got {value!r}")


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
    """Return whether plugin policy requests FlagGems ``aten.mm``.

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


def _registration_fingerprint() -> tuple[str, ...] | None:
    # PyTorch 2.11 has no public kernel identity API. Restrict its diagnostic
    # registration-stack adapter to the tested ABI; repr/source filenames are
    # never implementation identity. Other builds can install through the
    # public API, but repeated initialization cannot verify external ownership.
    if torch.__version__.split("+", 1)[0].split(".")[:2] != ["2", "11"]:
        return None
    return tuple(
        line
        for line in torch._C._dispatch_dump("aten::mm").splitlines()
        if line.startswith("CUDA:") or line.startswith("CUDA (inactive):")
    )


class _ObservedFlagGemsLibrary(torch.library.Library):
    """Observe successful registration, after FlagGems' own filtering.

    This is a Library passed through FlagGems' public lib= parameter, not a
    global monkeypatch. Retain the exact callable and boxed handle installed
    by that registration rather than inferring ownership from a whitelist.
    """

    def __init__(self):
        super().__init__("aten", "IMPL")
        self.mm = None
        self.mm_callable = None
        self.mm_registration = ()

    def impl(
        self, op_name, fn, dispatch_key="", *, with_keyset=False, allow_override=False
    ):
        super().impl(
            op_name,
            fn,
            dispatch_key,
            with_keyset=with_keyset,
            allow_override=allow_override,
        )
        if op_name in ("mm", "aten::mm") and dispatch_key == "CUDA":
            self.mm = _get_registered_mm_kernel()
            self.mm_callable = fn
            self.mm_registration = _registration_fingerprint()


def apply_shape_aware_mm(
    native: Any, gems_lib: _ObservedFlagGemsLibrary, threshold: int
) -> ShapeAwareMMState:
    """Install only the MM selector after observed FlagGems registration."""
    gems = gems_lib.mm
    if gems is None:
        raise RuntimeError("FlagGems did not register CUDA mm through lib=")
    if (
        gems_lib.mm_registration is not None
        and gems_lib.mm_registration != _registration_fingerprint()
    ):
        raise RuntimeError("CUDA mm changed after the observed FlagGems registration")

    def shape_aware_mm(dispatch_keys, a, b):
        target = native if _is_native_candidate(a, b, threshold) else gems
        return target.call_boxed(dispatch_keys, a, b)

    library = torch.library.Library("aten", "IMPL")
    _register_override(library, shape_aware_mm)
    fn = gems_lib.mm_callable
    # Names and source locations are diagnostic only: vendor loaders and
    # packaged distributions need not retain the flag_gems module/path prefix.
    result = MMStatus(
        "installed",
        f"native CUDA for M <= {threshold}; FlagGems otherwise",
        repr(native),
        f"{getattr(fn, '__module__', '')}.{getattr(fn, '__name__', '')}",
    )
    return ShapeAwareMMState(
        result, native, gems, gems_lib, library, _registration_fingerprint()
    )
