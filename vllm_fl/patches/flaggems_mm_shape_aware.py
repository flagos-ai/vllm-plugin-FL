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
from pathlib import Path
from threading import RLock
from typing import Any

import torch

logger = logging.getLogger(__name__)

ENABLE_ENV = "VLLM_FL_FLAGOS_MM_SHAPE_AWARE"
THRESHOLD_ENV = "VLLM_FL_FLAGOS_MM_DECODE_MAX_M"
DEFAULT_DECODE_MAX_M = 64


@dataclass(frozen=True)
class MMConfig:
    use_flaggems: bool
    enabled: bool
    threshold: int
    whitelist: tuple[str, ...] | None
    blacklist: tuple[str, ...] | None


@dataclass(frozen=True)
class MMStatus:
    status: str
    reason: str
    native: str | None = None
    flaggems: str | None = None


@dataclass(frozen=True)
class ShapeAwareMMState:
    """One immutable worker policy and its retained dispatcher registrations.

    Shutdown does not uninstall process-wide kernels. A subsequent worker
    must use identical configuration and retain dispatcher ownership; changing
    enable, threshold, or backend selection requires a new process.
    """

    config: MMConfig
    result: MMStatus
    native_mm: Any = None
    flaggems_mm: Any = None
    flaggems_library: Any = None
    library: Any = None
    registration: tuple[str, ...] = ()


_STATE: ShapeAwareMMState | None = None
_FAILED = False
_LOCK = RLock()


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


def _registration_fingerprint() -> tuple[str, ...]:
    # Include active AND shadowed registrations: two Libraries constructed at
    # the same source line still produce a different registration stack.
    return tuple(
        line
        for line in torch._C._dispatch_dump("aten::mm").splitlines()
        if line.startswith("CUDA:") or line.startswith("CUDA (inactive):")
    )


def _flaggems_source(fn: Any) -> str | None:
    """Verify code provenance, including vendor modules loaded as hopper.*.

    FlagGems architecture loaders need not preserve the flag_gems module-name
    prefix. Follow Python wrappers to the registered implementation's source
    and require that it belongs to the loaded FlagGems package instead.
    """
    import flag_gems

    try:
        source = inspect.getsourcefile(inspect.unwrap(fn))
        if source is None:
            return None
        path = Path(source).resolve()
        root = Path(flag_gems.__file__).resolve().parent
        return str(path) if path.is_relative_to(root) else None
    except (TypeError, ValueError):
        return None


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


def configure_flaggems_mm(
    enable_flaggems: Callable[[Any], None],
    *,
    use_flaggems: bool = True,
    whitelist: list[str] | None = None,
    blacklist: list[str] | None = None,
    default_enabled: bool = False,
) -> MMStatus:
    """Initialize FlagGems once and optionally install shape-aware CUDA mm.

    Repeated worker initialization checks configuration and ownership BEFORE
    invoking FlagGems again. This prevents capturing our wrapper as the native
    backend or letting a second enable() overwrite it. No per-mm-call checks,
    tensor reads, or synchronization are added to inference/capture.
    """
    global _STATE, _FAILED
    config = MMConfig(
        use_flaggems,
        is_shape_aware_mm_enabled(default=default_enabled),
        _parse_threshold_env(),
        tuple(sorted(whitelist)) if whitelist is not None else None,
        tuple(sorted(blacklist)) if blacklist is not None else None,
    )
    with _LOCK:
        if _FAILED:
            raise RuntimeError(
                "FlagGems initialization previously failed; restart the process"
            )
        if _STATE is not None:
            if config != _STATE.config:
                raise RuntimeError(
                    "FlagGems/MM configuration is process-lifetime; restart to change it"
                )
            if _STATE.library is not None:
                if _registration_fingerprint() != _STATE.registration:
                    raise RuntimeError(
                        "conflicting_owner: aten::mm/CUDA registration changed; restart the process"
                    )
                return MMStatus(
                    "already_active",
                    _STATE.result.reason,
                    _STATE.result.native,
                    _STATE.result.flaggems,
                )
            return _STATE.result

        active = (
            config.use_flaggems
            and config.enabled
            and is_mm_dispatch_enabled(whitelist, blacklist)
        )
        if not active:
            if config.use_flaggems:
                try:
                    enable_flaggems(None)
                except Exception:
                    _FAILED = True
                    raise
            result = MMStatus("disabled", "policy disabled or mm excluded")
            _STATE = ShapeAwareMMState(config, result)
            return result

        try:
            native = capture_native_mm_kernel()
            if "RegisterCUDA" not in repr(native):
                raise RuntimeError(
                    "Expected native CUDA mm before FlagGems initialization"
                )
            gems_lib = _ObservedFlagGemsLibrary()
            enable_flaggems(gems_lib)
            gems = gems_lib.mm
            fn = gems_lib.mm_callable
            source = _flaggems_source(fn)
            if (
                gems is None
                or source is None
                or gems_lib.mm_registration != _registration_fingerprint()
                or repr(gems) == repr(native)
            ):
                raise RuntimeError(
                    "FlagGems did not install the expected distinct CUDA mm kernel"
                )

            def shape_aware_mm(dispatch_keys, a, b):
                target = (
                    native if _is_native_candidate(a, b, config.threshold) else gems
                )
                return target.call_boxed(dispatch_keys, a, b)

            library = torch.library.Library("aten", "IMPL")
            _register_override(library, shape_aware_mm)
            registration = _registration_fingerprint()
            if registration == gems_lib.mm_registration:
                raise RuntimeError("Shape-aware mm override was not installed")
            result = MMStatus(
                "installed",
                f"native CUDA for M <= {config.threshold}; FlagGems otherwise",
                repr(native),
                f"{fn.__module__}.{fn.__name__} ({source}): {gems!r}",
            )
            _STATE = ShapeAwareMMState(
                config, result, native, gems, gems_lib, library, registration
            )
            logger.info(
                "Shape-aware MM status=%s policy=%s native=%s flaggems=%s",
                result.status,
                result.reason,
                result.native,
                result.flaggems,
            )
            return result
        except Exception:
            # Registration can have side effects; retries in this process must
            # never reinterpret a partially initialized backend as native.
            _FAILED = True
            raise
