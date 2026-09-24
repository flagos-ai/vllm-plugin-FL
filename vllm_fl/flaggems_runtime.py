# Copyright (c) 2026 BAAI. All rights reserved.
"""Process-wide FlagGems initialization after platform/model policy resolution."""

from dataclasses import dataclass
from threading import RLock

from vllm_fl.patches import flaggems_mm_shape_aware as mm


@dataclass(frozen=True)
class FlagGemsConfig:
    whitelist: tuple[str, ...] | None
    blacklist: tuple[str, ...] | None
    mm_threshold: int | None


@dataclass(frozen=True)
class FlagGemsRuntime:
    config: FlagGemsConfig
    mm_state: mm.ShapeAwareMMState | None


_STATE: FlagGemsRuntime | None = None
_FAILED = False
_LOCK = RLock()


def configure_flaggems(
    enable_flaggems,
    *,
    use_flaggems=True,
    whitelist=None,
    blacklist=None,
    default_mm_enabled=False,
) -> mm.MMStatus:
    """Initialize once. Changing an active process policy requires a restart.

    The worker resolves vendor, model and deployment exclusions before this
    call. Optional MM settings are interpreted only when MM is requested.
    No dispatcher inspection or policy checks run in the inference hot path.
    """
    global _STATE, _FAILED
    with _LOCK:
        if _FAILED:
            raise RuntimeError(
                "FlagGems initialization previously failed; restart the process"
            )
        if not use_flaggems:
            if _STATE is not None:
                raise RuntimeError(
                    "FlagGems configuration is process-lifetime; restart to disable it"
                )
            return mm.MMStatus("disabled", "FlagGems disabled by worker policy")

        active_mm = mm.is_mm_dispatch_enabled(
            whitelist, blacklist
        ) and mm.is_shape_aware_mm_enabled(default=default_mm_enabled)
        config = FlagGemsConfig(
            tuple(sorted(whitelist)) if whitelist is not None else None,
            tuple(sorted(blacklist)) if blacklist is not None else None,
            mm._parse_threshold_env() if active_mm else None,
        )
        if _STATE is not None:
            if config != _STATE.config:
                raise RuntimeError(
                    "FlagGems/MM configuration is process-lifetime; restart to change it"
                )
            state = _STATE.mm_state
            if state is not None:
                if (
                    state.registration is not None
                    and state.registration != mm._registration_fingerprint()
                ):
                    raise RuntimeError(
                        "conflicting_owner: aten::mm/CUDA registration changed; restart the process"
                    )
                return mm.MMStatus(
                    "already_active",
                    state.result.reason,
                    state.result.native,
                    state.result.flaggems,
                )
            return mm.MMStatus("disabled", "shape-aware MM disabled or mm excluded")

        try:
            native = mm.capture_native_mm_kernel() if active_mm else None
            library = mm._ObservedFlagGemsLibrary() if active_mm else None
            enable_flaggems(library)
            state = (
                mm.apply_shape_aware_mm(native, library, config.mm_threshold)
                if active_mm
                else None
            )
            _STATE = FlagGemsRuntime(config, state)
            return (
                state.result
                if state is not None
                else mm.MMStatus("disabled", "shape-aware MM disabled or mm excluded")
            )
        except Exception:
            # A partially registered dispatcher must never be recaptured as
            # the original backend by a retry in the same process.
            _FAILED = True
            raise
