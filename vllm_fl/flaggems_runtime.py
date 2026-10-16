# Copyright (c) 2026 BAAI. All rights reserved.
"""Process-wide FlagGems initialization after policy resolution."""
from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True)
class FlagGemsConfig:
    whitelist: tuple[str, ...] | None
    blacklist: tuple[str, ...] | None


_STATE: FlagGemsConfig | None = None
_FAILED = False
_LOCK = RLock()


def configure_flaggems(
    enable_flaggems, *, use_flaggems=True, whitelist=None, blacklist=None
) -> None:
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
            return
        config = FlagGemsConfig(
            tuple(sorted(whitelist)) if whitelist is not None else None,
            tuple(sorted(blacklist)) if blacklist is not None else None,
        )
        if _STATE is not None:
            if config != _STATE:
                raise RuntimeError(
                    "FlagGems configuration is process-lifetime; restart to change it"
                )
            return
        try:
            enable_flaggems(None)
            _STATE = config
        except Exception:
            _FAILED = True
            raise
