# Copyright (c) 2025 BAAI. All rights reserved.

import os
from typing import Any, Callable

from vllm.logger import init_logger

from vllm_fl.utils import use_flaggems

logger = init_logger("vllm.vllm_fl.envs")
_auto_logged: set[str] = set()

fl_vllm_environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_FL_PREFER_ENABLED": lambda: (
        os.environ.get("VLLM_FL_PREFER_ENABLED", "True").lower() in ("true", "1")
    ),
    "FLAGGEMS_ENABLE_OPLIST_PATH": lambda: os.environ.get(
        "FLAGGEMS_ENABLE_OPLIST_PATH", "/tmp/flaggems_enable_oplist.txt"
    ),
    "USE_FLAGGEMS": use_flaggems,
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in fl_vllm_environment_variables:
        return fl_vllm_environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(fl_vllm_environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in fl_vllm_environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def tristate(name: str) -> bool | None:
    """Explicit value of an optimization switch, or ``None`` if unset/"auto"."""
    value = os.environ.get(name, "auto").strip().lower()
    if value in ("", "auto"):
        return None
    return value in ("1", "true", "yes")


def switch(name: str, auto: bool, reason: str = "") -> bool:
    """Optimization switch: "1"/"true"/"yes" on, "0"/"false"/"no" off; unset or
    "auto" takes the caller's device-generic automatic choice (logged once)."""
    explicit = tristate(name)
    if explicit is not None:
        return explicit
    if name not in _auto_logged:
        _auto_logged.add(name)
        logger.info(
            "%s unset: automatic choice %s%s",
            name,
            "on" if auto else "off",
            f" ({reason})" if reason else "",
        )
    return auto
