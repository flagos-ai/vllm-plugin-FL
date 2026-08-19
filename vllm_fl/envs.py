# Copyright (c) 2025 BAAI. All rights reserved.

import logging
import os
from typing import Any, Callable

from vllm_fl.utils import use_flaggems

logger = logging.getLogger(__name__)

_FLAGGEMS_VLLM_STATUS_LOGGED = False

fl_vllm_environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_FL_PREFER_ENABLED": lambda: (
        os.environ.get("VLLM_FL_PREFER_ENABLED", "True").lower() in ("true", "1")
    ),
    "FLAGGEMS_ENABLE_OPLIST_PATH": lambda: os.environ.get(
        "FLAGGEMS_ENABLE_OPLIST_PATH", "/tmp/flaggems_enable_oplist.txt"
    ),
    "USE_FLAGGEMS": use_flaggems,
    "VLLM_FL_USE_FLAGGEMS_VLLM": lambda: (
        os.environ.get("VLLM_FL_USE_FLAGGEMS_VLLM", "1").lower() in ("1", "true")
    ),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in fl_vllm_environment_variables:
        value = fl_vllm_environment_variables[name]()
        if name == "VLLM_FL_USE_FLAGGEMS_VLLM":
            global _FLAGGEMS_VLLM_STATUS_LOGGED
            if not _FLAGGEMS_VLLM_STATUS_LOGGED:
                _FLAGGEMS_VLLM_STATUS_LOGGED = True
                backend = "flaggems_vllm" if value else "flag_gems"
                logger.info(
                    "VLLM_FL_USE_FLAGGEMS_VLLM=%s, using %s for non-aten ops",
                    os.environ.get("VLLM_FL_USE_FLAGGEMS_VLLM", "1"),
                    backend,
                )
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(fl_vllm_environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in fl_vllm_environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
