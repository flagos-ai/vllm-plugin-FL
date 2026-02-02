
# Copyright (c) 2025 BAAI. All rights reserved.

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

fl_vllm_environment_variables: dict[str, Callable[[], Any]] = {
    # path to the logs of redirect-output, abstrac of related are ok
    "USE_FLAGGEMS":
    lambda: (os.environ.get("USE_FLAGGEMS", "True").lower() in
            ("true", "1")),
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
