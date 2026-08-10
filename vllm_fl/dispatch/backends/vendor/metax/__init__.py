# Copyright (c) 2026 BAAI. All rights reserved.

"""
MACA backend for vllm-plugin-FL dispatch.
"""

from .metax import MacaBackend

__all__ = [
    "MacaBackend",
]

from . import patches  # noqa: F401 — apply MetaX kernel patches at backend load time
