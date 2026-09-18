# Copyright (c) 2026 BAAI. All rights reserved.

"""
ILUVATAR backend for vllm-plugin-FL dispatch.
"""

from .iluvatar import IluvatarBackend

__all__ = ["IluvatarBackend"]

from . import patches  # noqa: F401 — apply iluvatar kernel patches at backend load time
