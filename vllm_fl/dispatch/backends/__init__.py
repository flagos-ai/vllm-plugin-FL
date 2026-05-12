# Copyright (c) 2026 BAAI. All rights reserved.

"""
Backend implementations for vllm-plugin-FL dispatch.

Vendor backends are dynamically discovered and loaded by builtin_ops.py
based on the current platform. This package does not eagerly import vendor
backends to avoid loading unnecessary dependencies at startup.
"""

from .base import Backend
from .flaggems import FlagGemsBackend
from .reference import ReferenceBackend

__all__ = ["Backend", "FlagGemsBackend", "ReferenceBackend"]
