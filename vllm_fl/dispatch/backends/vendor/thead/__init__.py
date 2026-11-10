# Copyright (c) 2026 BAAI. All rights reserved.

"""
Thead backend for vllm-plugin-FL dispatch.

This backend provides operator implementations for T-Head PPU accelerators.
"""

from .thead import TheadBackend

__all__ = ["TheadBackend"]
