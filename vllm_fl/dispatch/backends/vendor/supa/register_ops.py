# Copyright (c) 2026 BAAI. All rights reserved.

"""Biren SUPA vendor backend registrations."""

from __future__ import annotations


def register_builtins(registry) -> None:
    """Register SUPA implementations when vendor kernels are added."""
    registry.register_many([])
