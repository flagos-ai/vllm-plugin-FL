"""Hygon interconnect communication implementations."""

from .custom_all_reduce import ensure_custom_allreduce_ops

__all__ = ["ensure_custom_allreduce_ops"]
