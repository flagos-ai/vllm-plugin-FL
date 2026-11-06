# Copyright (c) 2026 BAAI. All rights reserved.

"""
GCU (Enflame) backend for vllm-plugin-FL dispatch.
"""

from .gcu import GCUBackend
from .patch import apply_gcu_patches

apply_gcu_patches()

__all__ = ["GCUBackend"]
