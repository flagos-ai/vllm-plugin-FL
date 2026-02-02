# Copyright (c) 2026 BAAI. All rights reserved.

"""
CUDA backend for vllm-plugin-FL dispatch.
"""

from .cuda import CudaBackend

__all__ = ["CudaBackend"]
