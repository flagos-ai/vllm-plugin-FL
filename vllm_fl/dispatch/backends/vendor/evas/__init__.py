# Copyright (c) 2026 BAAI. All rights reserved.
"""
Evas (LPU) vendor backend for vllm-plugin-FL dispatch.

Targets the evas LPU device. Provides the LPU attention backend (SDPA prefill +
pure-PyTorch paged-attention decode) and aggregates the LPU compatibility
patches declared in ``patches/registry.py`` under apply_evas_patches().
"""

from .evas import EvasBackend

__all__ = ["EvasBackend"]
