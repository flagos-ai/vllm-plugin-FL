# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend (Huawei) backend for vllm-plugin-FL dispatch.
"""

from .ascend import AscendBackend

__all__ = ["AscendBackend"]
