# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend (Huawei) backend for vllm-plugin-FL dispatch.
"""

from .ascend import AscendBackend
from .patch import patch_mamba_config

patch_mamba_config()

__all__ = ["AscendBackend"]
