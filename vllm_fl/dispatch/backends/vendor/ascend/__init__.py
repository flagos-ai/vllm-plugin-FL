# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend (Huawei) backend for vllm-plugin-FL dispatch.
"""

from .ascend import AscendBackend
from .patch import patch_mamba_config
from .patches.accelerator_compat import patch_accelerator_memory

patch_accelerator_memory()
patch_mamba_config()

__all__ = ["AscendBackend"]
