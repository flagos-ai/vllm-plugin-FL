# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend (Huawei) backend for vllm-plugin-FL dispatch.
"""

from .ascend import AscendBackend
from .patch import patch_mamba_config
from .patches.patch_qwen3_mtp import patch_qwen3_mtp_platform

patch_mamba_config()
patch_qwen3_mtp_platform()

__all__ = ["AscendBackend"]
