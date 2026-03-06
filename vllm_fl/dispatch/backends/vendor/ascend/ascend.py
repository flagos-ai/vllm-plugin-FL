# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend backend implementation.

This backend provides operator implementations for Huawei Ascend NPUs.
"""

from __future__ import annotations

from typing import Optional

import torch

from vllm_fl.dispatch.backends.base import Backend


class AscendBackend(Backend):
    """
    Ascend backend for operator implementations.

    This backend uses Ascend CANN libraries to provide high-performance
    operator implementations for Huawei Ascend NPUs.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "ascend"

    @property
    def vendor(self) -> Optional[str]:
        return "ascend"

    def is_available(self) -> bool:
        """Check if Ascend hardware and libraries are available."""
        if AscendBackend._available is None:
            try:
                # Check for torch_npu (Ascend PyTorch extension)
                import torch_npu

                # Check if NPU device is available
                if torch.npu.is_available() and torch.npu.device_count() > 0:
                    AscendBackend._available = True
                else:
                    AscendBackend._available = False
            except (ImportError, AttributeError):
                AscendBackend._available = False
        return AscendBackend._available

    # ==================== Operator Implementations ====================

    def attention_backend(self, use_mla: bool = False) -> str:
        """
        Get the attention backend class path for Ascend NPU.

        This method returns the native Ascend attention backend that uses
        torch_npu operators (npu_fused_infer_attention_score, etc.)
        instead of flag_gems operators.

        Uses vllm_fl's native Ascend implementation which directly calls
        torch_npu operators without depending on vllm-ascend package.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string
        """
        if use_mla:
            return "vllm_fl.dispatch.backends.vendor.ascend.impl.attention.AscendMLABackend"
        return "vllm_fl.dispatch.backends.vendor.ascend.impl.attention.AscendAttentionBackend"
