# Copyright (c) 2026 BAAI. All rights reserved.

"""
AMD backend implementation (ROCm / HIP).

AMD GPUs (Instinct MI-series, Radeon PRO W7900D / gfx1100 RDNA3) expose a
CUDA-compatible surface through ROCm: ``torch.cuda`` maps onto HIP and
FlagGems reports ``vendor_name == "amd"`` with ``device.name == "cuda"``.

The plugin runs FlagGems (Triton via FlagTree) for the element-wise and
normalization operators, so this vendor backend only supplies the
attention-backend selection. We pick the Triton attention backends that run on
the FlagTree Triton compiler present on this hardware, avoiding the ROCm AITER /
Composable-Kernel and prebuilt flash-attention paths that are not part of the
validated FlagOS stack.
"""

from __future__ import annotations

import logging

import torch

from vllm_fl.dispatch.backends.base import Backend

logger = logging.getLogger(__name__)


class AMDBackend(Backend):
    """Vendor backend for AMD ROCm GPUs."""

    _available: bool | None = None

    @property
    def name(self) -> str:
        return "amd"

    @property
    def vendor(self) -> str | None:
        return "amd"

    def is_available(self) -> bool:
        """Return True when running on an AMD ROCm device.

        Detection is driven by the plugin platform's ``vendor_name`` (sourced
        from FlagGems), with a fallback probe of ``torch.version.hip`` so the
        backend is still usable if the platform shim is not yet constructed.
        """
        if AMDBackend._available is None:
            try:
                from vllm.platforms import current_platform

                vendor = getattr(current_platform, "vendor_name", None)
                if vendor == "amd":
                    AMDBackend._available = True
                else:
                    # Fallback: ROCm torch reports a non-None torch.version.hip.
                    AMDBackend._available = (
                        bool(getattr(torch.version, "hip", None))
                        and torch.cuda.is_available()
                    )
            except Exception:
                AMDBackend._available = False
        return AMDBackend._available

    # ==================== Operator Implementations ====================

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """Return the attention backend class path for AMD.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA).
            use_sparse: Whether to use DeepSeek Sparse Attention (DSA).

        Returns:
            Fully qualified attention backend class path string.

        We use the Triton backends (``TRITON_MLA`` for MLA models,
        ``TRITON_ATTN`` otherwise) because they run on the FlagTree Triton
        compiler validated on gfx1100. Sparse attention is not validated on
        this hardware yet; fall back to the dense Triton MLA path.
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if use_mla:
            return AttentionBackendEnum.TRITON_MLA.get_path()
        return AttentionBackendEnum.TRITON_ATTN.get_path()
