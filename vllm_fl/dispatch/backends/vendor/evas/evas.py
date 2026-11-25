# Copyright (c) 2026 BAAI. All rights reserved.
"""Evas (LPU) backend implementation."""

from __future__ import annotations

from typing import Optional

import torch

from vllm_fl.dispatch.backends.base import Backend


# === vLLM enum-slot registration ===
# Populates the SECOND attention-backend registry (the vLLM enum-slot registry),
# distinct from the dispatch OpRegistry. vLLM's Attention.__init__ does
# ``Attention.backend = AttentionBackendEnum[backend.get_name()]``, and the LPU
# backends' get_name() == "custom", so AttentionBackendEnum.CUSTOM must be bound
# to the LPU backend class path. The binding lives in the vendor backend so it
# ships with the backend rather than with the plugin-level files.
_backends_registered = False


def register_attention_backends() -> None:
    """Bind the LPU GQA attention backend under AttentionBackendEnum.CUSTOM.

    Idempotent. No-op off LPU (gated by is_available so the enum slot is left
    untouched on CUDA/NPU/etc.). GQA is the only attention backend that needs
    the enum slot: GLM-5.2 MLA short-circuits before get_attn_backend.
    """
    global _backends_registered
    if _backends_registered:
        return
    if not EvasBackend().is_available():
        return
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "vllm_fl.dispatch.backends.vendor.evas.impl.attention."
        "attention.EvasAttentionBackend",
    )
    _backends_registered = True


class EvasBackend(Backend):
    """Evas LPU backend for operator implementations.

    On evas/LPU, FlagGems is enabled by default for the evas-safe op whitelist
    (see vllm_fl.utils.use_flaggems), and no CUDA vendor backend is available.
    This backend provides the LPU attention backend class path via dispatch.

    Note: "evas" is intentionally absent from the supported-vendor list in
    ``PlatformFL.support_static_graph_mode`` (upstream file -- keep it
    pristine).  No vendor graph-capture path is validated on LPU (all e2e
    scripts run enforce_eager=True), while returning True would let the
    default O2/FULL_AND_PIECEWISE path reach LPU graph capture, which errors
    at warmup.  False forces cudagraph_mode=NONE.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "evas"

    @property
    def vendor(self) -> Optional[str]:
        return "evas"

    def is_available(self) -> bool:
        """Check if EVAS hardware and libraries are available."""
        if EvasBackend._available is None:
            if torch.lpu.is_available() and torch.lpu.device_count() > 0:
                EvasBackend._available = True
            else:
                EvasBackend._available = False
        return EvasBackend._available

    # ==================== Operator Implementations ====================
    def gelu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """GELU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import gelu_and_mul_evas
        return gelu_and_mul_evas(obj, x)

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """Return the LPU attention backend class path.

        GQA path (Qwen3 etc.): SDPA prefill + pure-PyTorch paged-attention decode.
        MLA path: the dense-MLA backend (DSV3, GLM-5.2 etc.) via the upstream
        MLACommon framework (``attention/mla_attention.EvasMLABackend``).

        Also binds the vLLM enum-slot registry (register_attention_backends) on
        first call — mirrors vendor/metax/metax.py:attention_backend calling
        register_attention_backends() before returning the path. Both registries
        must be populated for vLLM's Attention.__init__ to succeed.

        Args:
            use_mla: Whether Multi-head Latent Attention is requested.
            use_sparse: Whether sparse attention is requested (unused on LPU).

        Returns:
            Fully qualified class path string for vLLM's AttentionBackendEnum.
        """
        # Bind AttentionBackendEnum.CUSTOM before returning (metax pattern).
        register_attention_backends()
        if use_mla:
            # Dense MLA (DSV3, GLM-5.2 etc.): run the upstream MLAAttention /
            # MLACommon framework on LPU via a real pure-PyTorch impl (see
            # attention/mla_attention.py).
            return (
                "vllm_fl.dispatch.backends.vendor.evas.impl.attention."
                "mla_attention.EvasMLABackend"
            )
        if use_sparse:
            raise ValueError("use_sparse=True requires use_mla=True.")
        return (
            "vllm_fl.dispatch.backends.vendor.evas.impl.attention."
            "attention.EvasAttentionBackend"
        )

    # ==================== Model-specific entry points ====================
    # These are the PUBLIC API for model files (deepseek_v4.py, etc.) to access
    # vendor-specific infrastructure by name, without importing internals.
    # Each method uses lazy imports internally — the actual impl modules are
    # only loaded on first call, not when EvasBackend is defined.

    def load_weights(self, model, weights):
        """Evas (LPU) weight loading pipeline.

        Thin wrapper around ``load_weights_evas`` (the LPU loader lives with the
        patch that installs it). Handles FP8 reinterpret, routed/shared expert
        weight placement, and AutoWeightsLoader for the remaining weights
        (including the DSV4 attention projection layers).
        """
        from .patches.ops_fallback.dsv4_weights import load_weights_evas
        return load_weights_evas(model, weights)
