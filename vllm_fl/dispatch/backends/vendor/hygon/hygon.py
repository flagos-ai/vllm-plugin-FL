"""Hygon vendor backend for vLLM-FL."""

from __future__ import annotations

import os
import shutil
from typing import Optional

import torch

from vllm_fl.dispatch.backends.base import Backend


class HygonBackend(Backend):
    """Operators and backend selection for Hygon BW-series devices."""

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "hygon"

    @property
    def vendor(self) -> Optional[str]:
        return "hygon"

    def is_available(self) -> bool:
        if HygonBackend._available is not None:
            return HygonBackend._available

        try:
            from vllm.platforms import current_platform

            if getattr(current_platform, "vendor_name", "").lower() == "hygon":
                HygonBackend._available = True
                return True
        except Exception:
            pass

        if os.environ.get("GEMS_VENDOR", "").strip().lower() == "hygon":
            HygonBackend._available = True
            return True

        # Some BW1000 images expose both management commands.
        HygonBackend._available = bool(shutil.which("hy-smi") and shutil.which("rocm-smi"))
        return HygonBackend._available

    def rms_norm(self, obj, x: torch.Tensor, residual=None):
        from .impl.normalization import rms_norm_hygon

        return rms_norm_hygon(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ):
        from .impl.rotary import rotary_embedding_hygon

        return rotary_embedding_hygon(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if use_mla and use_sparse:
            return AttentionBackendEnum.ROCM_AITER_MLA_SPARSE.get_path()
        if use_mla:
            try:
                from vllm._aiter_ops import rocm_aiter_ops

                if rocm_aiter_ops.is_mla_enabled():
                    return AttentionBackendEnum.ROCM_AITER_MLA.get_path()
            except (ImportError, AttributeError):
                pass
            return AttentionBackendEnum.TRITON_MLA.get_path()
        if use_sparse:
            raise ValueError("use_sparse=True requires use_mla=True")
        return AttentionBackendEnum.ROCM_ATTN.get_path()
