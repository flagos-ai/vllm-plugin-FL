# Copyright (c) 2026 BAAI. All rights reserved.

"""
GCU backend implementation.
"""

from __future__ import annotations
from typing import Optional, Union
import sys
import torch
from vllm_fl.dispatch.backends.base import Backend


class GCUBackend(Backend):
    """GCU vendor backend (``torch.gcu`` / torch_gcu runtime)."""

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "gcu"

    @property
    def vendor(self) -> Optional[str]:
        return "gcu"

    def is_available(self) -> bool:
        if GCUBackend._available is None:
            gcu = getattr(torch, "gcu", None)
            if gcu is not None and gcu.is_available() and gcu.device_count() > 0:
                GCUBackend._available = True
            else:
                GCUBackend._available = False
        return GCUBackend._available

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        from .impl.activation import silu_and_mul_gcu

        return silu_and_mul_gcu(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.normalization import rms_norm_gcu

        return rms_norm_gcu(obj, x, residual)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .impl.rotary import rotary_embedding_gcu

        return rotary_embedding_gcu(
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

        if use_mla:
            if use_sparse:
                raise NotImplementedError("GCU does not support sparse attention yet")
            raise NotImplementedError("GCU does not support MLA yet")

        import flash_attn.vllm_flash_attn

        sys.modules["vllm.vllm_flash_attn"] = flash_attn.vllm_flash_attn

        return AttentionBackendEnum.FLASH_ATTN.get_path()
