# Copyright (c) 2026 BAAI. All rights reserved.

"""GemmaRMSNorm OOT shim routing rms_norm to FlagGems on PTPU."""

from __future__ import annotations

import logging
import types
from typing import Optional, Union

import torch

logger = logging.getLogger(__name__)


def register_gemma_rms_norm_oot() -> None:
    """Register :class:`GemmaRMSNormFL` as an OOT replacement for
    ``GemmaRMSNorm``. Idempotent. Safe no-op outside PTPU.

    Called from :func:`sunrise.patch.patch_op_cls` after the cross-
    platform OOT layers have been registered by ``register_oot_ops``.
    """
    try:
        from vllm.model_executor.custom_op import CustomOp, op_registry_oot
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
        from vllm.platforms import current_platform

        if getattr(current_platform, "device_type", None) != "ptpu":
            return

        if "GemmaRMSNorm" in op_registry_oot:
            return

        from vllm_fl.dispatch import call_op

        class GemmaRMSNormFL(GemmaRMSNorm):
            """Sunrise OOT replacement for :class:`GemmaRMSNorm`.

            See module docstring for the rationale + numbers.
            """

            def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
                super().__init__(hidden_size, eps)
                self._cached_offset_weight: Optional[torch.Tensor] = None

            def _offset_weight(self) -> torch.Tensor:
                # ``CustomOp.__new__`` may bypass our ``__init__`` if a
                # sibling subclass's ``__init__`` is invoked first
                # (depends on registration order in vLLM internals), so
                # use ``getattr`` for the lazy init.
                cached = getattr(self, "_cached_offset_weight", None)
                weight_data = self.weight.data
                if (
                    cached is None
                    or cached.device != weight_data.device
                    or cached.dtype != weight_data.dtype
                    or cached.shape != weight_data.shape
                ):
                    cached = (weight_data.detach() + 1.0).to(weight_data.dtype)
                    self._cached_offset_weight = cached
                return cached

            def forward_oot(
                self,
                x: torch.Tensor,
                residual: Optional[torch.Tensor] = None,
            ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                # Pretend the ``proxy`` object is a stock ``RMSNorm``-
                # like layer whose ``weight`` already includes Gemma's
                # ``+1`` offset. The sunrise ``rms_norm`` op only reads
                # ``.weight`` and ``.variance_epsilon``.
                proxy = types.SimpleNamespace(
                    weight=self._offset_weight(),
                    variance_epsilon=self.variance_epsilon,
                )
                return call_op("rms_norm", proxy, x, residual)

        CustomOp.register_oot(
            _decorated_op_cls=GemmaRMSNormFL, name="GemmaRMSNorm"
        )
        logger.info(
            "Registered GemmaRMSNormFL as OOT replacement for "
            "GemmaRMSNorm on Sunrise/PTPU"
        )
    except Exception as exc:
        logger.warning(
            "Failed to register GemmaRMSNormFL OOT replacement on "
            "Sunrise/PTPU: %s",
            exc,
        )
