# Copyright (c) 2026 BAAI. All rights reserved.

"""RMSNormGated OOT shim with fused FlagGems path on PTPU."""

from __future__ import annotations

import logging
import types
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def register_rms_norm_gated_oot() -> None:
    """Register :class:`RMSNormGatedFL` as an OOT replacement for
    ``RMSNormGated``. Idempotent. Safe no-op outside PTPU.

    Called from :func:`sunrise.patch.patch_op_cls` after the cross-
    platform OOT layers have been registered by ``register_oot_ops``.
    """
    try:
        from vllm.model_executor.custom_op import CustomOp, op_registry_oot
        from vllm.model_executor.layers.layernorm import RMSNormGated
        from vllm.platforms import current_platform

        if getattr(current_platform, "device_type", None) != "ptpu":
            return

        if "RMSNormGated" in op_registry_oot:
            return

        from vllm_fl.dispatch import call_op

        class RMSNormGatedFL(RMSNormGated):
            """Sunrise OOT replacement for :class:`RMSNormGated`.

            See module docstring for the rationale + numbers.
            """

            def forward_oot(
                self,
                x: torch.Tensor,
                z: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                fast_path = (
                    z is not None
                    and self.norm_before_gate
                    and self.group_size is None
                    and self.activation in ("silu", "swish")
                )
                if not fast_path:
                    # Unsupported configuration (no gate / pre-gate /
                    # group RMS / sigmoid activation) — keep upstream
                    # semantics unchanged.
                    return self.forward_native(x, z)

                # Pretend the ``proxy`` object is a stock RMSNorm
                # whose ``.variance_epsilon`` matches our ``.eps``.
                proxy = types.SimpleNamespace(
                    weight=self.weight,
                    variance_epsilon=self.eps,
                )
                out = call_op("rms_norm", proxy, x, None)
                return out * F.silu(z)

        CustomOp.register_oot(
            _decorated_op_cls=RMSNormGatedFL, name="RMSNormGated"
        )
        logger.info(
            "Registered RMSNormGatedFL as OOT replacement for "
            "RMSNormGated on Sunrise/PTPU"
        )
    except Exception as exc:
        logger.warning(
            "Failed to register RMSNormGatedFL OOT replacement on "
            "Sunrise/PTPU: %s",
            exc,
        )
