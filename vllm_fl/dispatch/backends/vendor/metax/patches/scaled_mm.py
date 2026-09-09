# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassInt8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)


class MctlassInt8ScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(
        cls, config: Int8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        return True, None
