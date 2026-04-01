# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm.model_executor.layers.layernorm import rms_norm, fused_add_rms_norm

import torch


def rms_norm_maca(
    obj,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    RMS normalization using Maca's CUDA implementation.
    """
    add_residual = residual is not None
    if add_residual:
        return fused_add_rms_norm(x, residual, obj.weight, obj.epsilon)
    else:
        return rms_norm(x, obj.weight, obj.epsilon)
