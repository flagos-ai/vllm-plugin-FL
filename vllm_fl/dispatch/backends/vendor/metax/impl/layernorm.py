# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import torch


def rms_norm_maca(
    obj,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    RMS normalization using Maca's CUDA implementation (via mcoplib._C ops).
    """
    if residual is not None:
        torch.ops._C.fused_add_rms_norm(x, residual, obj.weight.data,
                                        obj.variance_epsilon)
        return x, residual
    else:
        out = torch.empty_like(x)
        torch.ops._C.rms_norm(out, x, obj.weight.data, obj.variance_epsilon)
        return out
