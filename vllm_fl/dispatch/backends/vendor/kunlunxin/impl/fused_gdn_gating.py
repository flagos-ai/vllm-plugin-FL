# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# 2026 - Modified by Kunlunxin, Inc. All Rights Reserved.

"""
Kunlunxin implementation of fused_gdn_gating.
"""

from __future__ import annotations

import torch
import xtorch_ops


def fused_gdn_gating_kunlunxin(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Kunlunxin fused GDN gating computation.

    The xtorch_ops kernel computes g = -exp(A_log) * softplus(a + dt_bias).
    beta_output = sigmoid(b) is computed separately in PyTorch.

    Args:
        A_log: log of A parameter [num_heads]
        a: dt input [batch, num_heads]
        b: beta input [batch, num_heads]
        dt_bias: dt bias [num_heads]
        beta: softplus beta parameter
        threshold: softplus threshold parameter

    Returns:
        (g, beta_output) where:
            g: [1, batch, num_heads] float32 gating values
            beta_output: [1, batch, num_heads] sigmoid of b
    """
    org_shape = list(a.shape)

    # xtorch_ops.fused_gdn_gating returns a new tensor (not in-place)
    # It computes: output = -exp(A_log) * softplus(a + dt_bias, beta, threshold)
    output = xtorch_ops.fused_gdn_gating(A_log, a, dt_bias, beta, threshold)

    # Reshape to match expected output format [1, batch, num_heads]
    g = output.view(*org_shape).unsqueeze(0)

    # beta_output = sigmoid(b), computed separately
    beta_output = b.sigmoid().unsqueeze(0)

    return g, beta_output
