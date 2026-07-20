# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal connectivity test for the npu_fused_gdn_gating framework op.
#
# NOTE: this test automatically enables the CANN custom-op environment by
# discovering the vllm_fl/_cann_ops_custom package installed next to vllm_fl.
# No manual `source set_env.bash` is required.

import sys

from vllm_fl.utils import enable_custom_op

if not enable_custom_op():
    print(
        "ERROR: vllm_fl/_cann_ops_custom is not installed.\n"
        "Please build and install the CANN framework operators first, e.g.:\n"
        "  bash csrc/ascend/build_aclnn.sh <soc_version>",
        file=sys.stderr,
    )
    sys.exit(1)

import torch
import torch_npu

import vllm_fl._C_ascend  # noqa: F401

DEVICE = "npu:0"


def test_fused_gdn_gating():
    """Basic fused GDN gating connectivity test."""
    batch = 4
    num_heads = 8

    A_log = torch.randn(num_heads, dtype=torch.bfloat16, device=DEVICE)
    a = torch.randn(batch, num_heads, dtype=torch.bfloat16, device=DEVICE)
    b = torch.randn(batch, num_heads, dtype=torch.bfloat16, device=DEVICE)
    dt_bias = torch.randn(num_heads, dtype=torch.bfloat16, device=DEVICE)

    g, beta_output = torch.ops._C_ascend.npu_fused_gdn_gating(
        A_log, a, b, dt_bias, beta=1.0, threshold=20.0
    )

    assert g.shape == (1, batch, num_heads)
    assert beta_output.shape == (1, batch, num_heads)
    assert g.device == A_log.device


if __name__ == "__main__":
    test_fused_gdn_gating()
    print("npu_fused_gdn_gating test passed")
