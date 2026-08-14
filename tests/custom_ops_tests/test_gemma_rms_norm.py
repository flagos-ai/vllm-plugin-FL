# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal connectivity test for the npu_gemma_rms_norm framework op.
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


def test_gemma_rms_norm():
    """Basic GemmaRMSNorm connectivity test."""
    x = torch.randn(16, 128, dtype=torch.float16, device=DEVICE)
    gamma = torch.randn(128, dtype=torch.float16, device=DEVICE)

    y, rstd = torch.ops._C_ascend.npu_gemma_rms_norm(x, gamma, 1e-6)

    assert y.shape == x.shape
    assert y.device == x.device
    assert rstd.shape == (x.size(0), 1)


if __name__ == "__main__":
    test_gemma_rms_norm()
    print("npu_gemma_rms_norm test passed")
