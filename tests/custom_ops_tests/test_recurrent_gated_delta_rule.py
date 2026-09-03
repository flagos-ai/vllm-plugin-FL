# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal connectivity test for the npu_recurrent_gated_delta_rule framework op.
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


def test_recurrent_gated_delta_rule():
    """Basic recurrent gated delta rule connectivity test."""
    t = 16
    nk = 2
    nv = 4  # must be a multiple of nk
    dk = 32
    dv = 32
    batch = 2

    query = torch.randn(t, nk, dk, dtype=torch.bfloat16, device=DEVICE)
    key = torch.randn(t, nk, dk, dtype=torch.bfloat16, device=DEVICE)
    value = torch.randn(t, nv, dv, dtype=torch.bfloat16, device=DEVICE)
    beta = torch.randn(t, nv, dtype=torch.bfloat16, device=DEVICE)
    state = torch.randn(batch, nv, dv, dk, dtype=torch.bfloat16, device=DEVICE)

    actual_seq_lengths = torch.full((batch,), t, dtype=torch.int32, device=DEVICE)
    ssm_state_indices = torch.arange(batch, dtype=torch.int32, device=DEVICE)

    output = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        beta=beta,
        scale=dk ** -0.5,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
    )

    assert output.shape == value.shape
    assert output.device == value.device


if __name__ == "__main__":
    test_recurrent_gated_delta_rule()
    print("npu_recurrent_gated_delta_rule test passed")
