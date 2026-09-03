# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal connectivity test for the npu_chunk_gated_delta_rule framework op.
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

# Load the Ascend C++ extension that registers torch.ops._C_ascend.
import vllm_fl._C_ascend  # noqa: F401

DEVICE = "npu:0"


def test_chunk_gated_delta_rule_basic():
    """Basic test for npu_chunk_gated_delta_rule operator.

    The operator expects TND layout inputs:
    - query: (T, Nk, Dk) - total tokens, num key heads, key head dim
    - key: (T, Nk, Dk)
    - value: (T, Nv, Dv) - num value heads, value head dim
    - beta: (T, Nv) - beta values
    - initial_state: (B, Nv, Dk, Dv) - batch, transposed state
    - actual_seq_lengths: (B,) int32 - actual sequence lengths per batch
    - g: (T, Nv) optional - log-gate values
    - scale_value: float - attention scale

    Returns:
    - out: (T, Nv, Dv) bf16
    - final_state: (B, Nv, Dv, Dk)
    """
    batch = 2
    seqlen_per_batch = 8
    total_seqlen = batch * seqlen_per_batch
    num_heads = 4
    head_dim = 64

    # Input tensors in TND layout (T, N, D)
    # The operator requires bfloat16 inputs
    query = torch.randn(total_seqlen, num_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    key = torch.randn(total_seqlen, num_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    value = torch.randn(total_seqlen, num_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    beta = torch.randn(total_seqlen, num_heads, dtype=torch.bfloat16, device=DEVICE)

    # L2-normalize query and key as the op expects (mentioned in PR description)
    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)

    # Initial state: (B, Nv, Dk, Dv) - can be bf16 or fp32
    initial_state = torch.zeros(batch, num_heads, head_dim, head_dim, dtype=torch.bfloat16, device=DEVICE)

    # Actual sequence lengths per batch element (int32)
    actual_seq_lengths = torch.tensor([seqlen_per_batch] * batch, dtype=torch.int32, device=DEVICE)

    # Optional g tensor (log-gate values) - fp32
    g = torch.randn(total_seqlen, num_heads, dtype=torch.float32, device=DEVICE)

    # Scale value (typically 1/sqrt(head_dim))
    scale = head_dim ** -0.5

    # Call the operator
    out, final_state = torch.ops._C_ascend.npu_chunk_gated_delta_rule(
        query,
        key,
        value,
        beta,
        initial_state,
        actual_seq_lengths,
        g,
        scale,
    )

    # Check output shapes
    assert out.shape == (total_seqlen, num_heads, head_dim), f"Expected out shape {(total_seqlen, num_heads, head_dim)}, got {out.shape}"
    assert final_state.shape == (batch, num_heads, head_dim, head_dim), f"Expected final_state shape {(batch, num_heads, head_dim, head_dim)}, got {final_state.shape}"

    # Check output dtype (should be bf16 according to PR)
    assert out.dtype == torch.bfloat16, f"Expected out dtype bfloat16, got {out.dtype}"

    # Check device
    assert out.device.type == "npu"
    assert final_state.device.type == "npu"

    print(f"✓ Output shape: {out.shape}, dtype: {out.dtype}")
    print(f"✓ Final state shape: {final_state.shape}, dtype: {final_state.dtype}")


def test_chunk_gated_delta_rule_no_g():
    """Test without optional g parameter."""
    batch = 1
    seqlen = 16
    num_heads = 2
    head_dim = 32

    query = torch.randn(seqlen, num_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    key = torch.randn(seqlen, num_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    value = torch.randn(seqlen, num_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    beta = torch.randn(seqlen, num_heads, dtype=torch.bfloat16, device=DEVICE)

    query = torch.nn.functional.normalize(query, p=2, dim=-1)
    key = torch.nn.functional.normalize(key, p=2, dim=-1)

    initial_state = torch.zeros(batch, num_heads, head_dim, head_dim, dtype=torch.bfloat16, device=DEVICE)
    actual_seq_lengths = torch.tensor([seqlen], dtype=torch.int32, device=DEVICE)

    scale = head_dim ** -0.5

    # Call without g parameter
    out, final_state = torch.ops._C_ascend.npu_chunk_gated_delta_rule(
        query,
        key,
        value,
        beta,
        initial_state,
        actual_seq_lengths,
        None,  # g is optional
        scale,
    )

    assert out.shape == (seqlen, num_heads, head_dim)
    assert final_state.shape == (batch, num_heads, head_dim, head_dim)
    assert out.dtype == torch.bfloat16

    print(f"✓ Test without g passed: out shape {out.shape}")


if __name__ == "__main__":
    print("Testing npu_chunk_gated_delta_rule operator...")
    print("-" * 60)

    test_chunk_gated_delta_rule_basic()
    print("✓ Basic test passed")
    print()

    test_chunk_gated_delta_rule_no_g()
    print("✓ Test without g passed")
    print()

    print("-" * 60)
    print("All tests passed!")
