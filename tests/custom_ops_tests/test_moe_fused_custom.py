# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Correctness test for the AscendC fused-MoE custom ops used by the FL MoE path:
#   * torch.ops._C_ascend.moe_gating_top_k          (fused router: softmax + topk + renorm)
#   * torch.ops._C_ascend.npu_moe_init_routing_custom  (token expand/sort by expert)
#
# plus an end-to-end parity check of the new `_ascendc_fused_experts_impl`
# (custom routing + pre-transposed weights) against the legacy
# `_torch_fused_experts_impl` (npu_moe_init_routing_v2 + per-call transpose).
#
# NOTE: the custom `moe_grouped_matmul` (NZ-weight GMM) is intentionally not
# covered and not used by the plugin: its kernel crashes the aicore on
# ascend910b even for trivial single-expert inputs (CCU instruction address
# check error). The grouped matmuls stay on torch_npu.npu_grouped_matmul.

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
from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe import (
    _ascendc_fused_experts_impl,
    _torch_fused_experts_impl,
    fused_topk_ascend,
)

DEVICE = "npu:0"


def _ref_topk_softmax(gating_output: torch.Tensor, topk: int, renormalize: bool):
    """fp32 softmax -> topk -> optional L1 renorm (vLLM fused_topk semantics)."""
    probs = torch.softmax(gating_output.to(torch.float32), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids.to(torch.int32)


def test_moe_gating_top_k():
    """moe_gating_top_k vs fp32 torch reference (Qwen3.6 shape: 256 experts, top-8)."""
    torch.manual_seed(0)
    num_tokens, num_experts, topk = 129, 256, 8

    for in_dtype in (torch.float32, torch.bfloat16):
        logits = torch.randn(num_tokens, num_experts, dtype=in_dtype, device=DEVICE)

        for renormalize in (True, False):
            y, expert_idx, _ = torch.ops._C_ascend.moe_gating_top_k(
                logits,
                topk,
                1,  # k_group
                1,  # group_count
                0,  # group_select_mode
                int(renormalize),  # renorm
                0,  # norm_type: softmax
                False,  # out_flag
                1.0,  # routed_scaling_factor
                1e-20,  # eps
                None,  # bias_opt
            )
            ref_w, ref_ids = _ref_topk_softmax(logits, topk, renormalize)

            assert y.dtype == logits.dtype, (y.dtype, logits.dtype)
            assert expert_idx.dtype == torch.int32
            assert y.shape == (num_tokens, topk)
            # Same experts selected (order within top-k may differ for ties).
            for i in range(num_tokens):
                assert set(expert_idx[i].tolist()) == set(ref_ids[i].tolist()), (
                    f"token {i}: ids {expert_idx[i].tolist()} vs ref {ref_ids[i].tolist()}"
                )
            # Gather ref weights in the op's id order and compare values.
            ref_w_reorder = torch.gather(
                torch.softmax(logits.to(torch.float32), dim=-1), 1, expert_idx.long()
            )
            if renormalize:
                ref_w_reorder = ref_w_reorder / ref_w_reorder.sum(dim=-1, keepdim=True)
            # Kernel computes in reduced precision internally; allow bf16-level
            # tolerance (fp32 input: kernel still rounds to ~bf16 precision).
            rtol, atol = (1e-2, 1e-3) if in_dtype == torch.float32 else (2e-2, 1e-2)
            torch.testing.assert_close(
                y.to(torch.float32), ref_w_reorder, rtol=rtol, atol=atol
            )

    # fused_topk_ascend wrapper: fp32 weights out, ids dtype honored.
    hidden = torch.randn(num_tokens, 64, dtype=torch.bfloat16, device=DEVICE)
    logits = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device=DEVICE)
    w, ids, _ = fused_topk_ascend(hidden, logits, topk, True, torch.int64)
    assert w.dtype == torch.float32
    assert ids.dtype == torch.int64
    assert w.shape == (num_tokens, topk)
    torch.testing.assert_close(
        w.sum(dim=-1), torch.ones(num_tokens, device=DEVICE), rtol=1e-3, atol=1e-3
    )
    print("moe_gating_top_k test passed")


def test_fused_experts_parity():
    """_ascendc_fused_experts_impl vs legacy _torch_fused_experts_impl."""
    torch.manual_seed(1)
    num_tokens, hidden, intermediate = 33, 256, 128
    global_num_experts, topk = 16, 4
    ep_rank, num_local_experts = 1, 8  # exercise the expert_map path

    hidden_states = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=DEVICE)
    w1 = torch.randn(num_local_experts, 2 * intermediate, hidden, dtype=torch.bfloat16, device=DEVICE) * 0.02
    w2 = torch.randn(num_local_experts, hidden, intermediate, dtype=torch.bfloat16, device=DEVICE) * 0.02

    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32, device=DEVICE)
    expert_map[ep_rank * num_local_experts:(ep_rank + 1) * num_local_experts] = torch.arange(
        num_local_experts, dtype=torch.int32, device=DEVICE
    )
    topk_ids = torch.randint(0, global_num_experts, (num_tokens, topk), dtype=torch.int32, device=DEVICE)
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32, device=DEVICE)

    # Legacy path (original weight layout [E, 2I, H] / [E, H, I]).
    ref = _torch_fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids,
        global_num_experts=global_num_experts, expert_map=expert_map,
    )

    # AscendC path (pre-transposed layout [E, H, 2I] / [E, I, H]).
    w1_t = w1.transpose(1, 2).contiguous()
    w2_t = w2.transpose(1, 2).contiguous()
    out = _ascendc_fused_experts_impl(
        hidden_states, w1_t, w2_t, topk_weights, topk_ids,
        global_num_experts=global_num_experts, expert_map=expert_map,
    )

    assert out.shape == ref.shape == hidden_states.shape
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), rtol=2e-2, atol=2e-2)

    # Also without expert_map (dense path).
    topk_ids_local = torch.randint(0, num_local_experts, (num_tokens, topk), dtype=torch.int32, device=DEVICE)
    ref = _torch_fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids_local)
    out = _ascendc_fused_experts_impl(hidden_states, w1_t, w2_t, topk_weights, topk_ids_local)
    torch.testing.assert_close(out.to(torch.float32), ref.to(torch.float32), rtol=2e-2, atol=2e-2)
    print("fused_experts parity test passed")


if __name__ == "__main__":
    test_moe_gating_top_k()
    test_fused_experts_parity()
    print("All MoE custom-op tests passed")
