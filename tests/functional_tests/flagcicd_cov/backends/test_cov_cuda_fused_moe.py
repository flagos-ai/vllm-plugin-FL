# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-GPU numerical coverage for vendor/cuda/impl/fused_moe.py.

Outputs are compared against independent PyTorch references computed in the
test (sorted-token construction for moe_align_block_size, softmax top-k for
topk_softmax, an unbiased-scored grouped selection for grouped_topk, and a
per-expert dense matmul reference for the fused MoE triton kernel driven
through the public fused_experts_impl entry point).
"""

import pytest
import torch
import torch.nn.functional as F

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from vllm_fl.dispatch import get_default_manager
from vllm_fl.dispatch.policy import SelectionPolicy, policy_context

pytestmark = [pytest.mark.gpu]


def _force_vendor_cuda(op_name):
    return policy_context(
        SelectionPolicy.from_dict(per_op_order={op_name: ["vendor:cuda"]})
    )


def _reference_align(topk_ids, num_experts, block_size):
    """Independent construction of the block-aligned token layout.

    Returns (sorted_ids, expert_ids, num_post_pad) where each expert's tokens
    are padded to a multiple of block_size with numel() sentinels and the
    trailing padding rows fill the final block.
    """
    numel = topk_ids.numel()
    flat = topk_ids.reshape(-1).tolist()
    counts = [0] * num_experts
    for e in flat:
        counts[e] += 1
    padded_counts = [((c + block_size - 1) // block_size) * block_size for c in counts]
    max_len = sum(padded_counts)
    sorted_ids = torch.full((max_len,), numel, dtype=torch.int32)
    expert_ids = torch.empty((max_len // block_size,), dtype=torch.int32)
    offset = 0
    for e in range(num_experts):
        rows = [i for i, v in enumerate(flat) if v == e]
        for j, row in enumerate(rows):
            sorted_ids[offset + j] = row
        for b in range(padded_counts[e] // block_size):
            expert_ids[(offset // block_size) + b] = e
        offset += padded_counts[e]
    return sorted_ids, expert_ids, torch.tensor([max_len], dtype=torch.int32)


class TestMoeAlignBlockSizeCuda:
    """moe_align_block_size_cuda block-aligns the sorted token layout."""

    def test_selected_impl_is_cuda(self):
        """moe_align_block_size resolves to vendor.cuda under the policy."""
        with _force_vendor_cuda("moe_align_block_size"):
            assert (
                get_default_manager().get_selected_impl_id("moe_align_block_size")
                == "vendor.cuda"
            )

    @pytest.mark.parametrize(
        "tokens,topk,experts,block",
        [(4, 3, 4, 4), (8, 2, 8, 8), (16, 4, 4, 16)],
    )
    def test_layout_matches_reference(self, device, tokens, topk, experts, block):
        """The block-aligned layout matches the reference construction."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.fused_moe import (
            moe_align_block_size_cuda,
        )

        torch.manual_seed(tokens + experts)
        topk_ids = torch.randint(
            0, experts, (tokens, topk), device=device, dtype=torch.int32
        )
        sorted_ids, expert_ids, post_pad = moe_align_block_size_cuda(
            topk_ids, block, experts
        )
        ref_sorted, ref_experts, ref_count = _reference_align(topk_ids, experts, block)
        assert int(post_pad.item()) == int(ref_count.item())
        assert sorted_ids[: ref_sorted.numel()].tolist() == ref_sorted.tolist()
        # The padding region beyond num_tokens_post_pad is uninitialized
        # capacity; only the active prefix is contractual.
        active_blocks = int(ref_count.item()) // block
        assert expert_ids[:active_blocks].tolist() == ref_experts.tolist()

    def test_pad_sorted_ids_extends_capacity(self, device):
        """pad_sorted_ids=True rounds the buffer up to a block multiple."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.fused_moe import (
            moe_align_block_size_cuda,
        )

        topk_ids = torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.int32)
        block, experts = 4, 2
        plain_sorted, _, plain_post = moe_align_block_size_cuda(
            topk_ids, block, experts
        )
        padded_sorted, _, padded_post = moe_align_block_size_cuda(
            topk_ids, block, experts, pad_sorted_ids=True
        )
        assert int(padded_post.item()) == int(plain_post.item())
        assert padded_sorted.numel() >= plain_sorted.numel()
        assert padded_sorted.numel() % block == 0
        assert padded_sorted[: plain_sorted.numel()].tolist() == (
            plain_sorted.tolist()
        )

    def test_fewer_tokens_than_experts_caps_buffer(self, device):
        """numel() < num_experts caps the buffer at numel()*block_size."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.fused_moe import (
            moe_align_block_size_cuda,
        )

        topk_ids = torch.tensor([[0], [2]], device=device, dtype=torch.int32)
        block, experts = 8, 16
        sorted_ids, expert_ids, post_pad = moe_align_block_size_cuda(
            topk_ids, block, experts
        )
        # Unpadded capacity would be numel + experts*(block-1) = 2+16*7=114;
        # the small-batch cap is numel*block = 16.
        assert sorted_ids.numel() == 16
        assert expert_ids.numel() == 2
        assert int(post_pad.item()) == 16
        # Both experts' rows live in the first block; the second block is
        # padding for expert 2.
        assert expert_ids.tolist() == [0, 2]

    def test_expert_map_applied_when_not_ignoring_invalid(self, device):
        """expert_map remaps expert_ids when ignore_invalid_experts=False."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.fused_moe import (
            moe_align_block_size_cuda,
        )

        topk_ids = torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.int32)
        block, experts = 4, 2
        expert_map = torch.tensor([7, 3], device=device, dtype=torch.int32)
        sorted_ids, expert_ids, post_pad = moe_align_block_size_cuda(
            topk_ids, block, experts, expert_map=expert_map
        )
        assert int(post_pad.item()) == 8
        assert expert_ids.tolist() == [7, 3]


class TestTopkSoftmaxCuda:
    """topk_softmax_cuda returns softmax top-k weights and ids in place."""

    @pytest.mark.parametrize("renormalize", [False, True])
    def test_matches_softmax_reference(self, device, renormalize):
        """softmax top-k weights and ids match the reference."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.fused_moe import (
            topk_softmax_cuda,
        )

        torch.manual_seed(1234)
        num_tokens, num_experts, topk = 16, 64, 4
        gating = torch.randn(num_tokens, num_experts, device=device)
        weights = torch.empty(num_tokens, topk, device=device)
        ids = torch.empty(num_tokens, topk, device=device, dtype=torch.int32)
        token_expert_indices = torch.empty_like(ids)

        out_weights, out_ids = topk_softmax_cuda(
            weights, ids, token_expert_indices, gating, renormalize
        )
        # The wrapper returns the same (mutated) tensors.
        assert out_weights is weights and out_ids is ids

        probs = gating.softmax(dim=-1)
        ref_weights, ref_ids = probs.topk(topk, dim=-1)
        if renormalize:
            ref_weights = ref_weights / ref_weights.sum(-1, keepdim=True)
        assert torch.allclose(weights, ref_weights, rtol=1e-5, atol=1e-6)
        assert torch.equal(ids.long(), ref_ids)

    def test_bfloat16_gating(self, device):
        """bf16 gating matches the fp32 softmax top-k reference."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.fused_moe import (
            topk_softmax_cuda,
        )

        torch.manual_seed(99)
        gating = torch.randn(8, 32, device=device, dtype=torch.bfloat16)
        weights = torch.empty(8, 2, device=device)
        ids = torch.empty(8, 2, device=device, dtype=torch.int32)
        token_expert_indices = torch.empty_like(ids)
        topk_softmax_cuda(weights, ids, token_expert_indices, gating, False)
        probs = gating.float().softmax(dim=-1)
        ref_weights, ref_ids = probs.topk(2, dim=-1)
        assert torch.equal(ids.long(), ref_ids)
        assert torch.allclose(weights, ref_weights, rtol=1e-2, atol=1e-3)


class TestMoeSumCuda:
    """moe_sum_cuda reduces the topk dimension into out."""

    def test_matches_sum_reference(self, device):
        """fp32 moe_sum matches the topk-dim reduction."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.fused_moe import moe_sum_cuda

        torch.manual_seed(5)
        inp = torch.randn(8, 4, 128, device=device, dtype=torch.float32)
        out = torch.empty(8, 128, device=device, dtype=torch.float32)
        moe_sum_cuda(inp, out)
        assert torch.allclose(out, inp.sum(dim=1), rtol=1e-4, atol=4e-4)

    def test_bfloat16_routes_to_torch_sum(self, device):
        """topk outside the specialized kernels reduces through torch.sum."""
        """topk outside the specialized kernels reduces through torch.sum."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.fused_moe import moe_sum_cuda

        inp = torch.randn(4, 5, 256, device=device, dtype=torch.bfloat16)
        out = torch.empty(4, 256, device=device, dtype=torch.bfloat16)
        moe_sum_cuda(inp, out)
        ref = inp.float().sum(dim=1)
        assert torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-2)


class TestGroupedTopkCuda:
    """grouped_topk_cuda selects by biased scores, weights are unbiased."""

    @staticmethod
    def _reference(scores, n_group, topk_group, topk, renormalize, rsf, bias):
        """Mirror the noaux-tc kernel (scoring_func=0, i.e. no activation).

        Group scores are the sum of the two largest biased scores per group;
        selection uses the biased scores while the weights are the raw
        (unbiased) scores, renormalized against their sum plus 1e-20.
        """
        num_tokens, num_experts = scores.shape
        per_group = num_experts // n_group
        raw = scores.float()
        biased = raw + bias.unsqueeze(0)
        out_w = torch.empty(num_tokens, topk, dtype=torch.float32)
        out_i = torch.empty(num_tokens, topk, dtype=torch.int32)
        for t in range(num_tokens):
            row = biased[t]
            group_scores = (
                row.view(n_group, per_group).topk(2, dim=-1).values.sum(-1)
            )
            top_groups = torch.topk(group_scores, topk_group).indices
            mask = torch.zeros(n_group, dtype=torch.bool)
            mask[top_groups] = True
            expert_mask = (
                mask.unsqueeze(-1)
                .expand(n_group, per_group)
                .reshape(num_experts)
            )
            masked = row.masked_fill(~expert_mask, float("-inf"))
            ids = torch.topk(masked, topk).indices
            out_i[t] = ids.to(torch.int32)
            weights = raw[t][ids]
            if renormalize:
                weights = weights / (weights.sum() + 1e-20)
            out_w[t] = weights * rsf
        return out_w, out_i

    def test_matches_reference_with_bias(self, device):
        """Biased selection with unbiased weights matches the kernel mirror."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.fused_moe import (
            grouped_topk_cuda,
        )

        torch.manual_seed(7)
        num_tokens, num_experts, n_group, topk_group, topk = 8, 32, 4, 2, 4
        scores = torch.randn(num_tokens, num_experts, device=device)
        bias = (torch.randn(num_experts, device=device) * 0.1).float()
        rsf = 2.5
        weights, ids = grouped_topk_cuda(
            scores, n_group, topk_group, topk, False, rsf, bias
        )
        ref_w, ref_i = self._reference(
            scores, n_group, topk_group, topk, False, rsf, bias
        )
        assert torch.equal(ids, ref_i)
        assert torch.allclose(weights, ref_w, rtol=1e-5, atol=1e-6)

    def test_renormalize_matches_reference(self, device):
        """Renormalized grouped topk weights match the kernel mirror."""
        from vllm_fl.dispatch.backends.vendor.cuda.impl.fused_moe import (
            grouped_topk_cuda,
        )

        torch.manual_seed(11)
        num_tokens, num_experts, n_group, topk_group, topk = 6, 16, 2, 1, 2
        scores = torch.randn(num_tokens, num_experts, device=device)
        bias = torch.zeros(num_experts, device=device)
        weights, ids = grouped_topk_cuda(
            scores, n_group, topk_group, topk, True, 1.0, bias
        )
        ref_w, ref_i = self._reference(
            scores, n_group, topk_group, topk, True, 1.0, bias
        )
        assert torch.equal(ids, ref_i)
        assert torch.allclose(weights, ref_w, rtol=1e-5, atol=1e-6)


class TestInvokeFusedMoeTritonKernelCuda:
    """The fused MoE kernel path produces the per-expert reference output."""

    def test_fused_experts_under_vendor_policy_matches_reference(self, device):
        """The fused MoE path matches the per-expert dense reference."""
        from vllm_fl.ops.fused_moe.fused_moe import fused_experts_impl

        torch.manual_seed(21)
        num_tokens, hidden, inter = 16, 128, 96
        num_experts, topk = 4, 2
        dtype = torch.bfloat16

        hidden_states = torch.randn(num_tokens, hidden, device=device, dtype=dtype)
        w1 = torch.randn(
            num_experts, 2 * inter, hidden, device=device, dtype=dtype
        ) * (1.0 / hidden**0.5)
        w2 = torch.randn(
            num_experts, hidden, inter, device=device, dtype=dtype
        ) * (1.0 / inter**0.5)

        gating = torch.randn(num_tokens, num_experts, device=device)
        probs = gating.softmax(dim=-1)
        topk_weights, topk_ids = probs.topk(topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)

        with policy_context(
            SelectionPolicy.from_dict(
                per_op_order={
                    op: ["vendor:cuda"]
                    for op in (
                        "invoke_fused_moe_triton_kernel",
                        "moe_align_block_size",
                        "moe_sum",
                        "silu_and_mul",
                    )
                }
            )
        ):
            assert (
                get_default_manager().get_selected_impl_id(
                    "invoke_fused_moe_triton_kernel"
                )
                == "vendor.cuda"
            )
            out = fused_experts_impl(
                hidden_states,
                w1,
                w2,
                topk_weights.to(torch.float32),
                topk_ids.to(torch.int32),
            )

        # Independent per-expert dense reference.
        ref = torch.zeros(num_tokens, hidden, device=device, dtype=torch.float32)
        for t in range(num_tokens):
            for j in range(topk):
                e = int(topk_ids[t, j])
                gate_up = hidden_states[t].float() @ w1[e].float().T
                gate, up = gate_up[:inter], gate_up[inter:]
                inter_out = F.silu(gate) * up
                ref[t] += (
                    topk_weights[t, j].float() * (inter_out @ w2[e].float().T)
                )
        assert out.shape == (num_tokens, hidden)
        assert torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-2), (
            "fused MoE output deviates from the per-expert reference"
        )
