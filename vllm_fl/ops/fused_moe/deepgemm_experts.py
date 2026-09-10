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

"""PPU-native DeepGEMM BF16 unquantized MoE experts (env-gated).

Opt-in via ``VLLM_FL_MOE=deepgemm`` (see ``vllm_fl.utils.use_deepgemm_moe``).
Replaces the default FlagGems Triton ``fused_moe`` expert compute with
``deep_gemm``'s grouped BF16 GEMM — the same kernels the vendor's native vLLM
0.19 build used (``m_grouped_gemm_bf16_bf16_bf16_nt`` / ``..._gemv``).

Uses the **nopad** grouped GEMM with a compact (block_align=1) permute: each
expert's rows are packed with NO 128-row padding, and small-M decode auto-
dispatches to the GEMV kernel. (The contiguous/128-aligned layout wastes ~128x
compute per active expert on sparse decode — do NOT use it here.)

Pipeline (BF16, no FP8 scales):
    deepgemm_moe_permute(block_align=1) → nopad GEMM1 → silu_and_mul
    → nopad GEMM2 → weighted unpermute+reduce (ep_gather)

Permute/gather are vendor/vLLM Triton kernels (CUDA-graph safe); ``m_rows``
(exact per-expert token counts) is fed to the nopad kernel so no host sync /
internal bincount is needed.
"""

import deep_gemm
import torch
from deep_gemm.deep_gemm_tuner.deepgemm_tools import deepgemm_moe_permute

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M,
    ep_gather,
)
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.utils import _resize_cache

from vllm_fl.ops.fused_moe.activation import apply_moe_activation


class DeepGemmExpertsFL(TritonExperts):
    """OOT unquantized BF16 MoE experts backed by deep_gemm nopad grouped GEMM.

    Subclasses ``TritonExperts`` to inherit ``moe_problem_size``,
    ``adjust_N_for_activation`` and the ``TopKWeightAndReduceNoOP`` finalize
    contract; overrides ``workspace_shapes`` (compact M_sum = M*topk) and
    ``apply``.
    """

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: "mk.ExpertTokensMetadata | None",
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Compact layout (block_align=1): no per-expert 128-row padding.
        M_sum = compute_aligned_M(M, topk, local_num_experts, 1, expert_tokens_meta)
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M_sum, max(activation_out_dim, K))
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: "mk.ExpertTokensMetadata | None",
        apply_router_weight_on_input: bool,
    ):
        assert hidden_states.dtype == torch.bfloat16, (
            "DeepGemmExpertsFL only supports bf16 unquantized MoE"
        )
        assert hidden_states.is_contiguous()
        assert expert_map is None, (
            "DeepGemmExpertsFL does not support expert parallelism (expert_map)"
        )

        a1 = hidden_states  # [M, K]
        M, K = a1.shape
        local_num_experts, N, K_w = w1.shape  # w1: [E, 2I, K]
        assert K_w == K

        # Kernels use -1 for invalid ids -> topk_ids must be signed (router: int32).
        if not topk_ids.dtype.is_signed:
            topk_ids = topk_ids.to(torch.int32)

        # ---- compact permute: pack tokens per-expert (no 128 padding) ----
        # returns: a1_perm [M_sum, K], m_indices [M_sum], inv_perm [M, topk],
        #          m_rows (expert_num_tokens) [E]. M_sum == M * topk.
        a1_perm, _scale_out, m_indices, inv_perm, m_rows = deepgemm_moe_permute(
            a1, None, topk_ids, local_num_experts, block_align=1, block_k=K
        )
        M_sum = a1_perm.size(0)

        # ---- grouped GEMM 1 (nopad): [M_sum, K] x [E, 2I, K]^T -> [M_sum, 2I] ----
        mm1_out = _resize_cache(workspace2, (M_sum, N))
        deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
            a1_perm, w1, mm1_out, m_indices, m_rows
        )

        # ---- activation: silu_and_mul -> [M_sum, I] ----
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        act_out = _resize_cache(workspace13, (M_sum, activation_out_dim))
        apply_moe_activation(activation, act_out, mm1_out.view(-1, N))

        # ---- grouped GEMM 2 (nopad): [M_sum, I] x [E, K, I]^T -> [M_sum, K] ----
        mm2_out = _resize_cache(workspace2, (M_sum, K))
        deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
            act_out, w2, mm2_out, m_indices, m_rows
        )

        # ---- weighted unpermute + reduce over topk -> output [M, K] ----
        if apply_router_weight_on_input:
            topk_weights = torch.ones_like(topk_weights)
        ep_gather(
            input_tensor=mm2_out,
            recv_topk_ids=topk_ids,
            recv_topk_weight=topk_weights,
            input_index=inv_perm,
            expert_map=None,
            output_tensor=output,
        )
