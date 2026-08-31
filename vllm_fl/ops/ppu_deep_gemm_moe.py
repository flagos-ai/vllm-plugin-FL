"""Ported verbatim from the T-Head vendor fork:
vllm/model_executor/layers/fused_moe/experts/ppu_deep_gemm_moe.py.
Only change: vendor-internal import paths -> vllm_fl.ops.*.
"""
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm_fl.ops.ppu_deep_gemm_utils import (
    compute_aligned_M,
    deepgemm_moe_permute,
    deepgemm_unpermute_and_reduce,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    swiglu_limit_func,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_packed_for_deepgemm,
    silu_mul_per_token_group_quant_fp8_colmajor,
)
try:
    from vllm.model_executor.layers.quantization.utils.ppu_mxfp4_utils import (
        downcast_to_mxfp4,
    )
except ImportError:  # vendor-only module; only the MXFP4 experts need it
    downcast_to_mxfp4 = None
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kMxfp4Dynamic,
    kMxfp4Static,
)
from vllm_fl.ops.ppu_deep_gemm import (
    DeepGemmQuantScaleFMT,
    get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_supported,
    get_deep_gemm_config,
    m_grouped_fp8_gemm_nt_nopad,
    m_grouped_int8_gemm_nt_nopad,
    m_grouped_bf16_gemm_nt_nopad,
    m_grouped_fp4_gemm_nt_nopad,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.platforms import current_platform
import vllm.envs as envs

logger = init_logger(__name__)


# Add for nvtx profiling
NVTX_PROFILE = getattr(envs, 'VLLM_PPU_NVTX_PROFILE', False)
if NVTX_PROFILE:
    try:
        from torch.cuda.nvtx import range_pop as th_nvtx_range_pop
        from torch.cuda.nvtx import range_push as th_nvtx_range_push
    except ImportError:
        NVTX_PROFILE = False

if not NVTX_PROFILE:

    def th_nvtx_range_push(label):
        pass

    def th_nvtx_range_pop():
        pass


def _valid_deep_gemm_shape(M: int, N: int, K: int) -> bool:
    align, align_k = get_mk_alignment_for_contiguous_layout()
    return align <= M and N % align == 0 and K % align_k == 0


def _valid_deep_gemm(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> bool:
    """
    Check if the given problem size is supported by the DeepGemm grouped
    gemm kernel.  All of M, N, K and the quantization block_shape must be
    aligned by `dg.get_m_alignment_for_contiguous_layout()`.
    """
    if not has_deep_gemm():
        logger.debug_once("DeepGemm disabled: deep_gemm not available.")
        return False

    M = hidden_states.size(0)
    _, K, N = w2.size()

    if (current_platform.is_device_capability((8,0))) and (
        w1.dtype in [torch.float8_e4m3fn]
        or w2.dtype in [torch.float8_e4m3fn]
    ):
        logger.debug_once(
            "DeepGemm disabled: invalid weight dtype(s). w1.dtype: %s, w2.dtype: %s on sm80",
            w1.dtype,
            w2.dtype,
        )
        return False


    if (w1.dtype
        not in [torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.uint8, torch.float8_e4m3fn]
        or w2.dtype
        not in [torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.uint8, torch.float8_e4m3fn]
    ):
        logger.debug_once(
            "DeepGemm disabled: invalid weight dtype(s). w1.dtype: %s, w2.dtype: %s",
            w1.dtype,
            w2.dtype,
        )
        return False

    if (
        not hidden_states.is_contiguous()
        or not w1.is_contiguous()
        or not w2.is_contiguous()
    ):
        logger.debug_once(
            "DeepGemm disabled: weights or activations not contiguous. "
            "hidden_states.is_contiguous(): %s, w1.is_contiguous(): %s, "
            "w2.is_contiguous(): %s",
            hidden_states.is_contiguous(),
            w1.is_contiguous(),
            w2.is_contiguous(),
        )
        return False

    return True


class PPUDeepGemmExperts(mk.FusedMoEExpertsModular):
    """DeepGemm-based fused MoE expert implementation."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.block_wise = quant_config.block_shape is not None

        if self.block_wise:
            assert (
                quant_config.block_shape[1]
                == get_mk_alignment_for_contiguous_layout(is_blockwise=self.block_wise)[1]
            )

        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return is_deep_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (None, None),
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
            (kFp8StaticChannelSym, kFp8DynamicTokenSym),
            (kInt8StaticChannelSym, kInt8DynamicTokenSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.SWIGLUSTEP]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # NOTE(rob): discovered an IMA with this combination. Needs investigation.
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        block_m = get_mk_alignment_for_contiguous_layout(is_blockwise=self.block_wise)[0]
        M_sum = compute_aligned_M(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % block_m == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M_sum, max(activation_out_dim, K))
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _act_mul_quant(
        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation
    ) -> tuple[torch.Tensor, torch.Tensor]:

        scale_fmt = DeepGemmQuantScaleFMT.from_oracle()

        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        block_k = self.block_shape[1] if self.block_shape else activation_out_dim

        # 1. DeepGemm UE8M0: use packed per-token-group quant
        if scale_fmt == DeepGemmQuantScaleFMT.UE8M0:
            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
            )
            self.activation(activation, act_out, input)
            a2q, a2q_scale = per_token_group_quant_fp8_packed_for_deepgemm(
                act_out,
                block_k,
                out_q=output,
            )
            return a2q, a2q_scale

        # 2. Hopper / non‑E8M0: prefer the fused SiLU+mul+quant kernel
        # if activation == MoEActivation.SILU:
        #     use_ue8m0 = scale_fmt == DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
        #     return silu_mul_per_token_group_quant_fp8_colmajor(
        #         input=input,
        #         output=output,
        #         use_ue8m0=use_ue8m0,
        #     )

        # 3. fallback path for non-SiLU activations in non‑UE8M0 cases.
        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )

        # deepseek v4 clamp path
        if self.gemm1_clamp_limit is not None and activation == MoEActivation.SILU:
            swiglu_limit_func(
                act_out,
                input,
                self.gemm1_clamp_limit,
            )
        else:
            # Assign act path
            self.activation(activation, act_out, input)
        if output.dtype == torch.float8_e4m3fn:
            block_k = (
                self.block_shape[1] if self.block_shape else activation_out_dim
            )
            return per_token_group_quant_fp8(
                act_out, block_k, column_major_scales=True, out_q=output
            )
        elif output.dtype == torch.int8:
            return per_token_quant_int8(act_out)
        else:
            return act_out, None

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
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if self.quant_config.quant_dtype:
            assert a1q_scale is not None
            assert a2_scale is None
            assert self.w1_scale is not None
            assert self.w2_scale is not None
            quant_dtype = self.quant_config.quant_dtype
        else:
            quant_dtype = torch.bfloat16

        a1q = hidden_states
        E, N, K = w1.size()

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        assert w2.size(1) == K

        M_sum = compute_aligned_M(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout(is_blockwise=self.block_wise)[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(
            workspace13.view(dtype=quant_dtype), (M_sum, K)
        )
        is_block_wise_quant = is_channel_wise_quant = False
        if self.quant_config.use_fp8_w8a8 or self.quant_config.use_int8_w8a8:
            is_block_wise_quant = self.block_shape is not None
            is_channel_wise_quant = not is_block_wise_quant

        a1q, a1q_scale, expert_ids, inv_perm, expert_num_tokens = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
            is_block_wise_quant=is_block_wise_quant,
            is_channel_wise_quant=is_channel_wise_quant,
        )
        assert a1q.size(0) == M_sum

        mm1_out = _resize_cache(workspace2, (M_sum, N))

        nvtx_pushed = False
        if NVTX_PROFILE:
            if (
                getattr(envs, 'VLLM_PPU_NVTX_DUMP_TOPK', False)
                and not torch.cuda.is_current_stream_capturing()
            ):
                num_activated_experts = (expert_num_tokens > 0).sum().item()
                th_nvtx_range_push(
                    f"MoE,M_{hidden_states.shape[0]}_E_{w1.shape[0]}_H_{w1.shape[2]}_In_{w1.shape[1]}_topk_{topk_ids.shape[1]}_topkids{expert_num_tokens.flatten().cpu().tolist()}_unique_{num_activated_experts}"
                )
                nvtx_pushed = True

        # calculate expert_first_token_offset for deepgemm
        experts_for_rows = torch.zeros(
            local_num_experts, dtype=torch.int32, device="cuda"
        )
        counts = expert_num_tokens
        min_n = min(counts.size(0), local_num_experts)
        if min_n > 0:
            experts_for_rows[:min_n] = counts[:min_n]

        if quant_dtype == torch.float8_e4m3fn:
            m_grouped_fp8_gemm_nt_nopad(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                mm1_out,
                expert_ids,
                experts_for_rows,
            )
            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.float8_e4m3fn),
                (M_sum, activation_out_dim),
            )
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N), output=quant_out, activation=activation
            )

            mm2_out = _resize_cache(workspace2, (M_sum, K))
            m_grouped_fp8_gemm_nt_nopad(
                (a2q, a2q_scale),
                (w2, self.w2_scale),
                mm2_out,
                expert_ids,
                experts_for_rows,
            )
        elif quant_dtype == torch.int8:
            best_config = get_deep_gemm_config(M_sum, N, K, num_groups=E)
            m_grouped_int8_gemm_nt_nopad(
                (a1q, a1q_scale),
                (w1, self.w1_scale),
                mm1_out,
                expert_ids,
                experts_for_rows,
                best_config,
            )
            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.int8), (M_sum, activation_out_dim)
            )

            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N), output=quant_out, activation=activation
            )

            mm2_out = _resize_cache(workspace2, (M_sum, K))
            best_config = get_deep_gemm_config(
                a2q.shape[0],
                w2.shape[1],
                a2q.shape[-1],
                num_groups=E,
            )
            m_grouped_int8_gemm_nt_nopad(
                (a2q, a2q_scale),
                (w2, self.w2_scale),
                mm2_out,
                expert_ids,
                experts_for_rows,
                best_config,
            )
        else:
            m_grouped_bf16_gemm_nt_nopad(
                a1q, w1, mm1_out, expert_ids, experts_for_rows
            )
            activation_out_dim = self.adjust_N_for_activation(N, activation)
            quant_out = _resize_cache(workspace13, (M_sum, activation_out_dim))
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N), output=quant_out, activation=activation
            )

            mm2_out = _resize_cache(workspace2, (M_sum, K))
            m_grouped_bf16_gemm_nt_nopad(
                a2q, w2, mm2_out, expert_ids, experts_for_rows
            )

        if apply_router_weight_on_input:
            assert topk_weights is not None
            # topk_weights = torch.ones_like(topk_weights)

        if nvtx_pushed:
            th_nvtx_range_pop()

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )


# PPU MXFP4 DeepGEMM Experts
class PPUDeepGemmExpertsMXFP4(mk.FusedMoEExpertsModular):
    """
    PPU DeepGEMM-based fused MoE expert implementation for MXFP4.

    This class implements MXFP4 x MXFP4 (both activations and weights are mxfp4)
    on PPU platform using DeepGEMM.
    """

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        # MXFP4 (w4a4) requires sm90+ on PPU; disable on sm80
        if current_platform.is_device_capability((8, 0)):
            return False
        return is_deep_gemm_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, kMxfp4Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.SWIGLUSTEP, MoEActivation.SWIGLUOAI]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        block_m = get_mk_alignment_for_contiguous_layout(is_mxfp4=True)[0]
        M_sum = compute_aligned_M(
            M, topk, local_num_experts, block_m, expert_tokens_meta
        )
        assert M_sum % block_m == 0

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        workspace1 = (M_sum, max(activation_out_dim, K))
        workspace2 = (M_sum, max(N, K))
        output = (M, K)
        return (workspace1, workspace2, output)

    def _act_mul_quant(
        self, input: torch.Tensor, output: torch.Tensor, activation: MoEActivation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale_fmt = DeepGemmQuantScaleFMT.from_oracle()

        M_sum, N = input.size()
        activation_out_dim = self.adjust_N_for_activation(N, activation)

        block_k = self.block_shape[1] if self.block_shape else activation_out_dim

        act_out = torch.empty(
            (M_sum, activation_out_dim), dtype=input.dtype, device=input.device
        )
        self.activation(activation, act_out, input)
        a_q, a_scale = downcast_to_mxfp4(act_out, axis=1)
        return a_q, a_scale

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
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert self.quant_config.quant_dtype == "mxfp4"
        quant_dtype = torch.uint8

        a1q = hidden_states
        E, N, K = w1.size()

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        # for mxfp4, pack 2 e2m1 qweight into 1 uint8
        # w1.size(2) is hidden_size // 2 and w2.size(1) is hidden_size
        assert w2.size(1) == K * 2

        M_sum = compute_aligned_M(
            M=topk_ids.size(0),
            num_topk=topk_ids.size(1),
            local_num_experts=local_num_experts,
            alignment=get_mk_alignment_for_contiguous_layout(is_mxfp4=True)[0],
            expert_tokens_meta=expert_tokens_meta,
        )

        a1q_perm = _resize_cache(
            workspace13.view(dtype=quant_dtype), (M_sum, K)
        )
        is_block_wise_quant = is_channel_wise_quant = False
        is_block_wise_quant = self.block_shape is not None
        is_channel_wise_quant = not is_block_wise_quant

        a1q, a1q_scale, expert_ids, inv_perm, expert_num_tokens = deepgemm_moe_permute(
            aq=a1q,
            aq_scale=a1q_scale,
            topk_ids=topk_ids,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta,
            aq_out=a1q_perm,
            is_block_wise_quant=is_block_wise_quant,
            is_channel_wise_quant=is_channel_wise_quant,
        )
        assert a1q.size(0) == M_sum

        mm1_out = _resize_cache(workspace2, (M_sum, N))

        nvtx_pushed = False
        if NVTX_PROFILE:
            if (
                getattr(envs, 'VLLM_PPU_NVTX_DUMP_TOPK', False)
                and not torch.cuda.is_current_stream_capturing()
            ):
                num_activated_experts = (expert_num_tokens > 0).sum().item()
                th_nvtx_range_push(
                    f"MoE,M_{hidden_states.shape[0]}_E_{w1.shape[0]}_H_{w1.shape[2]}_In_{w1.shape[1]}_topk_{topk_ids.shape[1]}_topkids{expert_num_tokens.flatten().cpu().tolist()}_unique_{num_activated_experts}"
                )
                nvtx_pushed = True

        # calculate expert_first_token_offset for deepgemm
        experts_for_rows = torch.zeros(
            local_num_experts, dtype=torch.int32, device="cuda"
        )
        counts = expert_num_tokens
        min_n = min(counts.size(0), local_num_experts)
        if min_n > 0:
            experts_for_rows[:min_n] = counts[:min_n]

        m_grouped_fp4_gemm_nt_nopad(
            (a1q, a1q_scale),
            (w1, self.w1_scale),
            self.quant_config._w1.bias,
            mm1_out,
            expert_ids,
            experts_for_rows,
        )

        activation_out_dim = self.adjust_N_for_activation(N, activation)
        if self.gemm1_clamp_limit is not None and activation == MoEActivation.SILU:
            # deepseek_v4
            act_out = torch.empty(
                (M_sum, activation_out_dim), dtype=mm1_out.dtype, device=mm1_out.device
            )
            swiglu_limit_func(
                output=act_out,
                input=mm1_out.view(-1, N),
                swiglu_limit=self.gemm1_clamp_limit,
            )
            a2q, a2q_scale = downcast_to_mxfp4(act_out, axis=1)
        else:
            quant_out = _resize_cache(
                workspace13.view(dtype=torch.uint8),
                (M_sum, activation_out_dim),
            )
            a2q, a2q_scale = self._act_mul_quant(
                input=mm1_out.view(-1, N), output=quant_out, activation=activation
            )

        # for mxfp4, K is hidden_size // 2, need to set `K * 2` here
        mm2_out = _resize_cache(workspace2, (M_sum, K * 2))
        m_grouped_fp4_gemm_nt_nopad(
            (a2q, a2q_scale),
            (w2, self.w2_scale),
            self.quant_config._w2.bias,
            mm2_out,
            expert_ids,
            experts_for_rows,
        )

        if apply_router_weight_on_input:
            assert topk_weights is not None
            # topk_weights = torch.ones_like(topk_weights)

        if nvtx_pushed:
            th_nvtx_range_pop()

        deepgemm_unpermute_and_reduce(
            a=mm2_out,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            inv_perm=inv_perm,
            expert_map=expert_map,
            output=output,
        )
