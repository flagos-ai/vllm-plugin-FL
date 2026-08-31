"""Ported verbatim from the T-Head vendor fork:
vllm/model_executor/layers/fused_moe/ppu_deep_gemm_utils.py.
Only change: ppu_deep_gemm import path -> vllm_fl.ops.ppu_deep_gemm.
"""
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Taken from https://github.com/ModelTC/LightLLM/blob/8ed97c74c18f11505b048b1ba00ba5c0cef8bff6/lightllm/common/fused_moe/deepep_scatter_gather.py
and updated to fit vllm needs and terminology.
"""

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm import _custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens
from vllm.triton_utils import tl, triton
from vllm_fl.ops.ppu_deep_gemm import get_mk_alignment_for_contiguous_layout
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)


def expert_num_tokens_round_up_and_sum(
    expert_num_tokens: torch.Tensor, alignment: int
) -> int:
    # Round up each element in expert_num_tokens to the nearest multiple of
    # alignment.
    ent = (expert_num_tokens.to(torch.int64) + (alignment - 1)) // alignment * alignment
    return torch.sum(ent).item()


def compute_aligned_M(
    M: int,
    num_topk: int,
    local_num_experts: int,
    alignment: int,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
):
    if (expert_tokens_meta is not None) and (
        expert_tokens_meta.expert_num_tokens_cpu is not None
    ):
        return expert_num_tokens_round_up_and_sum(
            expert_tokens_meta.expert_num_tokens_cpu, alignment=alignment
        )

    # expert_num_tokens information is not available on the cpu.
    # compute the max required size.
    M_sum = (M * num_topk) + local_num_experts * (alignment - 1)
    M_sum = round_up(M_sum, alignment)
    return M_sum


@triton.jit
def apply_expert_map(expert_id, expert_map):
    if expert_id != -1:
        expert_id = tl.load(expert_map + expert_id).to(expert_id.dtype)
    return expert_id


@triton.jit
def round_up_128(x: int) -> int:
    y = 128
    return ((x + y - 1) // y) * y


@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)

    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )

    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert

    # Extract this block's offset from the register vector (warp shuffle,
    # no global memory round-trip) then write it once to expert_start_loc.
    cur_expert_start = tl.sum(
        tl.where(offset_cumsum == cur_expert, cumsum, tl.zeros_like(cumsum))
    )
    tl.store(expert_start_loc + cur_expert, cur_expert_start)
    tl.debug_barrier()
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)

    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    # any rows in the per-expert aligned region that do not correspond to
    # real tokens are left untouched here and should remain initialized to
    # -1 so DeepGEMM can skip them
    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E):
        offs = start_m + off_expert
        mask = offs < cur_expert_token_num
        tl.store(
            m_indices_start_ptr + offs,
            cur_expert,
            mask=mask,
        )


@triton.jit
def _fwd_kernel_ep_scatter_2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
    IS_BLOCK_WISE_QUANT: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE

    offset_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = offset_in_s < SCALE_HIDDEN_SIZE

    for token_id in range(start_token_id, total_token_num, grid_num):
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        if IS_BLOCK_WISE_QUANT:
            to_copy_s = tl.load(
                recv_x_scale + token_id * recv_x_scale_stride0 + offset_in_s,
                mask=mask_s,
            )
        else:
            to_copy_s = tl.load(recv_x_scale + token_id * recv_x_scale_stride0)

        for topk_index in tl.range(0, topk_num, 1, num_stages=4):
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)

            if HAS_EXPERT_MAP:
                expert_id = apply_expert_map(expert_id, expert_map)

            if expert_id >= 0:
                dest_token_index = tl.atomic_add(expert_start_loc + expert_id, 1)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index,
                )
                output_tensor_ptr = (
                    output_tensor + dest_token_index * output_tensor_stride0
                )
                output_tensor_scale_ptr = (
                    output_tensor_scale + dest_token_index * output_tensor_scale_stride0
                )
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
                if IS_BLOCK_WISE_QUANT:
                    tl.store(
                        output_tensor_scale_ptr + offset_in_s, to_copy_s, mask=mask_s
                    )
                else:
                    tl.store(output_tensor_scale_ptr, to_copy_s)


@triton.jit
def _fwd_kernel_ep_scatter_2_bf16(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    topk_num: tl.constexpr,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE

    for token_id_int32 in range(start_token_id, total_token_num, grid_num):
        token_id = token_id_int32.to(tl.int64)
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)

        for topk_idx in tl.range(0, topk_num, 1, num_stages=4):
            topk_index = topk_idx.to(tl.int64)
            expert_id = tl.load(recv_topk + token_id * recv_topk_stride0 + topk_index)

            if HAS_EXPERT_MAP:
                expert_id = apply_expert_map(expert_id, expert_map)

            if expert_id >= 0:
                dest_token_index_int32 = tl.atomic_add(expert_start_loc + expert_id, 1)
                dest_token_index = dest_token_index_int32.to(tl.int64)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_index,
                    dest_token_index_int32,
                )
                output_tensor_ptr = (
                    output_tensor + dest_token_index * output_tensor_stride0
                )
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)


@triton.jit
def _fwd_kernel_ep_scatter_2_optimal(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    with_scale: tl.constexpr,
    IS_BLOCK_WISE_QUANT: tl.constexpr,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    COPY_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    token_id_int32 = tl.program_id(0)
    token_id = token_id_int32.to(tl.int64)

    expert_offsets = tl.arange(0, BLOCK_SIZE)
    expert_mask = expert_offsets < topk_num
    expert_loc = tl.load(
        recv_topk + token_id * recv_topk_stride0 + expert_offsets,
        mask=expert_mask,
        other=-1,
    )

    if HAS_EXPERT_MAP:
        expert_loc = tl.load(expert_map + expert_loc, mask=(expert_loc != -1)).to(
            expert_loc.dtype
        )
    tt_mask = expert_mask & (expert_loc >= 0)
    if tt_mask.sum() == 0:
        return
    dest_token_index_int32 = tl.atomic_add(
        expert_start_loc + expert_loc, 1, mask=tt_mask
    )
    tl.store(
        output_index + token_id * output_index_stride0 + expert_offsets,
        dest_token_index_int32,
        mask=tt_mask,
        eviction_policy="evict_last",
    )
    tl.debug_barrier()
    dest_token_index = tl.load(
        output_index + token_id * output_index_stride0 + expert_offsets, mask=tt_mask
    ).to(tl.int64)

    value_offsets = tl.arange(0, COPY_SIZE)
    for _ in tl.range(0, triton.cdiv(HIDDEN_SIZE, COPY_SIZE), num_stages=num_stages):
        copy_mask = value_offsets < HIDDEN_SIZE
        to_copy = tl.load(
            recv_x + token_id * recv_x_stride0 + value_offsets, mask=copy_mask
        )
        output_offsets = (
            dest_token_index[:, None] * output_tensor_stride0 + value_offsets[None, :]
        )
        to_copy = to_copy[None, :].broadcast_to(BLOCK_SIZE, COPY_SIZE)
        tl.store(
            output_tensor + output_offsets,
            to_copy,
            mask=(copy_mask[None, :] & tt_mask[:, None]),
        )
        value_offsets += COPY_SIZE

    if with_scale:
        if IS_BLOCK_WISE_QUANT:
            scale_offsets = tl.arange(0, COPY_SIZE)
            for _ in tl.range(
                0,
                triton.cdiv(SCALE_HIDDEN_SIZE, COPY_SIZE),
            ):
                copy_mask = scale_offsets < SCALE_HIDDEN_SIZE
                # Use stride-based access for both input and output scale.
                # For row-major uint8 scales: stride1 == 1, equivalent to
                # contiguous access.  For col-major uint16 scales: stride1
                # is the outer dimension (M), so scale_offsets must be
                # multiplied by stride1.
                to_copy_scale = tl.load(
                    recv_x_scale + token_id * recv_x_scale_stride0 + scale_offsets * recv_x_scale_stride1,
                    mask=copy_mask,
                )
                output_scale_offsets = (
                    dest_token_index[:, None] * output_tensor_scale_stride0
                    + scale_offsets[None, :] * output_tensor_scale_stride1
                )
                to_copy_scale = to_copy_scale[None, :].broadcast_to(
                    BLOCK_SIZE, COPY_SIZE
                )
                tl.store(
                    output_tensor_scale + output_scale_offsets,
                    to_copy_scale,
                    mask=(copy_mask[None, :] & tt_mask[:, None]),
                )
                scale_offsets += COPY_SIZE
        else:
            to_copy_scale = tl.load(recv_x_scale + token_id * recv_x_scale_stride0)
            # to_copy_scale = to_copy_scale[None, :].broadcast_to(BLOCK_SIZE, 1)
            output_scale_offsets = dest_token_index * output_tensor_scale_stride0
            tl.store(
                output_tensor_scale + output_scale_offsets, to_copy_scale, mask=tt_mask
            )


@torch.no_grad()
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_map: torch.Tensor | None,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
    is_block_wise_quant: bool,
    is_channel_wise_quant: bool,
):
    # PPU NOTE: Using PPU DG nopad
    # token num of per expert is not need to be aligned to 128.
    BLOCK_E = 1
    if recv_x.dtype == torch.uint8:
        # for mxfp4 quantization, block size is 32 and pack 2 e2m1 qweight into 1 uint8
        # need to use 32 // 2
        BLOCK_D = 16
    else:
        BLOCK_D = 128  # block size of quantization

    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
    grid = num_experts

    assert m_indices.shape[0] % BLOCK_E == 0
    assert expert_start_loc.shape[0] == num_experts

    _fwd_kernel_ep_scatter_1[(grid,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        num_warps=num_warps,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
    )

    # FIXME(kai): cause crash for dsv4 flash base
    if 0 and expert_map is None and recv_x.dtype != torch.uint8:
        # cuda kernel do not support uint8 for mxfp4
        ops.ep_scatter_2_cuda(
            recv_x,
            recv_x_scale,
            recv_topk,
            expert_start_loc,
            output_tensor,
            output_index,
            output_tensor_scale,
            (recv_x_scale is not None),
        )


    grid = lambda meta: (recv_x.shape[0],)

    # For uint16 col-major scale, use aq_scale.shape[1] (S_groups // 2) as
    # SCALE_HIDDEN_SIZE and pass the actual strides so the kernel handles
    # the non-unit stride access correctly.
    if recv_x_scale is not None and recv_x_scale.dtype == torch.uint16:
        scale_hidden_size = recv_x_scale.shape[1]
    else:
        scale_hidden_size = hidden_size // BLOCK_D

    _fwd_kernel_ep_scatter_2_optimal[grid](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        0 if recv_x_scale is None else recv_x_scale.stride(0),
        0 if recv_x_scale is None else recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        0 if output_tensor_scale is None else output_tensor_scale.stride(0),
        0 if output_tensor_scale is None else output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        expert_map=expert_map,
        with_scale=(recv_x_scale is not None),
        topk_num=recv_topk.shape[1],
        HAS_EXPERT_MAP=expert_map is not None,
        HIDDEN_SIZE=hidden_size,
        SCALE_HIDDEN_SIZE=scale_hidden_size,
        IS_BLOCK_WISE_QUANT=is_block_wise_quant,
        BLOCK_SIZE=triton.next_power_of_2(recv_topk.shape[1]),
        COPY_SIZE=512,
        num_stages=3,
        num_warps=8,
    )
    return

    # keep codes only for debug
    grid = min(recv_topk.shape[0], 1024 * 8)

    if is_block_wise_quant or is_channel_wise_quant:
        _fwd_kernel_ep_scatter_2[(grid,)](
            recv_topk.shape[0],
            expert_start_loc,
            recv_x,
            recv_x.stride(0),
            recv_x.stride(1),
            recv_x_scale,
            recv_x_scale.stride(0),
            recv_x_scale.stride(1),
            recv_topk,
            recv_topk.stride(0),
            recv_topk.stride(1),
            output_tensor,
            output_tensor.stride(0),
            output_tensor.stride(1),
            output_tensor_scale,
            output_tensor_scale.stride(0),
            output_tensor_scale.stride(1),
            output_index,
            output_index.stride(0),
            output_index.stride(1),
            topk_num=recv_topk.shape[1],
            expert_map=expert_map,
            HAS_EXPERT_MAP=expert_map is not None,
            num_warps=num_warps,
            HIDDEN_SIZE=hidden_size,
            HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
            SCALE_HIDDEN_SIZE=hidden_size // BLOCK_D,
            SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size // BLOCK_D),
            IS_BLOCK_WISE_QUANT=is_block_wise_quant,
        )
        return

    _fwd_kernel_ep_scatter_2_bf16[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=recv_topk.shape[1],
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
    )
    return


@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    expert_map,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block = tl.program_id(0)
    start_cur_token = tl.program_id(1)
    grid_num = tl.num_programs(1)

    for cur_token in range(start_cur_token, total_token_num, grid_num):
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index in range(0, topk_num):
            expert_id = tl.load(
                recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index
            )

            if HAS_EXPERT_MAP:
                expert_id = apply_expert_map(expert_id, expert_map)

            if expert_id >= 0:
                source_token_index = tl.load(
                    input_index + cur_token * input_index_stride0 + topk_index
                )
                acc_weight = tl.load(
                    recv_topk_weight + cur_token * recv_topk_weight_stride0 + topk_index
                )
                tmp = tl.load(
                    input_tensor
                    + source_token_index.to(tl.int64) * input_tensor_stride0
                    + cur_block * BLOCK_D
                    + off_d
                )
                accumulator += tmp.to(tl.float32) * acc_weight

        tl.store(
            output_tensor
            + cur_token * output_tensor_stride0
            + cur_block * BLOCK_D
            + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    expert_map: torch.Tensor | None,
    output_tensor: torch.Tensor,
):
    num_warps = 2
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    BLOCK_D = min(hidden_size, 1024)

    # PPU NOTE: resize BLOCK_D to enable deep gemm backend for hidden_size
    # which is can't be divisible by 1024.
    while hidden_size % BLOCK_D:
        logger.warning_once(
            f"hidden_size ({hidden_size}) is not divisible by BLOCK_D ({BLOCK_D}), "
            f"falling back to BLOCK_D={BLOCK_D // 2} which may result in degraded "
            f"MoE kernel accuracy or performance. Consider setting VLLM_PPU_MOE_BACKEND "
            f"to use the Triton / Acext MoE backend for this model instead.",
        )
        BLOCK_D = BLOCK_D // 2
    assert hidden_size % BLOCK_D == 0
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))

    _fwd_kernel_ep_gather[grid](
        num_tokens,
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        expert_map=expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )
    return


def deepgemm_moe_permute(
    aq: torch.Tensor,
    aq_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    local_num_experts: int,
    expert_map: torch.Tensor | None,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
    aq_out: torch.Tensor | None = None,
    is_block_wise_quant: bool = True,
    is_channel_wise_quant: bool = False,
):
    assert aq.ndim == 2
    assert topk_ids.dtype.is_signed, "The kernel uses -1 to represent invalid topk_ids"
    assert not (is_block_wise_quant and is_channel_wise_quant), (
        "is_block_wise_quant and is_channel_wise_quant "
        "should not be true at the same time!"
    )

    H = aq.size(1)
    device = aq.device

    is_mxfp4 = (aq.dtype == torch.uint8)
    block_m, block_k = get_mk_alignment_for_contiguous_layout(
        is_blockwise=is_block_wise_quant, is_mxfp4=is_mxfp4
    )

    M_sum = compute_aligned_M(
        M=topk_ids.size(0),
        num_topk=topk_ids.size(1),
        local_num_experts=local_num_experts,
        alignment=block_m,
        expert_tokens_meta=expert_tokens_meta,
    )

    expert_start_loc = torch.empty(
        (local_num_experts), device=device, dtype=torch.int32
    )

    assert aq_out is None or aq_out.shape == (M_sum, H)
    if aq_out is None:
        aq_out = torch.empty((M_sum, H), device=device, dtype=aq.dtype)

    if is_block_wise_quant:
        if aq_scale is not None and aq_scale.dtype == torch.uint16:
            # uint16 col-major scale: physical layout [S//2, M_sum] contiguous,
            # logical layout [M_sum, S//2] via .t() (stride(0)=1, stride(1)=M_sum).
            scale_hidden = aq_scale.shape[1]  # S_groups // 2
            aq_scale_out = torch.empty(
                (scale_hidden, M_sum), device=device, dtype=torch.uint16
            ).t()
        elif aq_scale is not None and aq_scale.dtype == torch.uint8:
            # Legacy mxfp4 uint8 row-major scale
            assert block_k == 16
            aq_scale_out = torch.empty(
                (M_sum, H // block_k), device=device, dtype=torch.uint8
            )
        else:
            assert block_k == 128
            aq_scale_out = torch.empty(
                (M_sum, H // block_k), device=device, dtype=torch.float32
            )
    elif is_channel_wise_quant:
        aq_scale_out = torch.empty((M_sum, 1), device=device, dtype=torch.float32)
    else:
        aq_scale_out = None

    # DeepGEMM uses negative values in m_indices (here expert_ids) to mark
    # completely invalid / padded blocks that should be skipped. We always
    # initialize expert_ids to -1 so any row that is not explicitly written
    # by the scatter kernel will be treated as invalid and skipped by
    # DeepGEMM's scheduler.
    expert_ids = torch.full(
        (M_sum,),
        fill_value=-1,
        device=device,
        dtype=torch.int32,
    )
    inv_perm = torch.empty(topk_ids.shape, device=device, dtype=torch.int32)

    expert_num_tokens = None
    if expert_tokens_meta is not None:
        expert_num_tokens = expert_tokens_meta.expert_num_tokens
    else:
        expert_num_tokens = count_expert_num_tokens(
            topk_ids, local_num_experts, expert_map
        )

    ep_scatter(
        recv_x=aq,
        recv_x_scale=aq_scale,
        recv_topk=topk_ids.to(torch.int32),
        num_recv_tokens_per_expert=expert_num_tokens,
        expert_start_loc=expert_start_loc,
        expert_map=expert_map,
        output_tensor=aq_out,
        output_tensor_scale=aq_scale_out,
        m_indices=expert_ids,
        output_index=inv_perm,
        is_block_wise_quant=is_block_wise_quant,
        is_channel_wise_quant=is_channel_wise_quant,
    )

    return aq_out, aq_scale_out, expert_ids, inv_perm, expert_num_tokens


def deepgemm_unpermute_and_reduce(
    a: torch.Tensor,  # Grouped gemm output
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_perm: torch.Tensor,
    expert_map: torch.Tensor | None,
    output: torch.Tensor,
):
    return ep_gather(
        input_tensor=a,
        recv_topk_ids=topk_ids,
        recv_topk_weight=topk_weights,
        input_index=inv_perm,
        expert_map=expert_map,
        output_tensor=output,
    )
