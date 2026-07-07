# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend NPU pure-torch MoE kernels.

Replaces FlagGems Triton kernels that crash on Ascend NPU.
Uses a CPU side-channel to pass moe_align data to the GEMM kernel,
avoiding any NPU→CPU transfers during the hot path.
"""

import torch
import numpy as np
from vllm.utils.math_utils import round_up


def moe_align_block_size_torch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch moe_align_block_size for Ascend NPU (CPU-based)."""
    device = topk_ids.device
    num_tokens = topk_ids.numel()

    max_num_tokens_padded = num_tokens + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if num_tokens < num_experts:
        max_num_tokens_padded = min(num_tokens * block_size, max_num_tokens_padded)

    topk_ids_flat = topk_ids.view(-1).cpu()
    padding_value = num_tokens

    expert_counts = torch.bincount(topk_ids_flat.long(), minlength=num_experts)[:num_experts]

    sorted_ids_list = []
    expert_ids_list = []

    for e in range(num_experts):
        count = expert_counts[e].item()
        if count == 0 and ignore_invalid_experts:
            continue
        expert_tokens = (topk_ids_flat == e).nonzero(as_tuple=True)[0].to(torch.int32)
        padded_count = ((count + block_size - 1) // block_size) * block_size
        num_blocks = padded_count // block_size
        padded_tokens = torch.full((padded_count,), padding_value, dtype=torch.int32)
        padded_tokens[:count] = expert_tokens
        sorted_ids_list.append(padded_tokens)
        expert_ids_list.extend([e] * num_blocks)

    if not sorted_ids_list:
        sorted_ids = torch.full((max_num_tokens_padded,), padding_value, dtype=torch.int32)
        max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
        expert_ids_out = torch.full((max_num_m_blocks,), -1, dtype=torch.int32)
        num_tokens_post_pad = torch.zeros(1, dtype=torch.int32)
        return sorted_ids.to(device), expert_ids_out.to(device), num_tokens_post_pad.to(device)

    sorted_ids = torch.cat(sorted_ids_list)
    actual_len = sorted_ids.shape[0]

    if actual_len < max_num_tokens_padded:
        pad = torch.full((max_num_tokens_padded - actual_len,), padding_value, dtype=torch.int32)
        sorted_ids = torch.cat([sorted_ids, pad])
    else:
        sorted_ids = sorted_ids[:max_num_tokens_padded]

    expert_ids_tensor = torch.tensor(expert_ids_list, dtype=torch.int32)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    if expert_ids_tensor.shape[0] < max_num_m_blocks:
        pad = torch.full((max_num_m_blocks - expert_ids_tensor.shape[0],), -1, dtype=torch.int32)
        expert_ids_tensor = torch.cat([expert_ids_tensor, pad])

    num_tokens_post_pad = torch.tensor([actual_len], dtype=torch.int32)

    if expert_map is not None and not ignore_invalid_experts:
        expert_map_cpu = expert_map.cpu()
        valid = expert_ids_tensor >= 0
        expert_ids_tensor[valid] = expert_map_cpu[expert_ids_tensor[valid].long()]

    return sorted_ids.to(device), expert_ids_tensor.to(device), num_tokens_post_pad.to(device)


def invoke_fused_moe_torch(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    B_bias: torch.Tensor | None = None,
):
    """Ascend NPU fused MoE GEMM using per-expert torch.mm loop.

    npu_grouped_matmul is disabled — it crashes aicore on MoE models like
    Qwen3.6-35B-A3B (error 507015: aicore execution abnormal). The torch.mm
    loop is slightly slower but reliable.
    """
    # Use per-expert torch.mm loop — more reliable on Ascend NPU than
    # npu_grouped_matmul which can crash aicore on certain model shapes.
    _invoke_fused_moe_loop(
        A, B, C, A_scale, B_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight, top_k, config,
        use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a8=use_int8_w8a8,
        B_bias=B_bias,
    )


def _invoke_fused_moe_grouped_matmul(
    A, B, C, A_scale, B_scale, topk_weights,
    sorted_token_ids, expert_ids, num_tokens_post_padded,
    mul_routed_weight, top_k, config,
    use_fp8_w8a8=False, use_int8_w8a8=False, B_bias=None,
):
    """High-performance path using npu_grouped_matmul.

    Reconstructs the expert->token mapping from sorted_token_ids/expert_ids,
    then uses npu_grouped_matmul with split_item=2 for a single kernel call.
    """
    import torch_npu

    E, N, K = B.shape  # B is [E, N, K] (weight per expert: out x in)
    c_flat = C.view(-1, N)
    num_valid_tokens = c_flat.shape[0]
    block_size = config["BLOCK_SIZE_M"]
    device = A.device

    if sorted_token_ids is None:
        # Naive decode path: expert_ids[p] is the expert for pair p.
        # A[p // top_k] is the input token.
        num_pairs = min(len(expert_ids), num_valid_tokens)
        expert_ids_cpu = expert_ids[:num_pairs].cpu()

        # Build per-expert token counts and gather indices
        # Sort pairs by expert for grouped_matmul
        sorted_expert_ids, sort_order = expert_ids_cpu.sort()
        sort_order_dev = sort_order.to(device)

        # Get unique experts and counts
        unique_experts, counts = torch.unique_consecutive(
            sorted_expert_ids, return_counts=True
        )

        # Skip if all invalid
        valid = unique_experts >= 0
        if not valid.any():
            return

        # Build group_list (cumulative token counts per expert, only for active experts)
        # For npu_grouped_matmul with split_item=2, we need the weight stacked in
        # expert order matching the group_list. But our experts may be sparse.
        # Instead, build a dense gathered input and use per-expert weight list.

        # Gather input tokens in sorted-by-expert order
        a_indices = (sort_order_dev // max(top_k, 1)).long()
        gathered_a = A[a_indices]  # [num_pairs, K]

        # Build group_list as cumsum of counts for valid experts
        valid_mask_cpu = unique_experts >= 0
        valid_experts = unique_experts[valid_mask_cpu]
        valid_counts = counts[valid_mask_cpu]

        # For npu_grouped_matmul, we need contiguous expert weights
        # Gather only the active expert weights
        valid_experts_dev = valid_experts.to(device).long()
        gathered_B = torch.index_select(B, 0, valid_experts_dev)  # [num_active, N, K]
        # Transpose: [num_active, N, K] -> [num_active, K, N] for x @ W
        gathered_B_t = gathered_B.transpose(1, 2).contiguous()

        # Filter out invalid experts from gathered_a
        # The sort puts negatives first (since -1 < 0 < valid experts)
        invalid_count = counts[~valid_mask_cpu].sum().item() if (~valid_mask_cpu).any() else 0
        if invalid_count > 0:
            gathered_a = gathered_a[invalid_count:]
            sort_order_dev = sort_order_dev[invalid_count:]

        if gathered_a.shape[0] == 0:
            return

        # group_list: cumulative counts (group_list_type=1)
        group_list = valid_counts.to(torch.int64).cumsum(0).to(device)

        # Grouped matmul: one kernel for all experts
        result = torch_npu.npu_grouped_matmul(
            x=[gathered_a],
            weight=[gathered_B_t],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
        )[0]

        if B_bias is not None:
            # Apply per-expert bias
            offset = 0
            for i, expert_id in enumerate(valid_experts.tolist()):
                cnt = valid_counts[i].item()
                result[offset:offset + cnt] += B_bias[expert_id]
                offset += cnt

        # Apply routing weights
        if mul_routed_weight and topk_weights is not None:
            topk_weights_flat = topk_weights.view(-1)
            w = topk_weights_flat[sort_order_dev].unsqueeze(-1)
            result = result * w.to(result.dtype)

        # Scatter back to output
        c_flat[sort_order_dev] = result.to(c_flat.dtype)

    else:
        # Aligned path: sorted_token_ids + expert_ids from moe_align_block_size
        total_padded = int(num_tokens_post_padded.cpu().item())
        sorted_ids_cpu = sorted_token_ids[:total_padded].cpu()
        expert_ids_cpu = expert_ids.cpu()

        # Build per-expert valid token lists (CPU, fast)
        expert_to_valid = {}
        num_blocks = len(expert_ids_cpu)
        for block_idx in range(num_blocks):
            eid = int(expert_ids_cpu[block_idx])
            if eid < 0:
                continue
            start = block_idx * block_size
            end = min(start + block_size, total_padded)
            if start >= end:
                break
            block_ids = sorted_ids_cpu[start:end].numpy()
            valid = block_ids[block_ids < num_valid_tokens]
            if len(valid) > 0:
                expert_to_valid.setdefault(eid, []).append(valid)

        if not expert_to_valid:
            return

        # Concatenate all valid ids per expert and sort experts
        sorted_experts = sorted(expert_to_valid.keys())
        all_valid_ids = []
        expert_counts = []
        for eid in sorted_experts:
            ids = np.concatenate(expert_to_valid[eid]).astype(np.int64)
            all_valid_ids.append(ids)
            expert_counts.append(len(ids))

        # Build tensors
        all_ids_np = np.concatenate(all_valid_ids)
        all_ids_dev = torch.from_numpy(all_ids_np).to(device)
        a_indices = all_ids_dev // max(top_k, 1)
        gathered_a = A[a_indices.long()]

        # Gather expert weights in order
        expert_ids_tensor = torch.tensor(sorted_experts, dtype=torch.int64, device=device)
        gathered_B = torch.index_select(B, 0, expert_ids_tensor)
        gathered_B_t = gathered_B.transpose(1, 2).contiguous()

        # group_list: cumulative counts
        group_list = torch.tensor(expert_counts, dtype=torch.int64, device=device).cumsum(0)

        # Grouped matmul
        result = torch_npu.npu_grouped_matmul(
            x=[gathered_a],
            weight=[gathered_B_t],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
        )[0]

        if B_bias is not None:
            offset = 0
            for i, eid in enumerate(sorted_experts):
                cnt = expert_counts[i]
                result[offset:offset + cnt] += B_bias[eid]
                offset += cnt

        if mul_routed_weight and topk_weights is not None:
            topk_weights_flat = topk_weights.view(-1)
            w = topk_weights_flat[all_ids_dev].unsqueeze(-1)
            result = result * w.to(result.dtype)

        c_flat[all_ids_dev] = result.to(c_flat.dtype)


def _invoke_fused_moe_loop(
    A, B, C, A_scale, B_scale, topk_weights,
    sorted_token_ids, expert_ids, num_tokens_post_padded,
    mul_routed_weight, top_k, config,
    use_fp8_w8a8=False, use_int8_w8a8=False, B_bias=None,
):
    """Fallback per-expert torch.mm loop."""
    N = B.shape[1]
    block_size = config["BLOCK_SIZE_M"]
    c_flat = C.view(-1, N)
    num_valid_tokens = c_flat.shape[0]

    if topk_weights is not None:
        topk_weights_flat = topk_weights.view(-1)
    else:
        topk_weights_flat = None

    device = A.device
    expert_indices = {}

    if sorted_token_ids is None:
        expert_ids_cpu = expert_ids.cpu().numpy()
        expert_batches = {}
        for pair_idx in range(len(expert_ids_cpu)):
            if pair_idx >= num_valid_tokens:
                break
            expert_id = int(expert_ids_cpu[pair_idx])
            if expert_id < 0:
                continue
            expert_batches.setdefault(expert_id, []).append(pair_idx)
        for expert_id, rows in expert_batches.items():
            valid_ids = torch.tensor(rows, dtype=torch.int64, device=device)
            a_idx = valid_ids // max(top_k, 1)
            expert_indices[expert_id] = (valid_ids, a_idx)
    else:
        sorted_ids_cpu = sorted_token_ids.cpu().numpy()
        expert_ids_cpu = expert_ids.cpu().numpy()
        total_padded = int(num_tokens_post_padded.cpu().item())

        expert_batches = {}
        num_blocks = len(expert_ids_cpu)
        for block_idx in range(num_blocks):
            expert_id = int(expert_ids_cpu[block_idx])
            if expert_id < 0:
                continue
            start = block_idx * block_size
            end = min(start + block_size, total_padded)
            if start >= end:
                break
            block_ids = sorted_ids_cpu[start:end]
            valid = block_ids[block_ids < num_valid_tokens]
            if len(valid) == 0:
                continue
            expert_batches.setdefault(expert_id, []).append(valid)

        for expert_id, id_arrays in expert_batches.items():
            all_valid = np.concatenate(id_arrays).astype(np.int64)
            valid_ids = torch.from_numpy(all_valid).to(device)
            a_idx = valid_ids // max(top_k, 1)
            expert_indices[expert_id] = (valid_ids, a_idx)

    for expert_id, (valid_ids, a_idx) in expert_indices.items():
        a_block = A[a_idx]
        out = torch.mm(a_block, B[expert_id].t())

        if B_bias is not None:
            out = out + B_bias[expert_id]

        if mul_routed_weight and topk_weights_flat is not None:
            w = topk_weights_flat[valid_ids].unsqueeze(-1)
            out = out * w.to(out.dtype)

        c_flat[valid_ids] = out.to(c_flat.dtype)

    del expert_indices
