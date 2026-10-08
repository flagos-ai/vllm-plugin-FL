# SPDX-License-Identifier: Apache-2.0
"""Indexer reference decompositions; not a generic graph backend."""

import torch

INDEX_HEAD_DIM = 128


def _dequantize_grouped(
    values: torch.Tensor, scales: torch.Tensor | None
) -> torch.Tensor:
    output = values.float()
    if scales is None:
        return output
    scales = scales.float()
    if scales.ndim == output.ndim - 1:
        return output * scales.unsqueeze(-1)
    num_groups = scales.shape[-1]
    if num_groups == 1:
        return output * scales
    if output.shape[-1] % num_groups:
        raise ValueError("Quantized width must be divisible by the scale groups")
    group_size = output.shape[-1] // num_groups
    grouped = output.reshape(*output.shape[:-1], num_groups, group_size)
    return (grouped * scales.unsqueeze(-1)).reshape_as(output)


def _torch_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    del clean_logits
    q_values, q_scale = q
    k_values, k_scale = kv
    q_float = _dequantize_grouped(q_values, q_scale)
    k_float = _dequantize_grouped(k_values, k_scale)
    score = torch.einsum("mhd,nd->hmn", q_float, k_float)
    logits = (score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)).sum(0)
    columns = torch.arange(k_values.shape[0], device=q_values.device).unsqueeze(0)
    valid = (columns >= cu_seqlen_ks.reshape(-1, 1)) & (
        columns < cu_seqlen_ke.reshape(-1, 1)
    )
    return logits.masked_fill(~valid, float("-inf"))


def _torch_pack_seq(
    tensor: torch.Tensor, lengths: torch.Tensor, pad_value=-float("inf")
) -> torch.Tensor:
    lengths_cpu = lengths.detach().to("cpu", torch.int64).tolist()
    max_length = max(lengths_cpu, default=0)
    out = torch.full(
        (len(lengths_cpu), max_length, *tensor.shape[1:]),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    cursor = 0
    for request, length in enumerate(lengths_cpu):
        out[request, :length].copy_(tensor[cursor : cursor + length])
        cursor += length
    return out


def _torch_unpack_seq(tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    lengths_cpu = lengths.detach().to("cpu", torch.int64).tolist()
    pieces = [tensor[request, :length] for request, length in enumerate(lengths_cpu)]
    if not pieces:
        return tensor.new_empty((0, *tensor.shape[2:]))
    return torch.cat(pieces, dim=0)


def rotate_indexer_query(q):
    if q.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("Indexer rotation requires head_dim=128")
    return hadamard128(q.float()).to(q.dtype)


def hadamard128(x: torch.Tensor) -> torch.Tensor:
    """Normalized Walsh-Hadamard transform on the last dimension."""
    if x.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError(f"Hadamard input must have dim 128, got {x.shape[-1]}")
    out = x.float()
    width = 1
    while width < INDEX_HEAD_DIM:
        pair = out.reshape(*out.shape[:-1], -1, 2, width)
        left, right = pair.unbind(dim=-2)
        out = torch.stack((left + right, left - right), dim=-2).reshape_as(out)
        width *= 2
    return out * (INDEX_HEAD_DIM**-0.5)


def expand_pools_to_tokens(
    group_ids: torch.Tensor,
    group_valid: torch.Tensor,
    topk: int,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    if topk % pool_size:
        raise ValueError("topk must be divisible by pool_size")
    offsets = torch.arange(pool_size, device=group_ids.device, dtype=torch.int64)
    token_ids = (group_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets).reshape(
        group_ids.shape[0], topk
    )
    valid = group_valid.unsqueeze(-1).expand(-1, -1, pool_size).reshape_as(token_ids)
    if page_table is not None:
        safe_ids = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
        output = torch.gather(page_table, 1, safe_ids).to(torch.int32)
    elif topk_offsets is not None:
        output = (token_ids + topk_offsets.reshape(-1, 1).to(torch.int64)).to(
            torch.int32
        )
    else:
        output = token_ids.to(torch.int32)
    return torch.where(valid, output, torch.full_like(output, -1))


def append_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_lens: torch.Tensor,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    if pool_size == 1:
        return topk_result
    rows, history_len = topk_result.shape
    out_cols = history_len + pool_size - 1
    cols = torch.arange(out_cols, device=topk_result.device).unsqueeze(0)
    is_history = cols < history_len
    safe_hist = cols.clamp(max=history_len - 1).expand(rows, -1)
    history = torch.gather(topk_result, 1, safe_hist)
    tail_start = pool_lens.to(torch.int32) * pool_size
    tail_offset = cols - history_len
    tail_count = seq_lens.to(torch.int32) - tail_start
    is_tail = (tail_offset >= 0) & (tail_offset < tail_count.unsqueeze(1))
    tail_raw = tail_start.unsqueeze(1) + tail_offset
    if page_table is not None:
        safe_tail = tail_raw.clamp(min=0, max=page_table.shape[1] - 1)
        tail = torch.gather(page_table, 1, safe_tail).to(torch.int32)
    elif topk_offsets is not None:
        tail = (tail_raw + topk_offsets.reshape(-1, 1).to(torch.int64)).to(torch.int32)
    else:
        tail = tail_raw.to(torch.int32)
    out = torch.where(is_history, history, torch.full_like(history, -1))
    return torch.where(is_tail, tail, out)
