# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch


def rotary_embedding_maca(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapt split FL caches to vLLM's in-place, packed-cache CUDA ABI."""
    from vllm._custom_ops import rotary_embedding

    if cos.shape[-1] == query.shape[-1]:
        # The FL interface also accepts independent full-width cos/sin. The
        # native kernel shares one coefficient between each rotary pair and
        # cannot represent that general case. Preserve the reference contract.
        from vllm_fl.dispatch.backends.reference.impl.rotary import (
            rotary_embedding_torch,
        )

        q_out, k_out = rotary_embedding_torch(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=False,
        )
        if inplace:
            query.copy_(q_out)
            key.copy_(k_out)
            return query, key
        return q_out, k_out

    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[-1] * 2 != query.shape[-1]:
        raise ValueError(
            "MACA rotary expects matching half-width or full-width cos/sin caches"
        )
    if (
        query.shape[-1] != key.shape[-1]
        or query.dtype != key.dtype
        or query.device != key.device
    ):
        raise ValueError(
            "query and key must have the same head dimension, device and dtype"
        )
    if query.shape[-1] <= 0 or query.shape[-1] % 2:
        raise ValueError("the rotary head dimension must be a positive even number")
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("position_ids must have dtype int32 or int64")
    if query.ndim not in (3, 4) or key.ndim != query.ndim:
        raise ValueError("query and key must both be 3D or both be 4D")

    query_shape, key_shape = query.shape, key.shape
    if query.ndim == 4:
        # FL's batched interface is [batch, heads, sequence, rotary_dim].
        batch, _, seq_len, _ = query.shape
        if key.shape[0] != batch or key.shape[2] != seq_len:
            raise ValueError("query and key require matching batch/sequence dimensions")
        if position_ids.ndim == 1:
            if position_ids.numel() != seq_len:
                raise ValueError("shared position_ids must match the sequence length")
            position_ids = position_ids.expand(batch, -1)
        elif position_ids.shape != (batch, seq_len):
            raise ValueError("batched position_ids must have shape [batch, sequence]")
        query_work = (
            query.transpose(1, 2).contiguous().view(-1, query.shape[1], query.shape[-1])
        )
        key_work = (
            key.transpose(1, 2).contiguous().view(-1, key.shape[1], key.shape[-1])
        )
    else:
        if (
            position_ids.ndim != 1
            or query.shape[0] != key.shape[0]
            or position_ids.numel() != query.shape[0]
        ):
            raise ValueError(
                "3D query, key and position_ids must have matching token counts"
            )
        # Partial-RoPE and packed-QKV views have gaps between heads. Native
        # kernels address heads with head_size, so compact them before launch.
        query_work, key_work = query.contiguous(), key.contiguous()
    if not inplace:
        query_work, key_work = query_work.clone(), key_work.clone()

    cache = torch.cat((cos, sin), dim=-1).to(device=query.device, dtype=query.dtype)
    rotary_embedding(
        position_ids.to(device=query.device, dtype=torch.long).flatten(),
        query_work,
        key_work,
        query.shape[-1],
        cache,
        not rotary_interleaved,
    )
    if query.ndim == 4:
        query_work = query_work.view(
            query_shape[0], query_shape[2], query_shape[1], query_shape[3]
        ).transpose(1, 2)
        key_work = key_work.view(
            key_shape[0], key_shape[2], key_shape[1], key_shape[3]
        ).transpose(1, 2)
    if inplace:
        if query_work is not query:
            query.copy_(query_work)
        if key_work is not key:
            key.copy_(key_work)
        return query, key
    return query_work, key_work
