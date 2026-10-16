# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

import math

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_NATIVE_MM_MIN_SEQUENCE_LENGTH = 8192
_NATIVE_MM_HEAD_DIMS = (64, 72, 96, 128)


def supports_native_mm_attention(head_size, dtype):
    return head_size in _NATIVE_MM_HEAD_DIMS and dtype in (
        torch.float16,
        torch.bfloat16,
    )


def forward_mm_attention(
    self, query, key, value, cu_seqlens=None, max_seqlen=None, sequence_lengths=None
):
    """Vendor provider entry point; explicit SDPA/reference never enters it."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if self.attn_backend == AttentionBackendEnum.TORCH_SDPA:
        return self._forward_sdpa(query, key, value, cu_seqlens)
    if self.attn_backend != AttentionBackendEnum.FLASH_ATTN:
        raise ValueError(
            f"Unsupported Kunlunxin MM attention backend: {self.attn_backend}"
        )

    # The fused provider is self-attention only. Preserve SDPA's cross-attention
    # contract before splitting Q/K/V with a shared sequence-length list.
    if query.size(1) != key.size(1):
        return self._forward_sdpa(query, key, value, cu_seqlens)

    batch_size, query_length = query.shape[:2]
    reshaped = query.ndim != 4
    query, key, value = self.view_qkv_to_4d(
        query, key, value, batch_size, query_length, key.size(1)
    )
    lengths = (
        (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        if cu_seqlens is not None
        else [query_length]
    )
    outputs = []
    for q_part, k_part, v_part in zip(
        torch.split(query, lengths, dim=1),
        torch.split(key, lengths, dim=1),
        torch.split(value, lengths, dim=1),
    ):
        output = try_native_large_mm_attention(q_part, k_part, v_part, self.scale)
        if output is None:
            output = self._forward_sdpa(q_part, k_part, v_part)
        outputs.append(output)
    output = torch.cat(outputs, dim=1)
    return output.reshape(batch_size, query_length, -1) if reshaped else output


def try_native_large_mm_attention(query, key, value, scale):
    """Use fused self-attention for large BLHD multimodal sequences.

    Chunked math SDPA bounds score storage, but repeats scaling and full-key
    reads for every query chunk.  The worst-case MM profiling input can have
    65,536 vision tokens.  Use the vendor's memory-efficient kernel for the
    validated dtypes/head dimensions; other shapes retain the SDPA path.
    """
    if (
        query.ndim != 4
        or query.shape != key.shape
        or query.shape != value.shape
        or query.shape[1] < _NATIVE_MM_MIN_SEQUENCE_LENGTH
        or query.shape[-1] not in _NATIVE_MM_HEAD_DIMS
        or query.dtype not in (torch.float16, torch.bfloat16)
        or key.dtype != query.dtype
        or value.dtype != query.dtype
    ):
        return None

    import xtorch_ops

    batch_size, sequence_length, num_heads, head_dim = query.shape
    logger.info_once(
        "Using native Kunlunxin attention for long MM sequences "
        "(length=%d, head_dim=%d).",
        sequence_length,
        head_dim,
    )
    cu_seqlens_cpu = (
        torch.arange(batch_size + 1, dtype=torch.int32, device="cpu") * sequence_length
    )
    cu_seqlens = cu_seqlens_cpu.to(query.device)

    # The vendor binding calls this argument softmax_scale, but its kernel
    # applies an additional 1/sqrt(head_dim). Convert the SDPA scale to the
    # kernel's multiplicative alpha, as in Kunlunxin prefill_attention.
    alpha = scale * math.sqrt(head_dim)
    output, _ = xtorch_ops.flash_attn_varlen_func(
        query.contiguous().view(-1, num_heads, head_dim),
        key.contiguous().view(-1, num_heads, head_dim),
        value.contiguous().view(-1, num_heads, head_dim),
        cu_seqlens,
        cu_seqlens,
        sequence_length,
        sequence_length,
        dropout_p=0.0,
        softmax_scale=alpha,
        causal=False,
        is_varlen=True,
        is_prefill=True,
        cu_seqlens_qo_cpu=cu_seqlens_cpu,
    )
    return output.view(batch_size, sequence_length, num_heads, head_dim)
