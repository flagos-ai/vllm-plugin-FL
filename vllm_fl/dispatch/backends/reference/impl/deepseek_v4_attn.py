# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference (PyTorch) implementations for DeepseekV4 attention operators.
"""

### TODO(lms): support reference version
import torch


def deepseek_v4_fp8_einsum_torch(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    """
    Reference implementation of deepseek_v4_fp8_einsum using vLLM's fp8_einsum.

    Mutates `out` in-place.
    """
    pass


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    swa_kv_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    block_size: int,
) -> None:
    """
    Reference implementation of fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.

    Mutates q, swa_kv_cache_2d in-place.
    """
    pass


# ==================== Sparse Attention Indexer Ops ====================


def combine_topk_swa_indices_torch(
    topk_indices: torch.Tensor,
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_tokens: int,
    window_size: int,
    compress_ratio: int,
    block_size: int,
) -> None:
    """Reference implementation of combine_topk_swa_indices."""
    pass


def compute_global_topk_indices_and_lens_torch(
    topk_indices: torch.Tensor,
    global_indices: torch.Tensor,
    global_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    topk_tokens: int,
    compress_ratio: int,
    block_size: int,
) -> None:
    """Reference implementation of compute_global_topk_indices_and_lens."""
    pass

def dequantize_and_gather_k_cache_torch(
    k_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    block_size: int,
) -> None:
    """Reference implementation of dequantize_and_gather_k_cache."""
    pass


def fused_indexer_q_rope_quant_torch(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = False,
):
    """Reference implementation of fused_indexer_q_rope_quant."""
    pass

def fused_inv_rope_fp8_quant_torch(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    quant_group_size: int,
    chunks_per_head: int,
    rope_start: int,
    half_rope: int,
    tma_aligned_scales: bool,
    fp8_max: float,
    tma_aligned_T: int,
    num_tokens: int,
    n_groups: int,
    d: int,
    scale_inner: int,
):
    """Reference implementation of fused_inv_rope_fp8_quant."""


def fused_q_kv_rmsnorm_torch(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation of fused_q_kv_rmsnorm."""
    pass


def indexer_k_quant_and_cache_torch(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str,
) -> None:
    """Reference implementation of indexer_k_quant_and_cache."""
    pass


def cp_gather_indexer_k_quant_cache_torch(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    """Reference implementation of cp_gather_indexer_k_quant_cache."""
    pass

def top_k_per_row_prefill_torch(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    """Reference implementation of top_k_per_row_prefill."""
    pass


def pack_seq_triton_torch(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float | int = -float("inf"),
) -> torch.Tensor:
    """Reference implementation of pack_seq_triton."""
    pass


def top_k_per_row_decode_torch(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    """Reference implementation of top_k_per_row_decode."""
    pass


def unpack_seq_triton_torch(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of unpack_seq_triton."""
    pass
