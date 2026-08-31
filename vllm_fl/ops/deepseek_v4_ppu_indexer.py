# Copyright (c) 2026 BAAI. All rights reserved.
"""PPU sparse-attn indexer for DeepSeek-V4 (int8/fp8 Q via PPU deep_gemm).

Ported from the T-Head vendor fork (v1/attention/ops/ppu_mla_sparse.py):
the upstream sparse_attn_indexer drives NVIDIA fp8_fp4_mqa_logits, which
requires q.dtype == k.dtype; on PPU the indexer Q is INT8 (see
deepseek_v4_indexer_q.py) and the mqa logits must go through deep_gemm's
int8_(paged_)mqa_logits entry points. Differences from the vendor source:
  - deep_gemm helpers are imported directly from the PPU deep_gemm wheel
    (the fork routed them through vllm.utils.ppu_deep_gemm, absent here)
  - torch.ops._C.top_k_per_row_* / persistent_topk are replaced with
    FlagGems' top_k_per_row_prefill/decode (persistent_topk unavailable,
    top_k_per_row_decode covers the topk=512 case)
  - registered as torch.ops.vllm.fl_ppu_sparse_attn_indexer
"""
import functools  # noqa: F401
import importlib  # noqa: F401

import torch

import vllm.envs as envs  # noqa: F401
from vllm.logger import init_logger
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,  # noqa: F401
    _resolve_layer_name,
    direct_register_custom_op,
)

from vllm import _custom_ops as ops

from deep_gemm import (
    fp8_mqa_logits,
    fp8_paged_mqa_logits,
    int8_mqa_logits,
    int8_paged_mqa_logits,
)


def is_deep_gemm_supported() -> bool:
    return True


logger = init_logger(__name__)


RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    q_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), q_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


def ppu_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    q_dtype = q_quant.dtype
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, q_dtype, use_fp4_cache
        )

        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return ppu_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, q_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks:
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                if not getattr(ppu_sparse_attn_indexer, "_probed", False):
                    ppu_sparse_attn_indexer._probed = True
                    logger.warning(
                        "[fl-probe] kv_cache shape=%s stride=%s dtype=%s | "
                        "k_quant shape=%s dtype=%s | k_scale shape=%s | "
                        "block_table shape=%s dtype=%s | cu_seq_lens shape=%s | "
                        "total_seq_lens=%s num_reqs=%s head_dim=%s",
                        tuple(kv_cache.shape), tuple(kv_cache.stride()),
                        kv_cache.dtype, tuple(k_quant.shape), k_quant.dtype,
                        tuple(k_scale.shape), tuple(chunk.block_table.shape),
                        chunk.block_table.dtype, tuple(chunk.cu_seq_lens.shape),
                        chunk.total_seq_lens, chunk.num_reqs, head_dim,
                    )
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)

            if is_deep_gemm_supported():
                if use_fp4_cache:
                    raise RuntimeError("do not support fp4 indexer now")
                    # logits = fp8_fp4_mqa_logits(
                    #     (q_slice_cast, q_scale_slice),
                    #     (k_quant_cast, k_scale_cast),
                    #     weights[chunk.token_start : chunk.token_end],
                    #     chunk.cu_seqlen_ks,
                    #     chunk.cu_seqlen_ke,
                    #     clean_logits=False,
                    # )
                if q_dtype == torch.int8:
                    logits = int8_mqa_logits(
                        q_slice_cast,
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        clean_logits=False,
                    )
                elif q_dtype == torch.float8_e4m3fn:
                    logits = fp8_mqa_logits(
                        q_slice_cast,
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        clean_logits=False,
                    )
                else:
                    raise RuntimeError("PPU mqa_logtis only support int8 on btv1.0 and fp8 on >= btv1.5")
            else:
                raise RuntimeError("indexer need PPU deep gemm installed")

            num_rows = logits.shape[0]

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            ops.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )

        if is_deep_gemm_supported():
            if use_fp4_cache:
                raise RuntimeError("ppu do not support fp4 indexer now")
            if q_dtype == torch.int8:
                # logits = torch.ones([batch_size * next_n, max_model_len],
                #                      device=q_quant.device,
                #                      dtype=torch.float32)
                logits = int8_paged_mqa_logits(
                    padded_q_quant_cast,
                    kv_cache,
                    weights[:num_padded_tokens],
                    seq_lens,
                    decode_metadata.block_table,
                    decode_metadata.schedule_metadata,
                    max_context_len=max_model_len,
                    clean_logits=False,
                )
            elif q_dtype == torch.float8_e4m3fn:
                logits = fp8_paged_mqa_logits(
                    padded_q_quant_cast,
                    kv_cache,
                    weights[:num_padded_tokens],
                    seq_lens,
                    decode_metadata.block_table,
                    decode_metadata.schedule_metadata,
                    max_context_len=max_model_len,
                    clean_logits=False,
                )
            else:
                raise RuntimeError("PPU mqa_logtis only support int8 on btv1.0 and fp8 on >= btv1.5")
        else:
            raise RuntimeError("indexer need ppu deep gemm installed")
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if False:  # persistent_topk unavailable on this stack; use per-row decode
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def ppu_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer




direct_register_custom_op(
    op_name="fl_ppu_sparse_attn_indexer",
    op_func=ppu_sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=ppu_sparse_attn_indexer_fake,
)
