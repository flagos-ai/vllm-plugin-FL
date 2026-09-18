# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.worker.workspace import current_workspace_manager

from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm_fl.dispatch import CachedOp

# Runtime patch: graph-visible native prefill topK custom op. Import registers op before capture.
try:
    from . import prefill_topk_native_op as _prefill_topk_native_op  # noqa: F401
except Exception:
    _prefill_topk_native_op = None

_indexer_k_quant_and_cache = CachedOp("indexer_k_quant_and_cache")
_cp_gather_indexer_k_quant_cache = CachedOp("cp_gather_indexer_k_quant_cache")
_top_k_per_row_prefill = CachedOp("top_k_per_row_prefill")
_pack_seq_triton = CachedOp("pack_seq_triton")
_top_k_per_row_decode = CachedOp("top_k_per_row_decode")
_unpack_seq_triton = CachedOp("unpack_seq_triton")

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _dequantize_fp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return values.to(torch.bfloat16) * scales.to(torch.bfloat16).unsqueeze(-1)


# --- sparse_indexer_topk_fix (runtime patch, see scripts/sparse_indexer_topk_fix) ---
# Background: the reference impls of top_k_per_row_prefill/decode are no-op
# `pass`, and on MetaX the decode CachedOp resolves to that no-op while the
# flagos decode Triton kernel fails to compile. As a result every decode step
# left topk_indices_buffer at its -1 init and sparse attention selected nothing.
# This helper restores the semantics:
#   valid_len <= topk_tokens -> select ALL valid compressed entries (ks..ke-1),
#                               rest stays -1 (no kernel);
#   valid_len >  topk_tokens -> real top-k: flagos kernel (prefill, works on
#                               MetaX) or torch-native masked topk (decode,
#                               flagos kernel unavailable).

_TOPK_FIX_STATE: dict[str, int] = {}


def _log_topk_fix(kind: str, msg: str) -> None:
    pass


def _fill_select_all(
    ks: torch.Tensor,
    valid: torch.Tensor,
    out: torch.Tensor,
    topk_tokens: int,
) -> None:
    """out[r, j] = ks[r] + j for j < valid[r], else -1."""
    arange_k = torch.arange(topk_tokens, device=out.device, dtype=torch.long)
    filled = arange_k.unsqueeze(0) + ks.unsqueeze(1)
    filled = torch.where(
        arange_k.unsqueeze(0) < valid.unsqueeze(1),
        filled,
        torch.full_like(filled, -1),
    )
    out.copy_(filled.to(out.dtype))


def _topk_per_row_fixed(
    logits: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    out: torch.Tensor,
    topk_tokens: int,
    tag: str,
    kernel=None,
    kernel_args=(),
    max_valid_hint: int | None = None,
) -> None:
    num_rows = logits.shape[0]
    ks1 = ks.reshape(-1).to(torch.long)
    ke1 = ke.reshape(-1).to(torch.long)
    valid = (ke1 - ks1).clamp(min=0)
    max_valid = (
        max_valid_hint if max_valid_hint is not None else int(valid.max().item())
    )

    if max_valid <= topk_tokens:
        _fill_select_all(ks1, valid, out, topk_tokens)
        _log_topk_fix(f"{tag}-all", f"rows={num_rows} max_valid={max_valid}")
        return

    if (
        tag == "prefill"
        and os.environ.get("VLLM_FL_METAX_PREFILL_NATIVE_TOPK", "0") == "1"
    ):
        if _prefill_topk_native_op is None:
            raise RuntimeError(
                "VLLM_FL_METAX_PREFILL_NATIVE_TOPK=1 but prefill_topk_native_op import failed"
            )
        torch.ops.vllm_fl.dsv4_prefill_topk_native(
            logits,
            ks.reshape(-1).to(torch.int32),
            ke.reshape(-1).to(torch.int32),
            out,
            topk_tokens,
        )
        _log_topk_fix("prefill-native-topk", f"rows={num_rows} max_valid={max_valid}")
        return

    if kernel is not None:
        # Production kernel (prefill flagos) writes all rows; short rows are
        # re-filled below to guarantee select-all semantics.
        kernel(*kernel_args)
        _log_topk_fix(f"{tag}-kernel", f"rows={num_rows} max_valid={max_valid}")
        # GPU-only merge for short rows (valid <= topk_tokens): select-all
        # indices, no host sync (.item()/.any()/nonzero) -> graph-safe.
        short = valid <= topk_tokens
        arange_k = torch.arange(topk_tokens, device=out.device, dtype=torch.long)
        filled = arange_k.unsqueeze(0) + ks1.unsqueeze(1)
        filled = torch.where(
            arange_k.unsqueeze(0) < valid.unsqueeze(1),
            filled,
            torch.full_like(filled, -1),
        )
        merged = torch.where(short.unsqueeze(1), filled.to(out.dtype), out)
        out.copy_(merged)
        return

    # Decode path with long rows: flagos decode top-k kernel fails to compile
    # on MetaX (CompilationError). Use the MetaX-verified Triton tree top-k
    # (scripts/sparse_indexer_tree_topk/tree_topk.py, exact top-K + select-all
    # semantics, standalone-verified for N=753..8192). No reference/no-op
    # fallback is allowed here: a failure must raise.
    from .tree_topk import decode_topk_tree

    decode_topk_tree(logits, valid, out, topk_tokens)
    _log_topk_fix(
        f"{tag}-tree",
        f"rows={num_rows} max_valid={max_valid} (metax tree top-k)",
    )


# --- end sparse_indexer_topk_fix ---


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
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
        ((total_seq_lens, head_dim), fp8_dtype),
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


def deepseek_v4_metax_indexer(
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
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        _log_topk_fix(
            "dummy", "attn_metadata not dict -> fake return (capture/profile run)"
        )
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
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

        return deepseek_v4_metax_indexer_fake(
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
    _log_topk_fix(
        "entry",
        f"has_prefill={has_prefill} has_decode={has_decode} num_tokens={hidden_states.shape[0]}",
    )

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
        _indexer_k_quant_and_cache(
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
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk_index, chunk in enumerate(prefill_metadata.chunks):
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                _cp_gather_indexer_k_quant_cache(
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

            if current_platform.vendor_name == "metax" and not use_fp4_cache:
                from .prefill_indexer_rowshard import (
                    bf16_mqa_logits_prefill as bf16_mqa_logits,
                )

                from .indexer_m4 import try_prefill_indexer_m4

                q_bf16 = q_slice_cast.to(torch.bfloat16)
                kv_bf16 = _dequantize_fp8(k_quant_cast, k_scale_cast)

                def topk_into_local(scores, starts, ends, output):
                    _topk_per_row_fixed(
                        scores,
                        starts,
                        ends,
                        output,
                        topk_tokens,
                        "prefill",
                        max_valid_hint=chunk.total_seq_lens,
                        kernel=_top_k_per_row_prefill,
                        kernel_args=(
                            scores,
                            starts,
                            ends,
                            output,
                            scores.shape[0],
                            scores.stride(0),
                            scores.stride(1),
                            topk_tokens,
                        ),
                    )

                if try_prefill_indexer_m4(
                    q_bf16,
                    kv_bf16,
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices_buffer[
                        chunk.token_start : chunk.token_end, :topk_tokens
                    ],
                    topk_tokens,
                    topk_into_local,
                    bf16_mqa_logits,
                ):
                    continue
                logits = bf16_mqa_logits(
                    q_bf16,
                    kv_bf16,
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                )
            else:
                logits = fp8_fp4_mqa_logits(
                    (q_slice_cast, q_scale_slice),
                    (k_quant_cast, k_scale_cast),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
            num_rows = logits.shape[0]

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            _topk_per_row_fixed(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                topk_tokens,
                "prefill",
                max_valid_hint=chunk.total_seq_lens,
                kernel=_top_k_per_row_prefill,
                kernel_args=(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                ),
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
                padded_q_quant_decode_tokens = _pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = _pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = _pack_seq_triton(
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

        graphsafe_decode = (
            os.environ.get("VLLM_FL_METAX_GRAPHSAFE_DECODE_INDEXER", "0") == "1"
        )
        decode_static_context_len = max_model_len
        if current_platform.vendor_name == "metax" and not use_fp4_cache:
            from .indexer_reference import bf16_mqa_logits

            context_lens_per_batch = seq_lens.amax(dim=1)
            cu_seq_lens = torch.zeros(
                batch_size + 1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            torch.cumsum(context_lens_per_batch, dim=0, out=cu_seq_lens[1:])

            workspace_manager = current_workspace_manager()
            if graphsafe_decode:
                # CUDA graph capture cannot tolerate the original decode path CPU sync,
                # dynamic max shape, or tensor shapes derived from runtime lengths.
                # Use static per-sequence decode slots. For batch_size==1 we preserve
                # the actual gather length that already passed smoke. For batch_size>1
                # we gather a fixed context window per sequence, then real seq_lens mask
                # logits/top-k semantics. This avoids SPLIT_ATTENTION fallback while a
                # proper fused paged logits kernel is still missing on MetaX.
                static_context_len = int(decode_static_context_len)
                if batch_size == 1:
                    gather_cu_seq_lens = cu_seq_lens
                    static_total_context_len = static_context_len
                else:
                    gather_cu_seq_lens = (
                        torch.arange(
                            batch_size + 1,
                            dtype=torch.int32,
                            device=seq_lens.device,
                        )
                        * static_context_len
                    )
                    static_total_context_len = batch_size * static_context_len
                use_paged_indexer_logits = (
                    os.environ.get("VLLM_FL_METAX_PAGED_INDEXER_LOGITS", "0") == "1"
                    and next_n == 1
                    and head_dim == 128
                )
                logits = torch.empty(
                    (num_padded_tokens, static_context_len),
                    dtype=torch.float32,
                    device=kv_cache.device,
                )
                if use_paged_indexer_logits:
                    # Prototype: direct paged read of indexer KV cache, semantically
                    # matching cp_gather_indexer_k_quant_cache + batched_mqa_logits.
                    # Does not bypass logits/topk/indexer; only removes the contiguous
                    # k_quant/k_scale workspace materialization in decode next_n=1.
                    try:
                        from .paged_indexer_logits import (
                            paged_indexer_mqa_logits_match,
                        )
                    except ImportError as exc:
                        raise RuntimeError(
                            "paged_indexer_logits: paged_indexer_logits.py not importable"
                        ) from exc
                    q_b = padded_q_quant_cast.to(torch.bfloat16).reshape(
                        batch_size, -1, padded_q_quant_cast.shape[-1]
                    )
                    w_b = weights[:num_padded_tokens].reshape(batch_size, -1)
                    valid_b = seq_lens.amax(dim=1).to(torch.int32)
                    paged_indexer_mqa_logits_match(
                        q_b,
                        kv_cache,
                        w_b,
                        valid_b,
                        decode_metadata.block_table,
                        logits,
                        static_context_len,
                    )
                else:
                    values_spec, scales_spec = _gather_workspace_shapes(
                        max(total_seq_lens, static_total_context_len),
                        head_dim,
                        fp8_dtype,
                        False,
                    )
                    k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
                        values_spec,
                        scales_spec,
                    )
                    k_quant = k_quant_full[:static_total_context_len]
                    k_scale = k_scale_full[:static_total_context_len]
                    _cp_gather_indexer_k_quant_cache(
                        kv_cache,
                        k_quant,
                        k_scale,
                        decode_metadata.block_table,
                        gather_cu_seq_lens,
                    )
                    # Fused single-launch logits: replaces 64 x (_dequantize_fp8 +
                    # bf16_mqa_logits) launches. Exact per-bit semantics vs the
                    # loop version (standalone-verified N=1024..8192).
                    try:
                        from .batched_logits import batched_mqa_logits
                    except ImportError as exc:
                        raise RuntimeError(
                            "fused_logits: batched_logits.py not importable"
                        ) from exc
                    if next_n == 1:
                        q_b = padded_q_quant_cast.to(torch.bfloat16).reshape(
                            batch_size, -1, padded_q_quant_cast.shape[-1]
                        )
                        w_b = weights[:num_padded_tokens].reshape(batch_size, -1)
                        valid_b = seq_lens.amax(dim=1).to(torch.int32)
                        batched_mqa_logits(
                            q_b,
                            k_quant,
                            k_scale.view(torch.float32).squeeze(-1),
                            w_b,
                            valid_b,
                            logits,
                            static_context_len,
                        )
                    else:
                        for batch_idx in range(batch_size):
                            seq_start = batch_idx * static_context_len
                            seq_end = seq_start + static_context_len
                            k_bf16 = _dequantize_fp8(
                                k_quant[seq_start:seq_end],
                                k_scale[seq_start:seq_end]
                                .view(torch.float32)
                                .squeeze(-1),
                            )
                            q_bf16 = padded_q_quant_cast[batch_idx].to(torch.bfloat16)
                            context_lens = seq_lens[batch_idx].reshape(-1)
                            starts = torch.zeros_like(context_lens)
                            row_start = batch_idx * next_n
                            logits[row_start : row_start + next_n, :] = bf16_mqa_logits(
                                q_bf16,
                                k_bf16,
                                weights[row_start : row_start + next_n],
                                starts,
                                context_lens,
                            )
            else:
                seq_offsets = cu_seq_lens.tolist()
                total_context_len = seq_offsets[-1]

                # The gather must hold the WHOLE decode batch's compressed context,
                # which can exceed the static prefill-oriented sizing
                # (max_model_len * 40 // compress_ratio). Request the actual need;
                # the workspace manager grows on demand. Otherwise the slice below
                # silently truncates trailing sequences and the per-sequence logits
                # assignment shape-mismatches (worker crash at long context).
                values_spec, scales_spec = _gather_workspace_shapes(
                    max(total_seq_lens, total_context_len), head_dim, fp8_dtype, False
                )
                k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
                    values_spec,
                    scales_spec,
                )
                k_quant = k_quant_full[:total_context_len]
                k_scale = k_scale_full[:total_context_len]
                _cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    decode_metadata.block_table,
                    cu_seq_lens,
                )

                max_context_len = max(
                    seq_offsets[i + 1] - seq_offsets[i] for i in range(batch_size)
                )
                logits = torch.full(
                    (num_padded_tokens, max_context_len),
                    -float("inf"),
                    dtype=torch.float32,
                    device=kv_cache.device,
                )
                for batch_idx in range(batch_size):
                    context_lens = seq_lens[batch_idx].reshape(-1)
                    seq_start = seq_offsets[batch_idx]
                    seq_end = seq_offsets[batch_idx + 1]
                    k_bf16 = _dequantize_fp8(
                        k_quant[seq_start:seq_end],
                        k_scale[seq_start:seq_end].view(torch.float32).squeeze(-1),
                    )
                    q_bf16 = padded_q_quant_cast[batch_idx].to(torch.bfloat16)
                    starts = torch.zeros_like(context_lens)
                    row_start = batch_idx * next_n
                    logits[row_start : row_start + next_n, : seq_end - seq_start] = (
                        bf16_mqa_logits(
                            q_bf16,
                            k_bf16,
                            weights[row_start : row_start + next_n],
                            starts,
                            context_lens,
                        )
                    )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        _topk_per_row_fixed(
            logits,
            torch.zeros(num_rows, dtype=torch.long, device=logits.device),
            seq_lens.reshape(-1)[:num_rows],
            topk_indices,
            topk_tokens,
            "decode",
            kernel=None,
            max_valid_hint=(
                decode_static_context_len
                if graphsafe_decode
                else (
                    max(seq_offsets[i + 1] - seq_offsets[i] for i in range(batch_size))
                    if "seq_offsets" in locals()
                    else None
                )
            ),
        )

        # if current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):
        #     workspace_manager = current_workspace_manager()
        #     (topk_workspace,) = workspace_manager.get_simultaneous(
        #         ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        #     )
        #     torch.ops._C.persistent_topk(
        #         logits,
        #         seq_lens,
        #         topk_indices,
        #         topk_workspace,
        #         topk_tokens,
        #         attn_metadata_narrowed.max_seq_len,
        #     )
        # else:
        #     torch.ops._C.top_k_per_row_decode(
        #         logits,
        #         next_n,
        #         seq_lens,
        #         topk_indices,
        #         num_rows,
        #         logits.stride(0),
        #         logits.stride(1),
        #         topk_tokens,
        #     )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = _unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def deepseek_v4_metax_indexer_fake(
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
    op_name="deepseek_v4_metax_indexer",
    op_func=deepseek_v4_metax_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=deepseek_v4_metax_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


class SparseAttnIndexerFL(SparseAttnIndexer):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    # def __init__(
    #     self,
    #     k_cache,
    #     quant_block_size: int,
    #     scale_fmt: str,
    #     topk_tokens: int,
    #     head_dim: int,
    #     max_model_len: int,
    #     max_total_seq_len: int,
    #     topk_indices_buffer: torch.Tensor,
    #     skip_k_cache_insert: bool = False,
    #     use_fp4_cache: bool = False,
    # ):
    #     super().__init__()
    #     self.k_cache = k_cache
    #     self.quant_block_size = quant_block_size
    #     self.scale_fmt = scale_fmt
    #     self.topk_tokens = topk_tokens
    #     self.head_dim = head_dim
    #     self.max_model_len = max_model_len
    #     self.max_total_seq_len = max_total_seq_len
    #     self.topk_indices_buffer = topk_indices_buffer
    #     self.skip_k_cache_insert = skip_k_cache_insert
    #     self.use_fp4_cache = use_fp4_cache
    #     if current_platform.is_cuda() and not has_deep_gemm():
    #         raise RuntimeError(
    #             "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
    #         )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_oot(hidden_states, q_quant, k, weights)

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.deepseek_v4_metax_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
        )
