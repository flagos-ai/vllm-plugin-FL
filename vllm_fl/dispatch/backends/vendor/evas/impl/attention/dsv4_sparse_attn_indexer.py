# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU custom-op implementation: Sparse Attention Indexer (DSV4).

``EvasDSV4SparseAttnIndexer`` plus its ``evas_dsv4_sparse_attn_indexer``
operator, with the LPU scoring branches inlined — a custom-op layer and
operator implementation rather than an ``apply_lpu_*_patch()``.  Registered
lazily on LPU only, through the model_compat indexer-dispatch gate; the
non-LPU else branches are fork fidelity.  Which kernel scores never leaks into
this caller: scoring lives in the flag_gems ``_evas/fused`` overrides.

"""

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

from vllm.model_executor.layers.sparse_attn_indexer import (
    SparseAttnIndexer,
    _gather_workspace_shapes,
    kv_cache_as_quant_view,
    sparse_attn_indexer_fake,
)
from vllm_fl.dispatch import CachedOp
from vllm_fl.dispatch.backends.vendor.evas.impl.attention.dual_die_view import (
    repair_view,
)
from vllm_fl.utils import DeviceInfo

_indexer_k_quant_and_cache = CachedOp("indexer_k_quant_and_cache")
_cp_gather_indexer_k_quant_cache = CachedOp("cp_gather_indexer_k_quant_cache")
_top_k_per_row_prefill = CachedOp("top_k_per_row_prefill")
_pack_seq_triton = CachedOp("pack_seq_triton")
_top_k_per_row_decode = CachedOp("top_k_per_row_decode")
_unpack_seq_triton = CachedOp("unpack_seq_triton")

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


# ============================================================================
# LPU pure-PyTorch FP8 sparse-indexer scoring
# ============================================================================
# The LPU scoring path lives in the flag_gems ``_evas/fused`` overrides
# (``cp_gather_indexer_k_quant_cache`` / ``fp8_fp4_mqa_logits`` /
# ``fp8_fp4_paged_mqa_logits``), so the kernel behind each one can change
# without touching this caller, and the 132B SEGREGATED indexer cache layout is
# documented there.  The LPU branches below call those ops directly (deep_gemm
# is never called on LPU — its impl is None -> ``_missing()`` raises).


def evas_dsv4_sparse_attn_indexer(
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

    if not isinstance(attn_metadata, dict):
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

        return sparse_attn_indexer_fake(
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
    # Resolve once: both the prefill and decode branches take the LPU
    # pure-PyTorch scoring path, and a decode-only step never reaches the
    # (prefill-scoped) assignment below.
    is_lpu = DeviceInfo().device_type == "lpu"

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
        # MXFP4 (head_dim/2 bytes packed + head_dim//32 ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        # LPU has no deep_gemm impl (has_deep_gemm() is True but the vendored
        # package's ops are stubs -> calling raises).  The pure-PyTorch FP8
        # scoring path reads the 132B padded indexer cache directly and
        # computes the same logits in PyTorch, with the same interface as the
        # upstream DeepGEMM op.
        for chunk in prefill_metadata.chunks:
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather and not is_lpu:
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

            if is_lpu and not use_fp4_cache:
                # LPU pure-PyTorch FP8 prefill scoring via the flag_gems
                # ``_evas/fused`` overrides (cp_gather_indexer_k_quant_cache +
                # fp8_fp4_mqa_logits), same interface as the upstream CUDA ops.
                # The compressor already wrote every compressed K row to the
                # 132B padded cache, so gather the GLOBAL K once (one flat slot
                # per compressed token row) into a local buffer (CPU metadata
                # math + flat gather, both LPU-safe).  The [ks, ke) window is
                # applied downstream by top_k_per_row_prefill: the twin masks
                # the logits itself and the adopted kernel ignores ks/ke, so
                # masking here is not something the caller may assume.
                from flag_gems import (
                    cp_gather_indexer_k_quant_cache,
                    fp8_fp4_mqa_logits,
                )
                n_chunk = q_slice.shape[0]
                total_rows = int(chunk.cu_seq_lens[-1].item())
                k_fp8 = torch.empty(
                    (total_rows, head_dim),
                    dtype=torch.uint8,
                    device=kv_cache.device,
                )
                k_scale = torch.empty(
                    (total_rows,), dtype=torch.float32, device=kv_cache.device
                )
                cp_gather_indexer_k_quant_cache(
                    kv_cache, k_fp8, k_scale, chunk.block_table, chunk.cu_seq_lens
                )
                # Hand Q over as native fp8.  The adopted kernel declares Q in
                # float8_e4m3fn and raises on the uint8 container our quantizer
                # returns (K is different: it takes raw uint8 and bitcasts in
                # the kernel), so an unviewed uint8 Q made the wrapper latch to
                # the pure twin on every rank.  The bytes are already E4M3, so
                # the bridge is one zero-copy view at the caller: the container
                # is a call-site convention, exactly like the fp8 view _o_proj
                # applies in dsv4_attention.py, and the twin decodes either
                # container, so the kernel and pure arms see identical bytes.
                q_values = q_slice.reshape(n_chunk, -1, head_dim)
                if q_values.dtype == torch.uint8:
                    q_values = q_values.view(torch.float8_e4m3fn)
                logits = fp8_fp4_mqa_logits(
                    (q_values, q_scale_slice),
                    (k_fp8, k_scale),
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

            topk_view = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            # The slice keeps the buffer descriptor, so the kernel would part
            # the parent and store its rows there.  Hand it a faithful tensor
            # and own the write-back: the kernel only writes back when it did
            # the copy itself, and here the copy is ours.
            topk_indices = repair_view(topk_view)

            _top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
            if topk_indices is not topk_view:
                topk_view.copy_(topk_indices)

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
        # Per-token head weights follow q's layout: with padding they are
        # per-request contiguous and must be expanded to the same [B, next_n]
        # grid, otherwise index ``t`` of the padded logits kernel lands on
        # another request's token (or past the end of a decode-only batch).
        # The kernel indexes rows (``weights[t]``) on a flat [B*next_n, H]
        # tensor, so flatten the packed [B, next_n, H] back to 2D and give it
        # the same row-split storage the unpadded slice has (pack allocates
        # column-split for its consumers).  It is a few dozen KiB, so the copy
        # is free.  Padding with 0 keeps stale rows finite; they are dropped on
        # unpack anyway.
        padded_weights = (
            _pack_seq_triton(weights[:num_decode_tokens], decode_lens, pad_value=0)
            .reshape(num_padded_tokens, -1)
            .contiguous()
            if decode_metadata.requires_padding
            else weights[:num_padded_tokens]
        )
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )

        if is_lpu and not use_fp4_cache:
            # LPU pure-PyTorch paged FP8 scoring via the flag_gems
            # ``fp8_fp4_paged_mqa_logits`` override, same interface as the
            # upstream DeepGEMM op it replaces.
            # ``kv_cache`` was wrapped in ``kv_cache_as_quant_view`` above
            # (unsqueeze(-2) -> 4D for the deep_gemm FP8 path); squeeze it back
            # to the 3D ``[num_blocks, block_size, 132]`` view the op wants.
            from flag_gems import fp8_fp4_paged_mqa_logits
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_decode_tokens, padded_q_scale),
                kv_cache.squeeze(-2),
                padded_weights,
                seq_lens,
                decode_metadata.block_table,
                None,  # schedule_metadata (LPU ignores; DeepGEMM artifact)
                max_model_len=max_model_len,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                padded_weights,
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        _top_k_per_row_decode(
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
            topk_indices = _unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


direct_register_custom_op(
    op_name="evas_dsv4_sparse_attn_indexer",
    op_func=evas_dsv4_sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


class EvasDSV4SparseAttnIndexer(SparseAttnIndexer):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.

    On LPU this subclass is bound in place of the upstream class and registers its own
    op, `evas_dsv4_sparse_attn_indexer`, so no `CompilationConfig` entry is needed.
    """

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
        return torch.ops.vllm.evas_dsv4_sparse_attn_indexer(
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
