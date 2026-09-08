# Copyright (c) 2026 BAAI. All rights reserved.

"""GLM-5.2 Hygon-specific sparse attention Indexer.

This module is intentionally isolated from ``vllm_fl.ops.sparse_attn_indexer``.
It is injected only for ``glm_moe_dsa`` physical Indexer layers on Hygon by
``vllm_fl.patches.glm_index_share``.

The implementation supports the vLLM V3.2 FP8 Indexer path only:

* Q is FP8; its per-token scale has already been folded into ``weights``.
* K is inserted into the existing split-layout FP8 Indexer cache.
* Prefill MQA logits reuse FlagGems ``fp8_mqa_logits`` through FL dispatch.
* Decode paged MQA logits use a Hygon-specific split-layout kernel.

It deliberately does not support the DeepSeek-V4 MXFP4 Indexer path.
"""

from __future__ import annotations

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
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


# Existing FL operators that have already been adapted.
_indexer_k_quant_and_cache = CachedOp("indexer_k_quant_and_cache")
_cp_gather_indexer_k_quant_cache = CachedOp("cp_gather_indexer_k_quant_cache")
_glm_hygon_top_k_per_row_prefill = CachedOp("glm_top_k_per_row_prefill")
_glm_hygon_top_k_per_row_decode = CachedOp("glm_top_k_per_row_decode")
_pack_seq_triton = CachedOp("pack_seq_triton")
_unpack_seq_triton = CachedOp("unpack_seq_triton")

# GLM-5.2 + Hygon-only MQA logits operators.
_glm_hygon_indexer_fp8_mqa_logits = CachedOp("glm_indexer_fp8_mqa_logits")
_glm_hygon_indexer_fp8_paged_mqa_logits = CachedOp("glm_indexer_fp8_paged_mqa_logits")


RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


def _assert_hygon() -> None:
    """Fail if this GLM-specific implementation is used on another vendor."""

    vendor_name = getattr(
        current_platform,
        "vendor_name",
        None,
    )

    if vendor_name != "hygon":
        raise RuntimeError(
            "GlmHygonSparseAttnIndexer can only run on Hygon, "
            f"but current vendor is {vendor_name!r}."
        )


def _fp8_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
) -> tuple[
    tuple[tuple[int, int], torch.dtype],
    tuple[tuple[int, int], torch.dtype],
]:
    """Workspace for gathered FP8 K values and one FP32 scale per token."""

    return (
        (
            (total_seq_lens, head_dim),
            fp8_dtype,
        ),
        (
            (total_seq_lens, 4),
            torch.uint8,
        ),
    )


def glm_hygon_sparse_attn_indexer_fl(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
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
) -> torch.Tensor:
    """Run the GLM-5.2 V3.2 FP8 sparse Indexer on Hygon."""

    _assert_hygon()

    # GLM-5.2 V3.2 currently has:
    #
    #   head_dim = 128
    #   quant_block_size = 128
    #
    # so every K vector owns exactly one FP32 scale.
    if quant_block_size != head_dim:
        raise NotImplementedError(
            "GLM Hygon FP8 Indexer currently requires one quantization "
            "block per K vector: quant_block_size == head_dim."
        )

    fp8_dtype = current_platform.fp8_dtype()

    if q_quant.dtype != fp8_dtype:
        raise TypeError(
            "GLM Hygon Indexer expects q_quant to use "
            "current_platform.fp8_dtype(), "
            f"but got {q_quant.dtype}."
        )

    # During dummy/profile run attn_metadata is not a per-layer dict.
    attn_metadata = get_forward_context().attn_metadata

    k_cache_prefix = _resolve_layer_name(
        k_cache_prefix
    )

    if not isinstance(attn_metadata, dict):
        values_spec, scales_spec = (
            _fp8_workspace_shapes(
                total_seq_lens,
                head_dim,
                fp8_dtype,
            )
        )

        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            (
                (
                    RADIX_TOPK_WORKSPACE_SIZE,
                ),
                torch.uint8,
            ),
        )

        # Preserve vLLM sparse-indexer memory profiling behavior.
        max_logits_elems = (
            envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
            * 1024
            * 1024
        )

        _ = torch.empty(
            max_logits_elems,
            dtype=torch.uint8,
            device=hidden_states.device,
        )

        return glm_hygon_sparse_attn_indexer_fl_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
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
        )

    attn_metadata_narrowed = (
        attn_metadata[k_cache_prefix]
    )

    assert isinstance(
        attn_metadata_narrowed,
        DeepseekV32IndexerMetadata,
    )

    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # During speculative decoding K can be padded to the graph batch
    # size while slot_mapping only covers actual tokens.
    num_tokens = slot_mapping.shape[0]

    if k is not None:
        k = k[:num_tokens]

    # ------------------------------------------------------------
    # Insert current K into the Indexer FP8 cache.
    # ------------------------------------------------------------
    if not skip_k_cache_insert:
        if scale_fmt is None:
            raise RuntimeError(
                "GLM Hygon FP8 Indexer requires "
                "a non-null scale_fmt."
            )

        _indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    topk_indices_buffer[
        : hidden_states.shape[0]
    ] = -1

    # ============================================================
    # Prefill
    # ============================================================
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill

        assert prefill_metadata is not None

        workspace_manager = current_workspace_manager()

        values_spec, scales_spec = (
            _fp8_workspace_shapes(
                total_seq_lens,
                head_dim,
                fp8_dtype,
            )
        )

        (
            k_quant_full,
            k_scale_full,
        ) = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )

        for chunk in prefill_metadata.chunks:
            k_quant = k_quant_full[
                : chunk.total_seq_lens
            ]

            k_scale = k_scale_full[
                : chunk.total_seq_lens
            ]

            if not chunk.skip_kv_gather:
                _cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            q_slice = q_quant[
                chunk.token_start:
                chunk.token_end
            ]

            # Gather workspace stores the FP32 scale as four raw
            # bytes per token.
            k_scale_fp32 = (
                k_scale.view(torch.float32)
                .squeeze(-1)
            )

            # GLM-Hygon prefill:
            #
            #   FlagGems fp8_mqa_logits(
            #       q_fp8,
            #       (k_fp8, k_scale),
            #       ...
            #   )
            #
            # q_scale is not passed because the Indexer already folds
            # q_scale into weights during fused_indexer_q_rope_quant.
            logits = (
                _glm_hygon_indexer_fp8_mqa_logits(
                    q_slice,
                    (
                        k_quant,
                        k_scale_fp32,
                    ),
                    weights[
                        chunk.token_start:
                        chunk.token_end
                    ],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    False,
                )
            )

            num_rows = logits.shape[0]

            topk_indices = (
                topk_indices_buffer[
                    chunk.token_start:
                    chunk.token_end,
                    :topk_tokens,
                ]
            )

            _glm_hygon_top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    # ============================================================
    # Decode
    # ============================================================
    if has_decode:
        decode_metadata = (
            attn_metadata_narrowed.decode
        )

        assert decode_metadata is not None

        decode_lens = (
            decode_metadata.decode_lens
        )

        # Preserve the existing V3.2 padding behavior.
        if decode_metadata.requires_padding:
            padded_q_quant_decode_tokens = (
                _pack_seq_triton(
                    q_quant[:num_decode_tokens],
                    decode_lens,
                )
            )

        else:
            padded_q_quant_decode_tokens = (
                q_quant[:num_decode_tokens]
                .reshape(
                    decode_lens.shape[0],
                    -1,
                    *q_quant.shape[1:],
                )
            )

        batch_size = (
            padded_q_quant_decode_tokens.shape[0]
        )

        next_n = (
            padded_q_quant_decode_tokens.shape[1]
        )

        num_padded_tokens = (
            batch_size * next_n
        )

        seq_lens = (
            decode_metadata.seq_lens[
                :batch_size
            ]
        )

        # IMPORTANT:
        #
        # Pass the original raw V3.2 Indexer cache here.
        #
        # Do NOT call kv_cache_as_quant_view(), because that helper is
        # for the DeepGEMM view.  The Hygon decode kernel reads the
        # split physical layout directly:
        #
        # [all K values][all FP32 scales].
        logits = (
            _glm_hygon_indexer_fp8_paged_mqa_logits(
                padded_q_quant_decode_tokens,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                max_model_len,
                head_dim,
                quant_block_size,
            )
        )

        num_rows = logits.shape[0]

        topk_indices = (
            topk_indices_buffer[
                :num_padded_tokens,
                :topk_tokens,
            ]
        )

        _glm_hygon_top_k_per_row_decode(
            logits,
            next_n,
            seq_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
            attn_metadata_narrowed.max_seq_len,
        )

        if decode_metadata.requires_padding:
            topk_indices = (
                _unpack_seq_triton(
                    topk_indices.reshape(
                        batch_size,
                        -1,
                        topk_indices.shape[-1],
                    ),
                    decode_lens,
                )
            )

            topk_indices_buffer[
                : topk_indices.shape[0],
                : topk_indices.shape[-1],
            ] = topk_indices

    return topk_indices_buffer


def glm_hygon_sparse_attn_indexer_fl_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
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
) -> torch.Tensor:
    """Fake implementation used by torch.compile/profile."""

    return topk_indices_buffer


direct_register_custom_op(
    op_name=(
        "glm_hygon_sparse_attn_indexer_fl"
    ),
    op_func=glm_hygon_sparse_attn_indexer_fl,
    mutates_args=[
        "topk_indices_buffer",
    ],
    fake_impl=(
        glm_hygon_sparse_attn_indexer_fl_fake
    ),
    dispatch_key=current_platform.dispatch_key,
)


class GlmHygonSparseAttnIndexer(
    SparseAttnIndexer
):
    """Sparse Indexer used only by GLM-5.2 on Hygon."""

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: (
            torch.Tensor
            | tuple[
                torch.Tensor,
                torch.Tensor,
            ]
        ),
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_oot(
            hidden_states,
            q_quant,
            k,
            weights,
        )

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_quant: (
            torch.Tensor
            | tuple[
                torch.Tensor,
                torch.Tensor,
            ]
        ),
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        _assert_hygon()

        # This class is deliberately not used for DeepSeek-V4 MXFP4.
        if self.use_fp4_cache:
            raise NotImplementedError(
                "GlmHygonSparseAttnIndexer "
                "supports the V3.2 FP8 Indexer "
                "cache only; MXFP4 is intentionally "
                "not routed here."
            )

        if isinstance(q_quant, tuple):
            raise TypeError(
                "GlmHygonSparseAttnIndexer expects "
                "a single FP8 q_quant tensor."
            )

        return (
            torch.ops.vllm
            .glm_hygon_sparse_attn_indexer_fl(
                hidden_states,
                _encode_layer_name(
                    self.k_cache.prefix
                ),
                self.k_cache.kv_cache,
                q_quant,
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
            )
        )
