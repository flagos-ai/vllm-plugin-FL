# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU evas DSV4 attention: subclass of the ``DeepseekV4Attention`` ABC.

The ABC's own sequential ``attention_impl`` (input GEMMs + KV insert +
``forward_mqa`` + ``_o_proj``) stays upstream; only the two abstract methods it
calls are implemented here against the flaggems ``_evas`` pure-PyTorch ops,
near-verbatim with identical signatures: sparse MLA for ``forward_mqa`` and its
decode/prefill halves, inverse-RoPE + FP8 quant + grouped einsum for
``_o_proj``.  Only ``backend_cls`` is fixed; the cache spec and backend
selection are inherited.

"""

from __future__ import annotations

import os
import sys
from typing import cast

import torch
import torch.nn.functional as F

import flag_gems

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm_fl.dispatch.backends.vendor.evas.impl.attention.dual_die_view import (
    repair_view,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_fl.dispatch.backends.vendor.evas.impl.fp8_methods import (
    wo_a_grouped_views,
)


_qnorm_rope_dummy_seen: list[bool] = []


def _qnorm_rope_dummy_batch(slot_mapping: torch.Tensor) -> bool:
    """True when ``slot_mapping`` carries no cache slot at all.

    ``-1`` means "no cache slot".  The vendored kernel indexes the paged cache
    straight from the raw slot (``block_idx = slot // block_size``,
    ``pos_in_block = slot % block_size``), so ``-1`` lands on block 0 row 63 of
    a 64-slot block (``-1 // 64 == 0``, ``-1 % 64 == 63``) -- a legal address
    that can belong to a live sequence, i.e. silent KV pollution, and the
    vendored bytes are hash-pinned.  So a negative slot never reaches it.

    An all-``-1`` mapping is vLLM's profiling dummy batch: ``_dummy_run``
    fills the whole mapping while ``query_start_loc`` still reports N tokens,
    and it discards the outputs, so the caller skips the whole op.  A partly
    negative mapping would be a live batch with padding, which the runner only
    produces for data/sequence parallelism padded batches; nothing on the
    current path does, so it raises instead of guessing a policy.

    Costs one device-to-host read of a single boolean in the common all-valid
    case, the same order of host round trip ``_write_paged_runs`` pays.  It is
    deliberately not latched after the first all-valid batch: the point of the
    test is to stay loud if a runner ever starts padding live batches.
    """
    if not bool((slot_mapping < 0).any()):
        return False
    if bool((slot_mapping >= 0).any()):
        raise RuntimeError(
            "EvasDSV4Attention: slot_mapping mixes -1 padding with real "
            "slots, and the vendored qnorm_rope_kv_insert kernel writes a -1 "
            "slot to cache block 0 row 63. The current runner only pads whole "
            "dummy batches (DP=1, sequence parallelism off); re-open D5 before "
            "enabling a runner that pads live batches."
        )
    if not _qnorm_rope_dummy_seen:
        _qnorm_rope_dummy_seen.append(True)
        print(
            "[evas-callsite] fused_deepseek_v4_qnorm_rope_kv_rope_quant"
            "_insert profiling dummy batch (every slot -1) -> skip",
            file=sys.stderr,
            flush=True,
        )
    return True


def _c4a_decode_topk_args(
    topk_indices_buffer: torch.Tensor,
    swa_metadata: DeepseekSparseSWAMetadata,
    attn_metadata: DeepseekV4FlashMLAMetadata,
    num_decode_tokens: int,
    num_decodes: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Positional args for ``compute_global_topk_indices_and_lens`` (C4A decode).

    The sparse metadata buffers are batch-sized: ``token_to_req_indices`` and
    ``is_valid_token`` carry one entry per token of the whole batch (decode rows
    first), and ``block_table`` one row per request.  The upstream nvidia call
    sites pass ``token_to_req_indices`` un-sliced because the upstream Triton
    kernel only indexes the first ``num_decode_tokens`` rows of the grid.  The
    evas wrapper instead validates one-entry-per-token and raises otherwise, so
    slice every token-indexed input to the decode window here; the decode rows
    are the first ``num_decode_tokens`` of the batch, so the values are
    identical to what the upstream kernel reads.

    Every row slice below is repaired before the call: a slice keeps the parent
    storage descriptor, so the kernel's ``to_split_dim`` is a no-op on it and
    its ``unzip`` then parts the *parent* buffer -- die1 would read the second
    half of the whole batch instead of the second half of the window, and the
    ``topk_indices`` grid/stores are sized from that same descriptor.
    ``block_table`` is passed unrepaired on purpose: the kernel replicates it
    per die (``replicate_to_dies``) instead of unzipping it.
    """
    return (
        repair_view(topk_indices_buffer[:num_decode_tokens]),
        repair_view(swa_metadata.token_to_req_indices[:num_decode_tokens]),
        attn_metadata.block_table[:num_decodes],
        block_size,
        repair_view(swa_metadata.is_valid_token[:num_decode_tokens]),
    )


class EvasDSV4Attention(DeepseekV4Attention):
    """LPU evas DSV4 sparse-MLA attention (base ABC subclass)."""

    backend_cls = DeepseekV4FlashMLABackend

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # Match the flashmla padding (fp8 decode kernel constraint).  The evas
        # ops handle any head count; padding keeps the out buffer + kv_insert
        # zero-fill contract identical to the base.
        if num_heads > 128:
            raise ValueError(
                f"EvasDSV4Attention does not support {num_heads} heads "
                "(FP8 decode kernel requires h_q in {64, 128})."
            )
        return 64 if num_heads <= 64 else 128

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, object] | list[dict[str, object]] | None
        ),
    ) -> torch.Tensor:
        # The base 0.24 method calls torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
        # which has no impl on LPU (vllm._C is not compiled).  Route to the
        # flaggems evas op (8-arg, no padded_heads) and pad q to padded_heads
        # afterwards (the CUDA kernel zero-fills the padding head slots
        # internally; the evas op leaves that to the caller).
        if not isinstance(attn_metadata, dict):
            if self.n_local_heads < self.padded_heads:
                return F.pad(
                    q,
                    (0, 0, 0, self.padded_heads - self.n_local_heads),
                    value=0.0,
                )
            return q

        from vllm.v1.attention.backends.mla.sparse_swa import (
            DeepseekSparseSWAMetadata,
        )

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache
        assert positions.dtype == torch.int64
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        cache_dtype = swa_kv_cache.dtype
        block_size = swa_metadata.block_size

        if cache_dtype == torch.uint8:
            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
            if not _qnorm_rope_dummy_batch(swa_metadata.slot_mapping):
                flag_gems.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                    q,
                    kv,
                    swa_kv_cache_2d,
                    swa_metadata.slot_mapping,
                    positions,
                    cos_sin_cache,
                    self.eps,
                    block_size,
                )
        elif cache_dtype == torch.bfloat16:
            # 3-D cache layout, the shape the base class contract passes for
            # this dtype.  LPU inference itself runs FP8, so this branch is
            # kept for contract completeness, not exercised on the main path.
            swa_kv_cache_3d = swa_kv_cache.view(-1, block_size, self.head_dim)
            flag_gems.fused_deepseek_v4_qnorm_rope_kv_rope_insert(
                q,
                kv,
                swa_kv_cache_3d,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                self.eps,
                block_size,
            )
        else:
            raise NotImplementedError(
                f"EvasDSV4Attention: unsupported SWA cache dtype {cache_dtype}"
            )

        if self.n_local_heads < self.padded_heads:
            q = F.pad(
                q,
                (0, 0, 0, self.padded_heads - self.n_local_heads),
                value=0.0,
            )
        return q

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # Warmup dummy run: no real metadata.
            output.zero_()
            return
        if not isinstance(attn_metadata, dict):
            output.zero_()
            return
        swa_metadata = cast(
            DeepseekSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.prefix),
        )

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: DeepseekSparseSWAMetadata,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        if os.environ.get("VLLM_FL_DSV4_DEBUG_ARGS"):
            print(
                f"[attn-probe] decode enter layer={getattr(self, 'layer_name', '?')} "
                f"swa_only={swa_only}", flush=True,
            )
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = flag_gems.compute_global_topk_indices_and_lens(
                    *_c4a_decode_topk_args(
                        self.topk_indices_buffer,
                        swa_metadata,
                        attn_metadata,
                        num_decode_tokens,
                        num_decodes,
                        block_size,
                    )
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # Queries in the same seq are treated as different queries; attention
        # only attends by generated indices.  q arrives padded to padded_heads.
        q = q.unsqueeze(1)

        # SWA cache (num_blocks, swa_block_size, 1, head_bytes).
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        # One FlashMLASchedMeta per layer type (shared across same-type layers
        # in this decode step); built by the sparse metadata builder.
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None

        out, _ = flag_gems.flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=self.head_dim,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=output.unsqueeze(1),
        )
        if os.environ.get("VLLM_FL_DSV4_DEBUG_ARGS"):
            print(
                f"[attn-probe] decode exit layer={getattr(self, 'layer_name', '?')}",
                flush=True,
            )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: DeepseekSparseSWAMetadata,
    ) -> None:
        swa_only = attn_metadata is None
        if os.environ.get("VLLM_FL_DSV4_DEBUG_ARGS"):
            print(
                f"[attn-probe] prefill enter layer={getattr(self, 'layer_name', '?')} "
                f"swa_only={swa_only}", flush=True,
            )

        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0

        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
        )
        assert chunk_plan
        workspace_manager = current_workspace_manager()
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
            chunk_size = chunk_end - chunk_start
            kv = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_M, q.shape[-1]), torch.bfloat16),
            )[0]
            if not swa_only:
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                flag_gems.dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            swa_block_table = swa_metadata.block_table[num_decodes:]
            flag_gems.dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=chunk_N,
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = flag_gems.combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_M,
                chunk_N,
            )
            # The kernel only reads q, but the slice keeps q descriptor and the
            # kernel would then part q at the parent width.  Repair it here.
            out, _, _ = flag_gems.flash_mla_sparse_fwd(
                q=repair_view(q[query_start:query_end]),
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
            )
            output[query_start:query_end] = out
        if os.environ.get("VLLM_FL_DSV4_DEBUG_ARGS"):
            print(
                f"[attn-probe] prefill exit layer={getattr(self, 'layer_name', '?')}",
                flush=True,
            )

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # Inverse RoPE + FP8 group quant.  The dtype is named here because the
        # kernel module probes the device capability for a default and then
        # asserts on e4m3fn.  The container is settled to native float8_e4m3fn
        # because the einsum below converts a uint8 x to fp8 through the host:
        # it does x.cpu().view(float8_e4m3fn).to(device), a full activation
        # round trip that .view(dtype) makes unnecessary on device.  The kernel
        # returns float8_e4m3fn already and the pure twin returns the LPU fp8
        # container (uint8), so this is one byte-preserving view on the twin
        # and a no-op on the kernel.
        o_fp8, o_scale = flag_gems.fused_inv_rope_fp8_quant(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            dtype=torch.float8_e4m3fn,
            tma_aligned_scales=False,
        )
        if o_fp8.dtype == torch.uint8:
            o_fp8 = o_fp8.view(torch.float8_e4m3fn)
        # wo_a.weight is stored [out, in] = [n_groups*o_lora_rank, in_width]
        # (see Fp8LinearMethodLPU.create_weights and the checkpoint); the
        # grouped einsum needs [h, d, r] = [n_local_groups, o_lora_rank,
        # heads_per_group*head_dim] with r the contraction dim -- a plain
        # row-major reshape, no transpose.  See wo_a_grouped_views.
        h = self.n_local_groups
        d = self.o_lora_rank
        r = (self.n_local_heads // h) * self.head_dim
        wo_a = self.wo_a
        bn, bk = getattr(wo_a, "weight_block_size", [128, 128])
        y, ys = wo_a_grouped_views(
            wo_a.weight, wo_a.weight_scale_inv, h, d, r, bn, bk
        )
        z = flag_gems.fp8_einsum(
            "bhr,hdr->bhd", o_fp8, o_scale, y, ys, block_size=[bn, bk],
        )
        return self.wo_b(z.flatten(1))
