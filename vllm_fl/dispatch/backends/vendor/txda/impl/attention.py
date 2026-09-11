# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda (tsingmicro) SDPA reference attention backend.

The flag_gems flash_attn_varlen_func kernel computes silently wrong values on
TX8110 (probe: maxrel=inf/nan/37.7 for causal/noncausal varlen cases), so the
flag_gems AttentionFLBackend cannot be used. This backend reuses the flag_gems
metadata machinery (KV layout, block table, slot mapping) but computes attention
with torch SDPA, which is numerically correct on txda. Compiler-independent:
works under both the flagtree and triton compilers.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from vllm.v1.attention.backend import AttentionType
from vllm_fl.dispatch.backends.flaggems.impl.attention import (
    AttentionFLBackend,
    AttentionFLImpl,
)

_DEBUG = os.environ.get("FL_DEBUG_TXDA_ATTN") == "1"
_PRINTED = [0]


class TxdaSDPAAttentionBackend(AttentionFLBackend):
    """Attention backend for tsingmicro TX devices using torch SDPA."""

    # AttentionFLBackend inherits forward_includes_kv_cache_update=True from
    # vllm's AttentionBackend default, but its do_kv_cache_update is designed
    # to be called separately by vLLM's unified_kv_cache_update custom op --
    # which Attention.forward only invokes when this flag is False. With the
    # True default the KV cache is never written and forward reads zeros
    # (silent garbage). Every vLLM backend overrides it to False; so do we.
    forward_includes_kv_cache_update: bool = False

    # get_name is inherited: vLLM requires the name to be a member of
    # AttentionBackendEnum ("CUSTOM"), which AttentionFLBackend already returns.

    @staticmethod
    def get_impl_cls() -> type["TxdaSDPAAttentionImpl"]:
        return TxdaSDPAAttentionImpl


class TxdaSDPAAttentionImpl(AttentionFLImpl):
    """
    SDPA-based attention impl for TX8110.

    do_kv_cache_update and forward are overridden to avoid the flag_gems
    kernels (reshape_and_cache_flash / flash_attn_varlen_func) that compute
    silently wrong values on TX8110. The KV cache layout
    (2, num_blocks, block_size, num_kv_heads, head_size) and metadata are
    unchanged, so AttentionFLMetadataBuilder is reused as-is.
    """

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """Write key/value into the paged KV cache via plain indexing.

        Avoids flag_gems reshape_and_cache_flash (wrong on TX8110). Indexing
        by (block_id, offset) is layout-independent.
        """
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        key_cache, value_cache = kv_cache.unbind(0)
        block_size = key_cache.shape[1]
        block_ids = slot_mapping // block_size
        offsets = slot_mapping % block_size
        # Padded slots carry slot_mapping == -1; without this guard they index
        # the last block (floor division: -1 // block_size == -1) and corrupt
        # the cache with padded garbage.
        valid = slot_mapping >= 0
        key_cache[block_ids[valid], offsets[valid]] = key[valid]
        value_cache[block_ids[valid], offsets[valid]] = value[valid]

        if _DEBUG and _PRINTED[0] < 400:
            _PRINTED[0] += 1
            print(
                f"[txda-debug] kv_update#{_PRINTED[0]} n={key.shape[0]} "
                f"k0={key[0].reshape(-1)[:4].tolist()} "
                f"v0={value[0].reshape(-1)[:4].tolist()} "
                f"slot0={slot_mapping[0].item()} slotN={slot_mapping[-1].item()} "
                f"bids0={block_ids[:4].tolist()} offs0={offsets[:4].tolist()} "
                f"block_size={block_size}",
                flush=True,
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward with per-request torch SDPA on the paged KV cache."""
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported for TxdaSDPAAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        if self.alibi_slopes is not None:
            raise NotImplementedError("alibi not supported on TXDA_SDPA")
        if self.logits_soft_cap:
            raise NotImplementedError("logits soft cap not supported on TXDA_SDPA")
        if attn_metadata.use_cascade:
            raise NotImplementedError("cascade attention not supported on TXDA_SDPA")

        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        output = output[:num_actual_tokens]

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return self._forward_encoder(query, key, value, output, attn_metadata)

        key_cache, value_cache = kv_cache.unbind(0)
        cu_seqlens_q = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table

        num_reqs = cu_seqlens_q.shape[0] - 1
        window_left = self.sliding_window[0]  # -1 means no sliding window
        for i in range(num_reqs):
            qs, qe = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
            q_len = qe - qs
            seq_len = seq_lens[i].item()
            if q_len == 0 or seq_len == 0:
                output[qs:qe] = 0
                continue

            # Gather the request's KV from the paged cache. Advanced indexing
            # (key_cache[blocks]) has no PrivateUse1 kernel on txda, so it falls
            # back to a CPU copy of the whole cache and hangs at engine scale.
            # Gather via per-block slices + cat instead (probe-verified).
            blocks = block_table[i].tolist()
            k = torch.cat([key_cache[b] for b in blocks], dim=0).reshape(
                -1, self.num_kv_heads, self.head_size
            )[:seq_len]
            v = torch.cat([value_cache[b] for b in blocks], dim=0).reshape(
                -1, self.num_kv_heads, self.head_size
            )[:seq_len]
            q = query[qs:qe]

            out_i = self._sdpa(q, k, v, seq_len, window_left)
            # output is [num_tokens, num_heads, head_size]; out_i matches directly.
            output[qs:qe] = out_i

            if _DEBUG and i == 0 and _PRINTED[0] < 400:
                _PRINTED[0] += 1
                print(
                    f"[txda-debug] fwd#{_PRINTED[0]} layer={getattr(layer, 'name', '?')} "
                    f"n={num_actual_tokens} cu_q={cu_seqlens_q.tolist()} "
                    f"seq_lens={seq_lens.tolist()} reqs={num_reqs} "
                    f"bt0={blocks[:4]} seq_len={seq_len} q_len={q_len} "
                    f"k_rb0={k[0].reshape(-1)[:4].tolist()} "
                    f"q0={q[0].reshape(-1)[:4].tolist()} "
                    f"out0={out_i[0].reshape(-1)[:4].tolist()}",
                    flush=True,
                )

        return output

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int,
        window_left: int,
    ) -> torch.Tensor:
        """SDPA for one request.

        q: [q_len, num_heads, head_size]; k/v: [seq_len, num_kv_heads, head_size].
        Returns [q_len, num_heads, head_size].
        """
        q_len = q.shape[0]
        # Query tokens are the last q_len of the request's sequence.
        q_start = seq_len - q_len

        q = q.permute(1, 0, 2).unsqueeze(0)  # [1, H, q_len, D]
        kk = k.permute(1, 0, 2).unsqueeze(0)  # [1, kv_h, seq, D]
        vv = v.permute(1, 0, 2).unsqueeze(0)
        if self.num_queries_per_kv > 1:
            kk = kk.repeat_interleave(self.num_queries_per_kv, dim=1)
            vv = vv.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Mask: key position j is visible to query row i iff
        # j <= q_start + i, plus the sliding-window left bound.
        causal = False
        attn_mask = None
        if q_len == 1:
            # Decode: the single new token attends all keys; no mask needed.
            pass
        elif q_start == 0 and window_left < 0:
            causal = True  # Full prefill: plain causal.
        else:
            rows = torch.arange(q_len, device=q.device).unsqueeze(1)
            cols = torch.arange(seq_len, device=q.device).unsqueeze(0)
            visible = cols <= (q_start + rows)
            if window_left >= 0:
                visible &= cols >= (q_start + rows - window_left)
            attn_mask = visible

        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            kk,
            vv,
            attn_mask=attn_mask,
            is_causal=causal,
            scale=self.scale,
        )
        return out.permute(0, 2, 1, 3)  # [1, q_len, H, D]

    def _forward_encoder(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        """Encoder attention over contiguous q/k/v (no paged cache)."""
        cu_seqlens_q = attn_metadata.query_start_loc
        num_reqs = cu_seqlens_q.shape[0] - 1
        for i in range(num_reqs):
            qs, qe = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
            q_len = qe - qs
            if q_len == 0:
                continue
            q = query[qs:qe].permute(1, 0, 2).unsqueeze(0)
            k = key[qs:qe].permute(1, 0, 2).unsqueeze(0)
            v = value[qs:qe].permute(1, 0, 2).unsqueeze(0)
            if self.num_queries_per_kv > 1:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=self.scale
            )
            output[qs:qe] = out.permute(0, 2, 1, 3).squeeze(0)
        return output
