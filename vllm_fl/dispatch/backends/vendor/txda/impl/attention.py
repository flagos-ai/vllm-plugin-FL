# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda (tsingmicro) SDPA attention backend.

The flag_gems ``flash_attn_varlen_func`` kernel computes silently wrong values
on TX8110 (probe: maxrel=inf/nan/37.7 across the causal and non-causal varlen
cases), so the flag_gems attention impl cannot be used there. This backend
reuses the flag_gems metadata machinery (KV cache layout, block table, slot
mapping) but computes attention itself: KV writes go through basic indexing and
the attention math through torch SDPA, both of which are numerically correct on
txda. Compiler-independent: works under both the flagtree and triton compilers.
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


def _scatter_slots(cache: torch.Tensor, src: torch.Tensor, slots: list[int]) -> None:
    """Copy ``src`` rows into the paged ``cache`` at the given token slots.

    Uses basic (int/slice) indexing only. A tensor-index assignment on txda has
    no kernel and round-trips the *whole* cache through host memory, so its cost
    tracks the cache size instead of the rows written: on the engine's 284 MiB
    cache that is ~166 ms whether 1 row or 2048 are written, which alone costs
    seconds per decode step. Basic indexing is ~0.02 ms per op.
    """
    block_size = cache.shape[1]
    n = len(slots)
    i = 0
    while i < n:
        slot = slots[i]
        # Padded positions carry slot_mapping == -1; without this guard they
        # index the last block (floor division: -1 // block_size == -1) and
        # corrupt the cache with padded garbage.
        if slot < 0:
            i += 1
            continue
        # Consecutive slots copy in a single op. Stopping at the block boundary
        # keeps every write inside one block, where a slice is contiguous.
        j = i + 1
        while j < n and slots[j] == slot + (j - i) and (slot + (j - i)) % block_size:
            j += 1
        block_id, offset = divmod(slot, block_size)
        cache[block_id, offset : offset + (j - i)] = src[i:j]
        i = j


class TxdaSDPAAttentionBackend(AttentionFLBackend):
    """Attention backend for tsingmicro TX devices using torch SDPA.

    Inherits ``forward_includes_kv_cache_update = False`` from
    AttentionFLBackend, which is what makes vLLM call ``do_kv_cache_update``
    separately instead of expecting forward() to write the cache. The KV cache
    layout and metadata are the flag_gems ones, so AttentionFLMetadataBuilder is
    reused as-is.

    ``get_name`` is deliberately not overridden: vLLM requires the name to be a
    member of AttentionBackendEnum, which AttentionFLBackend already satisfies.
    """

    @staticmethod
    def get_impl_cls() -> type["TxdaSDPAAttentionImpl"]:
        return TxdaSDPAAttentionImpl


class TxdaSDPAAttentionImpl(AttentionFLImpl):
    """
    SDPA-based attention impl for TX8110.

    ``do_kv_cache_update`` and ``forward`` are overridden to avoid the flag_gems
    kernels (``reshape_and_cache_flash`` / ``flash_attn_varlen_func``) that
    compute silently wrong values on TX8110. The KV cache layout
    (2, num_blocks, block_size, num_kv_heads, head_size) and the metadata are
    unchanged.
    """

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """Write key/value into the paged KV cache.

        Avoids flag_gems reshape_and_cache_flash (wrong on TX8110).
        """
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        key_cache, value_cache = kv_cache.unbind(0)
        slots = slot_mapping.tolist()
        _scatter_slots(key_cache, key, slots)
        _scatter_slots(value_cache, value, slots)

        if _DEBUG and _PRINTED[0] < 400:
            _PRINTED[0] += 1
            print(
                f"[txda-debug] kv_update#{_PRINTED[0]} n={key.shape[0]} "
                f"k0={key[0].reshape(-1)[:4].tolist()} "
                f"v0={value[0].reshape(-1)[:4].tolist()} "
                f"slot0={slots[0]} slotN={slots[-1]} "
                f"block_size={key_cache.shape[1]}",
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
        """Forward with per-request torch SDPA over the paged KV cache."""
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
        # .item() on txda raises "txMemcpyAsync(...) = Invalid parameters" -- the
        # 0-dim scalar read path is broken, while tolist() (0-dim included) and
        # .cpu() work. Read both row vectors once and index on the host instead.
        cu_q = cu_seqlens_q.tolist()
        sl = seq_lens.tolist()
        for i in range(num_reqs):
            qs, qe = cu_q[i], cu_q[i + 1]
            q_len = qe - qs
            seq_len = sl[i]
            if q_len == 0 or seq_len == 0:
                output[qs:qe] = 0
                continue

            # Gather this request's KV from the paged cache. Advanced indexing
            # (key_cache[blocks]) has no PrivateUse1 kernel on txda, so it falls
            # back to a CPU copy of the whole cache and hangs at engine scale.
            # Gather via per-block slices + cat instead (probe-verified).
            #
            # The block table row is padded out to max_model_len / block_size,
            # and each padded entry still costs a device op here: on a
            # 2048-token config a 10-token decode would gather 128 blocks per
            # layer to keep 1. Only the blocks that hold tokens are read.
            n_blocks = -(-seq_len // key_cache.shape[1])
            blocks = block_table[i].tolist()[:n_blocks]
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
                    f"n={num_actual_tokens} cu_q={cu_q} "
                    f"seq_lens={sl} reqs={num_reqs} "
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
        cu_q = cu_seqlens_q.tolist()  # .item() is broken on txda; see forward()
        for i in range(num_reqs):
            qs, qe = cu_q[i], cu_q[i + 1]
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
