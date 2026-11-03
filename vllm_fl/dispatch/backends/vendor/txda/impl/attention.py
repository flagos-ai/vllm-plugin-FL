# Copyright (c) 2026 BAAI. All rights reserved.
# Ported from the vLLM 0.24 TXDA enablement (537284eb, #460) for vLLM 0.28.

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

import torch

from vllm.v1.attention.backend import AttentionCGSupport, AttentionType

from vllm_fl.dispatch.backends.flaggems.impl.attention import (
    AttentionFLBackend,
    AttentionFLImpl,
    AttentionFLMetadataBuilder,
)


class TxdaSDPAAttentionBackend(AttentionFLBackend):
    """Attention backend for tsingmicro TX devices using torch SDPA."""

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_builder_cls() -> type[TxdaSDPAMetadataBuilder]:
        return TxdaSDPAMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
        )

    @staticmethod
    def get_impl_cls() -> type[TxdaSDPAAttentionImpl]:
        return TxdaSDPAAttentionImpl


class TxdaSDPAMetadataBuilder(AttentionFLMetadataBuilder):
    # The reference implementation reads lengths and page IDs on the host.
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        if vllm_config.parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError(
                "TXDA SDPA does not support decode context parallelism"
            )
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        # Shared prefix blocks remain in the ordinary block table. Do not ask
        # the inherited builder for the separate cascade attention algorithm.
        return super().build(0, common_attn_metadata, fast_build)

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class TxdaSDPAAttentionImpl(AttentionFLImpl):
    """
    SDPA-based attention impl for TX8110.

    do_kv_cache_update and forward are overridden to avoid the flag_gems
    kernels (reshape_and_cache_flash / flash_attn_varlen_func) that compute
    silently wrong values on TX8110. The KV cache layout
    (2, num_blocks, block_size, num_kv_heads, head_size) and metadata are
    unchanged; the metadata builder disables cascade and graph capture.
    """

    can_return_lse_for_decode: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not TxdaSDPAAttentionBackend.supports_attn_type(self.attn_type):
            raise NotImplementedError(
                "TXDA SDPA does not support encoder-decoder attention"
            )
        if self.alibi_slopes is not None:
            raise NotImplementedError("ALiBi is not supported on TXDA SDPA")
        if self.logits_soft_cap:
            raise NotImplementedError("Logits soft cap is not supported on TXDA SDPA")

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
        # Dynamic boolean indexing is expensive on TXDA, and the native
        # index_put_ fallback transfers the cache through the CPU. Copy only
        # selected rows, coalescing consecutive slots within each block so
        # prefill does not require a Python iteration/copy per token.
        slots = slot_mapping.tolist()
        index = 0
        while index < len(slots):
            slot = slots[index]
            if slot < 0:
                index += 1
                continue
            block, offset = divmod(slot, block_size)
            end = index + 1
            while (
                end < len(slots)
                and end - index < block_size - offset
                and slots[end] == slot + end - index
            ):
                end += 1
            count = end - index
            # Native copy_ supports strided input and NHD/HND cache views,
            # including the blocks-first layout used by hybrid models.
            key_cache[block, offset : offset + count].copy_(key[index:end])
            value_cache[block, offset : offset + count].copy_(value[index:end])
            index = end

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
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

        if attn_metadata.use_cascade:
            raise NotImplementedError("cascade attention not supported on TXDA_SDPA")

        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
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
            block_size = key_cache.shape[1]
            num_blocks = (seq_len + block_size - 1) // block_size
            blocks = block_table[i, :num_blocks].tolist()
            k = torch.cat([key_cache[b] for b in blocks], dim=0).reshape(
                -1, self.num_kv_heads, self.head_size
            )[:seq_len]
            v = torch.cat([value_cache[b] for b in blocks], dim=0).reshape(
                -1, self.num_kv_heads, self.head_size
            )[:seq_len]
            q = query[qs:qe]

            out_i = self._sdpa(
                q, k, v, seq_len, window_left, causal=attn_metadata.causal
            )
            # output is [num_tokens, num_heads, head_size]; out_i matches directly.
            output[qs:qe] = out_i

        return output

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int,
        window_left: int,
        causal: bool = True,
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

        # SDPA's is_causal flag uses upper-left alignment. Chunked prefill
        # needs bottom-right alignment, and single-token decode still needs
        # the sliding-window lower bound when configured.
        is_causal = causal and q_start == 0 and window_left < 0
        attn_mask = None
        if window_left >= 0 or (causal and q_start != 0 and q_len > 1):
            rows = torch.arange(q_len, device=q.device).unsqueeze(1)
            cols = torch.arange(seq_len, device=q.device).unsqueeze(0)
            positions = q_start + rows
            if causal:
                visible = cols <= positions
            else:
                visible = torch.ones(
                    (q_len, seq_len), dtype=torch.bool, device=q.device
                )
            if window_left >= 0:
                visible &= cols >= (positions - window_left)
                if not causal and self.sliding_window[1] >= 0:
                    visible &= cols <= (positions + self.sliding_window[1])
            attn_mask = visible
            is_causal = False

        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            kk,
            vv,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=self.scale,
        )
        return out.squeeze(0).transpose(0, 1)  # [q_len, H, D]

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
            output[qs:qe] = self._sdpa(
                query[qs:qe],
                key[qs:qe],
                value[qs:qe],
                q_len,
                self.sliding_window[0],
                causal=False,
            )
        return output
