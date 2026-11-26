# SPDX-License-Identifier: Apache-2.0
"""Thin Iluvatar CoreX GQA Paged Attention Backend (`DG-R2-K8V4`, `K8V8`, and `BF16`).

Bridges vLLM V1's `TritonAttentionMetadata` and 3-plane `DG-R2-K8V4` KV-cache
bundle directly to FlagGems' pure-TLE `paged_gqa_attention`,
`paged_gqa_decode_attention_quant`, and `reshape_and_cache_k8v4` kernels with
zero per-step Python tensor allocations.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)

from flag_gems.runtime.backend._iluvatar.fused.paged_gqa_attention import (
    RCP_LN2,
    paged_gqa_attention,
    paged_gqa_decode_attention_quant,
    prewarm_attention_scratch,
    reshape_and_cache_k8v4,
)
from flag_gems.utils.quant_precondition import _alloc_q_and_s

from ..patches.kv_cache_k8v4 import bind_kv_cache_bundle, get_k8v4_kv_cache_shape

# Prefer the C-level capture query to avoid Python wrapper overhead on every layer forward.
_IS_CAPTURING_FN = getattr(torch._C, "_cuda_isCurrentStreamCapturing", torch.cuda.is_current_stream_capturing)


class IluvatarAttentionImpl(TritonAttentionImpl):
    """Thin vLLM Paged GQA adapter delegating directly to FlagGems `paged_gqa_attention`.

    Pre-validates static layer geometry (`_fl_layer_ok`), pre-scales softmax
    temperature by `log2(e)` (`_fl_sm_scale_log2e`) for hardware base-2 `exp2`,
    and memoizes carved 3-plane `DG-R2-K8V4` K/V + `bf16x2` scale views via
    `_get_kv_bundle` so `forward` and `forward_quant` act as pure pass-throughs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize base Triton attention state and pre-compute static FlagGems eligibility."""
        super().__init__(*args, **kwargs)
        gqa = self.num_heads // self.num_kv_heads if self.num_kv_heads > 0 else 0
        self._fl_layer_ok = (
            self.alibi_slopes is None and self.sliding_window == (-1, -1) and not bool(self.logits_soft_cap)
            and self.sinks is None and self.chunk_lookback == -1 and self.attn_type == "decoder"
            and self.head_size in (64, 128, 256) and self.num_kv_heads > 0 and self.num_heads % self.num_kv_heads == 0
            and 1 <= gqa <= 32 and self.kv_cache_dtype in ("auto", "bfloat16", "int8_per_token_head")
        )
        # Pre-multiply softmax scale by log2(e) so Triton kernels use native `tl.math.exp2`.
        self._fl_sm_scale_log2e = self.scale * RCP_LN2
        self._fl_bound_kv_sig: tuple[Any, ...] | None = None
        self._fl_bound_kv_bundle: tuple[Any, ...] | None = None

    def _get_kv_bundle(self, kv_cache: torch.Tensor) -> tuple[Any, ...]:
        """Return memoized `(kc, vc, ..., attn_ks, attn_vs, strides, flags)` tuple for `kv_cache`."""
        sig = self._fl_bound_kv_sig
        if sig is not None and kv_cache.data_ptr() == sig[0] and kv_cache.shape == sig[1] and kv_cache.dtype == sig[2]:
            return self._fl_bound_kv_bundle  # type: ignore[return-value]
        return bind_kv_cache_bundle(self, kv_cache)

    def do_kv_cache_update(
        self, layer: Any, key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        """Write pre-rotated `(key, value)` into `DG-R2-K8V4` / `K8V8` / `BF16` paged cache."""
        if not self._fl_layer_ok or kv_cache.ndim != 5 or kv_cache.shape[2] != 16:
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        k_len, sm_len = int(key.shape[0]), int(slot_mapping.shape[0])
        if (n := min(k_len, sm_len)) <= 0:
            return
        kc, vc, _, _, _, attn_ks, attn_vs, *_, int8_kv, _, _ = self._get_kv_bundle(kv_cache)
        reshape_and_cache_k8v4(
            key if k_len == n else key[:n], value if int(value.shape[0]) == n else value[:n],
            kc, vc, slot_mapping if sm_len == n else slot_mapping[:n],
            k_scale_cache=attn_ks if int8_kv else None, v_scale_cache=attn_vs if int8_kv else None, use_db_lsq=True,
        )

    def do_rope_and_kv_cache_update(
        self, layer: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        positions: torch.Tensor, cos_sin_cache: torch.Tensor, is_neox: bool,
        kv_cache: torch.Tensor, slot_mapping: torch.Tensor | None, attn_metadata: Any = None,
    ) -> torch.Tensor:
        """Execute fused 4-in-1 Q/K NeoX RoPE + DB-LSQ K8V4 cache write (`vllm::fused_rope_and_cache_k8v4`)."""
        q_len = int(query.shape[0])
        q_out = torch.empty((q_len, self.num_heads, self.head_size), dtype=query.dtype, device=query.device)
        if slot_mapping is None or kv_cache.numel() == 0:
            return q_out.zero_()
        sm_len, pos_len = int(slot_mapping.shape[0]), int(positions.shape[0])
        if (n := min(q_len, sm_len, pos_len)) <= 0:
            return q_out
        kc, vc, _, _, _, attn_ks, attn_vs, *_, int8_kv, _, _ = self._get_kv_bundle(kv_cache)
        reshape_and_cache_k8v4(
            key if int(key.shape[0]) == n else key[:n], value if int(value.shape[0]) == n else value[:n],
            kc, vc, slot_mapping if sm_len == n else slot_mapping[:n],
            k_scale_cache=attn_ks if int8_kv else None, v_scale_cache=attn_vs if int8_kv else None,
            query=query if q_len == n else query[:n], query_out=q_out if q_len == n else q_out[:n],
            positions=positions if pos_len == n else positions[:n], cos_sin_cache=cos_sin_cache,
            use_db_lsq=True, is_neox=bool(is_neox),
        )
        return q_out

    def do_qk_norm_rope_and_kv_cache_update(
        self, layer: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor, rms_eps: float,
        positions: torch.Tensor, cos_sin_cache: torch.Tensor, is_neox: bool,
        kv_cache: torch.Tensor, slot_mapping: torch.Tensor | None, attn_metadata: Any = None,
    ) -> torch.Tensor:
        """Execute fused 5-in-1 per-head QK-Norm + Q/K NeoX RoPE + DB-LSQ K8V4 cache write."""
        q_len = int(query.shape[0])
        q_out = torch.empty((q_len, self.num_heads, self.head_size), dtype=query.dtype, device=query.device)
        if slot_mapping is None or kv_cache.numel() == 0:
            return q_out.zero_()
        sm_len, pos_len = int(slot_mapping.shape[0]), int(positions.shape[0])
        if (n := min(q_len, sm_len, pos_len)) <= 0:
            return q_out
        kc, vc, _, _, _, attn_ks, attn_vs, *_, int8_kv, _, _ = self._get_kv_bundle(kv_cache)
        reshape_and_cache_k8v4(
            key if int(key.shape[0]) == n else key[:n], value if int(value.shape[0]) == n else value[:n],
            kc, vc, slot_mapping if sm_len == n else slot_mapping[:n],
            k_scale_cache=attn_ks if int8_kv else None, v_scale_cache=attn_vs if int8_kv else None,
            query=query if q_len == n else query[:n], query_out=q_out if q_len == n else q_out[:n],
            positions=positions if pos_len == n else positions[:n], cos_sin_cache=cos_sin_cache,
            q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight, rms_eps=float(rms_eps),
            use_db_lsq=True, is_neox=bool(is_neox),
        )
        return q_out

    def forward(
        self, layer: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        kv_cache: torch.Tensor, attn_metadata: Any, output: torch.Tensor,
        output_scale: torch.Tensor | None = None, output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pass-through BF16 forward dispatching Prefill / Mixed / Decode to `paged_gqa_attention`.

        Forwards POD sub-batch horizons (`num_prefills`, `prefill_seq_idx`,
        `max_prefill_seq_len`, `max_decode_seq_len`, `mean_decode_seq_len`) during
        eager Mixed/Prefill steps while omitting host-dynamic POD scalars (`None`)
        during CUDA Graph capture so captured Pure-Decode graphs remain shape-invariant.
        """
        is_capturing = bool(_IS_CAPTURING_FN())
        if attn_metadata is None:
            if self._fl_layer_ok and query.is_cuda and not is_capturing:
                prewarm_attention_scratch(query.device, self.num_heads, self.head_size)
            return output.fill_(0)

        if (
            not self._fl_layer_ok or output_scale is not None or output_block_scale is not None
            or not isinstance(attn_metadata, TritonAttentionMetadata) or bool(attn_metadata.use_cascade)
            or getattr(attn_metadata, "sinks", None) is not None
            or getattr(attn_metadata, "mm_prefix_range", None) is not None
            or getattr(attn_metadata, "mm_prefix_range_tensor", None) is not None
            or attn_metadata.causal is not True or query.dtype != torch.bfloat16 or not query.is_cuda
            or output.dtype != torch.bfloat16 or not output.is_contiguous()
            or kv_cache.ndim != 5 or kv_cache.shape[2] != 16 or kv_cache.dtype not in (torch.bfloat16, torch.int8)
        ):
            return super().forward(layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)

        if (num_actual := int(attn_metadata.num_actual_tokens)) <= 0:
            return output
        q = query if int(query.shape[0]) == num_actual else query[:num_actual]
        out = output if int(output.shape[0]) == num_actual else output[:num_actual]
        if int(q.shape[0]) <= 0:
            return output

        kc, vc, _, _, _, attn_ks, attn_vs, kc_s, vc_s, _, _, _, attn_ks_s, attn_vs_s, wide_kv, int8_kv, _, _ = (
            self._get_kv_bundle(kv_cache)
        )
        raw_ms = getattr(attn_metadata, "max_seq_len", None)
        max_seq_len = int(raw_ms) if isinstance(raw_ms, int) else int(attn_metadata.block_table.shape[1]) * 16
        paged_gqa_attention(
            q, kc, vc, attn_metadata.query_start_loc, attn_metadata.seq_lens, attn_metadata.block_table,
            max_query_len=int(attn_metadata.max_query_len), max_seq_len=max_seq_len,
            sm_scale_log2e=self._fl_sm_scale_log2e, output=out,
            key_scale=attn_ks if int8_kv else None, value_scale=attn_vs if int8_kv else None,
            compact=None, allow_zero_len=False,
            num_prefills=None if is_capturing else getattr(attn_metadata, "num_prefills", None),
            max_prefill_seq_len=None if is_capturing else getattr(attn_metadata, "max_prefill_seq_len", None),
            max_prefill_q_len=None if is_capturing else getattr(attn_metadata, "max_prefill_q_len", None),
            prefill_seq_idx=None if is_capturing else getattr(attn_metadata, "prefill_seq_idx", None),
            num_decodes=None if is_capturing else getattr(attn_metadata, "num_decode_seqs", None),
            max_decode_seq_len=None if is_capturing else getattr(attn_metadata, "max_decode_seq_len", None),
            mean_decode_seq_len=None if is_capturing else getattr(attn_metadata, "mean_decode_seq_len", None),
            is_capturing=is_capturing, kc_strides=kc_s, vc_strides=vc_s, ks_strides=attn_ks_s, vs_strides=attn_vs_s,
            wide_kv=wide_kv,
        )
        return output

    def forward_quant(
        self, layer: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        kv_cache: torch.Tensor, attn_metadata: Any, do_hadamard: bool, num_scale_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Pure-Decode fused attention + INT8 output quantizer invoked by `fl_unified_attention_quant`.

        When `QuaRot-R2` has already absorbed head rotations (`do_hadamard=False`,
        `num_scale_groups == 1`) and the step is Pure Decode (`max_query_len == 1`),
        fuses Stage-2 split-K reduction directly with dynamic per-row INT8 quantization
        into `(out_q, out_s)`, saving an entire `[M, H*D]` BF16 HBM round-trip.
        Returns `None` on Prefill/Mixed steps so `_fl_unified_attention_quant_impl`
        transparently falls back to `forward` + `_launch_fused_act_quant`.
        """
        if do_hadamard or num_scale_groups != 1 or not self._fl_layer_ok:
            return None
        is_capturing = bool(_IS_CAPTURING_FN())
        m = int(query.shape[0])
        k = self.num_heads * self.head_size
        if attn_metadata is None:
            if query.is_cuda and not is_capturing:
                prewarm_attention_scratch(query.device, self.num_heads, self.head_size)
            out_q, out_s = _alloc_q_and_s(query, m, k, 1)
            return out_q.zero_(), out_s.fill_(1e-12)
        if (
            not isinstance(attn_metadata, TritonAttentionMetadata) or bool(attn_metadata.use_cascade)
            or getattr(attn_metadata, "sinks", None) is not None
            or getattr(attn_metadata, "mm_prefix_range", None) is not None
            or getattr(attn_metadata, "mm_prefix_range_tensor", None) is not None
            or attn_metadata.causal is not True or int(attn_metadata.max_query_len) != 1
            or int(attn_metadata.num_actual_tokens) != m or m <= 0
            or query.dtype != torch.bfloat16 or not query.is_cuda
            or kv_cache.ndim != 5 or kv_cache.shape[2] != 16 or kv_cache.dtype not in (torch.bfloat16, torch.int8)
        ):
            return None
        kc, vc, _, _, _, attn_ks, attn_vs, kc_s, vc_s, _, _, _, attn_ks_s, attn_vs_s, wide_kv, int8_kv, _, _ = (
            self._get_kv_bundle(kv_cache)
        )
        return paged_gqa_decode_attention_quant(
            query, kc, vc, attn_metadata.query_start_loc, attn_metadata.seq_lens, attn_metadata.block_table,
            sm_scale_log2e=self._fl_sm_scale_log2e,
            key_scale=attn_ks if int8_kv else None, value_scale=attn_vs if int8_kv else None,
            is_capturing=is_capturing, kc_strides=kc_s, vc_strides=vc_s, ks_strides=attn_ks_s, vs_strides=attn_vs_s,
            wide_kv=wide_kv,
        )


class IluvatarAttentionBackend(TritonAttentionBackend):
    """Iluvatar Triton attention backend with `DG-R2-K8V4` 3-plane storage and pure-TLE forward.

    Sets `forward_includes_kv_cache_update = False` so vLLM emits explicit
    `vllm.unified_kv_cache_update` nodes during FX tracing, allowing
    `RopeKvFusionPass` to fuse Q/K NeoX RoPE and `DG-R2-K8V4` cache writes cleanly.
    """

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        """Return the registered vLLM attention backend name (`TRITON_ATTN`)."""
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls():
        """Return `IluvatarAttentionImpl` for per-layer attention execution."""
        return IluvatarAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int, block_size: int, num_kv_heads: int, head_size: int, cache_dtype_str: str = "auto"
    ) -> tuple[int, ...]:
        """Return 3-plane `DG-R2-K8V4` shape for `int8_per_token_head`, or standard 2-plane shape otherwise."""
        if cache_dtype_str == "int8_per_token_head":
            return get_k8v4_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)
        return TritonAttentionBackend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)

    @staticmethod
    def get_builder_cls():
        """Return vLLM's `TritonAttentionMetadataBuilder`."""
        return TritonAttentionMetadataBuilder
