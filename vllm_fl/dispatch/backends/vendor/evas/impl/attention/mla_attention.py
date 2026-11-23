# Copyright (c) 2026 BAAI. All rights reserved.
"""Default evas MLA attention backend for LPU (upstream MLACommon framework).

``EvasMLAAttentionBackend`` is the shared base, ``EvasMLABackend`` the dense
backend: its ``MLACommonImpl`` subclass keeps the framework's prefill
``forward_mha`` / decode ``forward_impl`` and overrides only
``_flash_attn_varlen_diff_headdims``, ``forward_mqa``
(``flash_mla_with_kvcache``) and ``do_kv_cache_update``
(``_write_paged_runs``).  The forward runs through the unified ops
(``opaque_attention_op()``); GLM has its own module.

"""

from __future__ import annotations

from typing import ClassVar

import torch

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    AttentionType,
)

from vllm_fl.dispatch.backends.vendor.evas.impl.attention.cache import (
    _write_paged_runs,
)


class EvasMLAAttentionBackend(AttentionBackend):
    """Thin shared base for evas MLA backends (default + GLM).

    Fixes the generic choices common to every evas MLA backend (block_size=256,
    bf16, ``"CUSTOM"`` enum slot, kernel block size 256) so concrete backends
    only override the model-specific pieces. The cache layout and impl/builder
    classes stay abstract; never instantiated directly.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float32]
    _supported_block_sizes: ClassVar[list[int]] = [256]

    @staticmethod
    def get_name() -> str:
        # "CUSTOM" so AttentionBackendEnum["CUSTOM"] resolves (same enum slot
        # the GQA backend registers under on LPU).
        return "CUSTOM"

    @staticmethod
    def get_kv_cache_storage_dtype() -> torch.dtype:
        # BF16 (NOT int8). The FL ModelRunnerFL._reshape_kv_cache_tensors has an
        # int8 storage path that multiplies the last dim by dtype_size; returning
        # bf16 gives a clean (num_blocks, 1, 256, head_dim) bf16 view.
        return torch.bfloat16

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return list(EvasMLAAttentionBackend._supported_block_sizes)


class EvasMLABackend(EvasMLAAttentionBackend, MLACommonBackend):
    """Default evas dense-MLA backend for LPU (upstream MLACommon framework).

    Combines the generic LPU MLA config from ``EvasMLAAttentionBackend``
    (``"CUSTOM"`` enum slot, block_size=256, bf16 storage) with the upstream
    ``MLACommonBackend`` layout/builder contract (3D ``(num_blocks, block_size,
    head_size)`` cache, C-contiguous stride, ``MLACommonMetadataBuilder``,
    ``is_mla()=True``); overrides only ``get_impl_cls`` to plug the LPU
    pure-PyTorch impl.
    """

    @staticmethod
    def get_impl_cls() -> type:
        return EvasMLAAttentionImpl


class EvasMLAAttentionImpl(MLACommonImpl):
    """Thin dense-MLA impl on the upstream MLACommon framework for LPU.

    ``MLACommonImpl`` already implements prefill ``forward_mha``
    (kv_b_proj decompress → ``_concat_k_nope_k_pe`` → prefill hook) and the
    layer's decode ``forward_impl`` (W_UK_T bmm up-project + ``_v_up_proj``);
    this impl only replaces the hardware-bound ops with flaggems evas
    overrides: ``_flash_attn_varlen_diff_headdims`` (flaggems
    ``flash_attn_varlen_func`` prefill; natively supports v_head_dim !=
    qk_head_dim so ``_pad_v=False``), ``forward_mqa`` (flaggems
    ``flash_mla_with_kvcache`` dense paged decode), and ``do_kv_cache_update``
    (shared ``_write_paged_runs``).

    ``MLACommonImpl.__init__`` is called: it does not select a CUDA prefill
    backend from env — the ``MLAAttention`` layer builds ``self.prefill_backend``
    via ``get_mla_prefill_backend``, which on LPU resolves to
    ``FlashAttnPrefillBackend`` routed to the flaggems evas varlen by
    ``_patch_fa_utils_for_lpu``.
    """

    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj,
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        # NOTE: do NOT call super().__init__() — MLACommonImpl.__init__ picks a
        # CUDA prefill backend from env and raises when flash_attn is missing
        # (the case on LPU).  dcp/pcp ranks are set by AttentionImplBase.__new__.
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported for MLA on LPU")
        if any((alibi_slopes, sliding_window, logits_soft_cap)):
            raise NotImplementedError(
                "EvasMLAAttentionImpl does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Only DECODER attention is supported for MLA on LPU"
            )

        # vllm 0.24 MLAAttention passes this whenever use_sparse (GLM is_v32);
        # backbone skip layers (indexer=None) still receive it (mla_attention.py:457-459).
        # Stored for fidelity; the GLM subclass overrides forward and never
        # calls through this impl for the sparse path.
        self.topk_indices_buffer = topk_indices_buffer

        # MLACommonImpl.__init__ is LPU-safe: it does not select a CUDA prefill
        # backend from env — the MLAAttention layer builds self.prefill_backend
        # via get_mla_prefill_backend -> FlashAttnPrefillBackend, which routes to
        # the flaggems evas varlen. Call it for the standard attrs + metadata
        # wiring, then apply the LPU overrides below.
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_b_proj,
            indexer,
            q_pad_num_heads,
        )
        # BF16 cache -> no pre-quantized query input.
        self.supports_quant_query_input = False

    # ------------------------------------------------------------------ prefill
    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        # LPU safety: chunked prefill (with-context) needs the workspace merge
        # path in forward_mha, not supported yet — single-turn e2e
        # (max_context_len_cpu == 0) never sets chunked_context.
        assert attn_metadata.prefill is not None
        assert attn_metadata.prefill.chunked_context is None, (
            "chunked prefill (with-context) not supported on LPU yet"
        )
        super().forward_mha(
            q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata, k_scale,
            output, output_scale
        )

    def _flash_attn_varlen_diff_headdims(
        self,
        q,
        k,
        v,
        return_softmax_lse=False,
        softmax_scale=None,
        **kwargs,
    ):
        """flaggems evas ``flash_attn_varlen_func`` (pure-PyTorch prefill).

        Single override point for upstream ``_run_prefill_new_tokens_fa`` /
        ``_run_prefill_context_chunk_fa``. Evas varlen supports v_head_dim !=
        qk_head_dim natively, so ``_pad_v`` stays False. ``cu_seqlens`` are
        CPU-cloned: the body reads each bound via ``.item()`` in a Python loop
        (timing-sensitive on LPU async worker buffers).
        """
        kwargs["cu_seqlens_q"] = kwargs["cu_seqlens_q"].cpu()
        kwargs["cu_seqlens_k"] = kwargs["cu_seqlens_k"].cpu()
        out = self._flag_gems_flash_attn_varlen(
            q=q.contiguous(),
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            **kwargs,
        )
        lse = None
        if isinstance(out, tuple):
            out, lse = out[0], out[1]
        if return_softmax_lse:
            return out, lse
        return out

    # ------------------------------------------------------------------- decode
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)  # [B, num_heads, kv_lora_rank + qk_rope]
        # flash_mla_with_kvcache wants [batch, seq_q, h_q, d_qk].
        q = q.unsqueeze(1).contiguous()

        k_cache = kv_c_and_k_pe_cache
        if k_cache.ndim == 3:
            # Add the implicit single kv-head dim: [num_blocks, block_size, 1, d_qk].
            k_cache = k_cache.unsqueeze(-2)
        assert k_cache.ndim == 4

        # MLA decode factorizes o_t = sum_j a_tj c_j (attention over the latent),
        # then _v_up_proj up-projects [B, N, kv_lora_rank] via W_UV. The dense
        # decode reads V as the first head_dim_v dims of the cached vector
        # ([latent | k_pe]) while the score uses all d_qk dims, so head_dim_v
        # MUST be kv_lora_rank (not v_head_dim) for _v_up_proj's (B, N, L) view.
        out, lse = self._flag_gems_flash_mla_with_kvcache(
            q=q,
            k_cache=k_cache,
            block_table=attn_metadata.decode.block_table,
            cache_seqlens=attn_metadata.decode.seq_lens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=None,
            softmax_scale=self.scale,
            causal=True,
            is_fp8_kvcache=False,
        )
        # out [B, 1, num_heads, kv_lora_rank] -> [B, num_heads, kv_lora_rank].
        out = out.squeeze(1)
        return out, lse

    # ------------------------------------------------------- KV cache update
    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        if slot_mapping is None:
            return

        # Concatenate [latent | RoPE] into the 576-dim cached vector.  k_pe
        # arrives [T, 1, qk_rope]; collapse to [T, qk_rope].
        k_pe_2d = k_pe.reshape(kv_c_normed.shape[0], self.qk_rope_head_dim)
        latent = torch.cat([kv_c_normed, k_pe_2d], dim=-1)  # [T, head_size]

        # Run-grouped consecutive-slot paged write (shared with GLM), 3D cache.
        _write_paged_runs(
            kv_cache, latent, slot_mapping, kv_cache.shape[1]
        )

    # ------------------------------------------------------------- helpers
    def _flag_gems_flash_attn_varlen(self, **kwargs):
        import flag_gems

        return flag_gems.flash_attn_varlen_func(**kwargs)

    def _flag_gems_flash_mla_with_kvcache(self, **kwargs):
        import flag_gems

        fn = getattr(flag_gems, "flash_mla_with_kvcache", None)
        if fn is None:
            from flag_gems.fused import flash_mla_with_kvcache as fn
        return fn(**kwargs)
