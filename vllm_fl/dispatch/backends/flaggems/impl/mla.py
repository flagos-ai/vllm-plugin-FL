# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/attention/backends/mla/flashattn_mla.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from typing import ClassVar, Optional, Union

import torch

from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
)
from vllm.utils.torch_utils import is_quantized_kv_cache

from vllm.logger import init_logger
from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

from flag_gems import (
    flash_attn_varlen_func,
    concat_and_cache_mla as flag_gems_concat_and_cache_mla,
    flash_mla_with_kvcache,
)

logger = init_logger(__name__)


class MLAFLMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


class MLAFLBackend(MLACommonBackend):
    @staticmethod
    def get_name() -> str:
        return "MLAFL"

    @staticmethod
    def get_impl_cls() -> type["MLAFLImpl"]:
        return MLAFLImpl

    @staticmethod
    def get_builder_cls() -> type["MLACommonMetadataBuilder"]:
        return MLAFLMetadataBuilder


class MLAFLImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
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
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonMLAImpl"
            )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonMLA V1 with FP8 KV cache not yet supported"
            )

        # FlagGems' flash_mla kernel computes sm_scale = 1/sqrt(d) internally
        # where d = head_dim = kv_lora_rank + qk_rope_head_dim.
        # But the correct sm_scale for MLA is 1/sqrt(qk_head_dim)
        # where qk_head_dim = qk_nope_head_dim + qk_rope_head_dim.
        self._qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        """Override KV cache update to use FlagGems' concat_and_cache_mla."""
        if kv_cache.numel() == 0:
            return
        flag_gems_concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
    ):
        """Override _compute_prefill_context to use FlagGems gather kernel.

        Replaces the native CUDA kernels gather_and_maybe_dequant_cache and
        cp_gather_cache with FlagGems' gather_and_maybe_dequant_cache kernel.
        The rest of the prefill-context logic (kv_b_proj, split, attention,
        merge) remains the same as MLACommonImpl.
        """
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.chunked_context is not None

        use_fp8_prefill = prefill_metadata.q_data_type == current_platform.fp8_dtype()

        output = None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace

        if use_fp8_prefill:
            q = q.to(prefill_metadata.q_data_type)

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]

            # ---- Gather KV cache into workspace ----
            if not use_fp8_prefill:
                if getattr(current_platform, "vendor_name", "") == "thead":
                    # T-Head backend: use FlagGems gather_and_maybe_dequant_cache
                    from flag_gems import gather_and_maybe_dequant_cache as flag_gems_gather_and_maybe_dequant_cache
                    flag_gems_gather_and_maybe_dequant_cache(
                        src_cache=kv_c_and_k_pe_cache,
                        dst=workspace,
                        block_table=prefill_metadata.block_table,
                        cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                        token_to_seq=prefill_metadata.chunked_context.token_to_seq[i],
                        num_tokens=int(prefill_metadata.chunked_context.chunk_total_token[i]),
                        kv_cache_dtype=self.kv_cache_dtype,
                        scale=k_scale,
                        seq_starts=prefill_metadata.chunked_context.starts[i],
                    )
                else:
                    # Other backends: use vLLM native op
                    ops.gather_and_maybe_dequant_cache(
                        src_cache=kv_c_and_k_pe_cache,
                        dst=workspace,
                        block_table=prefill_metadata.block_table,
                        cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                        token_to_seq=prefill_metadata.chunked_context.token_to_seq[i],
                        num_tokens=int(prefill_metadata.chunked_context.chunk_total_token[i]),
                        kv_cache_dtype=self.kv_cache_dtype,
                        scale=k_scale,
                        seq_starts=prefill_metadata.chunked_context.starts[i],
                    )
            else:
                # FP8 path: gather cache without dequantization
                ops.cp_gather_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=prefill_metadata.block_table,
                    cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                    batch_size=attn_metadata.num_prefills,
                    seq_starts=prefill_metadata.chunked_context.starts[i],
                )

            # Extract kv_c_normed from workspace
            kv_c_normed = workspace[:toks][..., : self.kv_lora_rank]
            # When FP8 weights are used without FP8 prefill, kv_b_proj expects
            # model dtype input and will quantize internally.
            # For quantized layers (AWQ/GPTQ) that lack a .weight attribute,
            # use params_dtype which is the expected input dtype.
            _kv_b_proj_w_dtype = (
                self.kv_b_proj.weight.dtype
                if hasattr(self.kv_b_proj, "weight")
                else self.kv_b_proj.params_dtype
            )
            # For NVFP4, weights are packed uint8 — keep input in model dtype
            # since the NVFP4 linear layer quantizes internally.
            if (
                use_fp8_prefill or _kv_b_proj_w_dtype != current_platform.fp8_dtype()
            ) and _kv_b_proj_w_dtype != torch.uint8:
                kv_c_normed = kv_c_normed.to(self.kv_b_proj.weight.dtype)

            k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )

            # To Do: Use epilogue of kv_b_proj to generate fp8 kv_nope.
            if use_fp8_prefill:
                kv_nope = kv_nope.to(prefill_metadata.q_data_type)
                k_pe = k_pe.to(prefill_metadata.q_data_type)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = self._concat_k_nope_k_pe(k_nope, k_pe)

            attn_output, attn_softmax_lse = self._run_prefill_context_chunk(
                prefill=prefill_metadata,
                chunk_idx=i,
                q=q,
                k=k,
                v=v,
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                merge_attn_states(
                    output=output_tmp,
                    output_lse=output_lse_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        # FlagGems' flash_attn_varlen_func (mha_varlan_fwd) requires k.size() == v.size(),
        # but MLA has different head dims: qk_head_dim (k) != v_head_dim (v).
        # Pad v with zeros to match q/k's head dimension.
        v_head_dim = v.shape[-1]
        if v.shape[-1] != q.shape[-1]:
            v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        # FlagGems does not support fa_version parameter, remove if present
        kwargs.pop("fa_version", None)

        # Ensure return_softmax_lse is passed to flash_attn_varlen_func
        if return_softmax_lse:
            kwargs["return_softmax_lse"] = True

        attn_out = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        # Unpack output if there are multiple results
        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        # Slice output back to original v_head_dim because we padded v above.
        # The padded dimensions contain zeros from padding and would be incorrect.
        if attn_out.shape[-1] != v_head_dim:
            attn_out = attn_out[..., :v_head_dim]

        if return_softmax_lse:
            return attn_out.clone(), lse.clone() if isinstance(lse, torch.Tensor) else lse
        return attn_out.clone()

    def forward_mqa(
        self,
        q: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 MLA FL not yet supported")

        # head_dim_v = kv_lora_rank (nope dim), head_dim = kv_lora_rank + qk_rope_head_dim
        head_dim_v = self.kv_lora_rank
        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        # q shape: (B, N, D) where D = kv_lora_rank + qk_rope_head_dim
        B = q.shape[0]
        q_num_heads = q.shape[1]
        head_dim = q.shape[2]

        # flash_mla_with_kvcache expects q shape: (batch_size, seq_len_q, num_heads_q, head_dim)
        # Current q shape: (B, N, D) -> reshape to (B, 1, N, D) for seq_len_q=1
        q_4d = q.unsqueeze(1)

        # flash_mla_with_kvcache expects k_cache shape: (num_blocks, block_size, num_heads_kv, head_dim)
        # vLLM's MLA kv cache is 3D: (num_blocks, block_size, head_dim) with implicit h_kv=1
        # Add the h_kv dimension
        k_cache_4d = kv_c_and_k_pe_cache.unsqueeze(2)  # -> (num_blocks, block_size, 1, head_dim)

        # Create FlashMLASchedMeta required by flash_mla_with_kvcache
        # MLACommonDecodeMetadata doesn't have scheduler_metadata, so we create a new one
        # using get_mla_metadata() which returns (FlashMLASchedMeta, None)
        # Use scheduler_metadata from attn_metadata if available, otherwise create empty one
        if hasattr(attn_metadata.decode, "scheduler_metadata"):
            scheduler_metadata = attn_metadata.decode.scheduler_metadata
        else:
            from flag_gems.fused.flash_mla_with_kvcache import get_mla_metadata
            scheduler_metadata, _ = get_mla_metadata()

        # Use self.scale which already includes the mscale correction from the model
        # (self.scale = qk_head_dim**-0.5 * mscale * mscale for YaRN models)
        softmax_scale = self.scale

        # Debug: print shapes once
        import os
        if not hasattr(self, '_debug_printed2'):
            self._debug_printed2 = True
            has_sched = hasattr(attn_metadata.decode, 'scheduler_metadata')
            print(f'[MLA DEBUG2] has scheduler_metadata: {has_sched}', flush=True)
            if has_sched:
                sm = attn_metadata.decode.scheduler_metadata
                print(f'[MLA DEBUG2] scheduler_metadata type: {type(sm)}, have_initialized: {getattr(sm, "have_initialized", "N/A")}', flush=True)

        # Call flash_mla_with_kvcache which returns (out, lse)
        o, lse = flash_mla_with_kvcache(
            q=q_4d,
            k_cache=k_cache_4d,
            block_table=attn_metadata.decode.block_table,
            cache_seqlens=attn_metadata.decode.seq_lens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=scheduler_metadata,
            softmax_scale=softmax_scale,
            causal=True,
        )

        # flash_mla_with_kvcache returns:
        # out: (batch_size, seq_len_q, num_heads_q, head_dim_v)
        # lse: (batch_size, num_heads_q, seq_len_q)
        # Reshape to (B, N, head_dim_v) for vLLM
        o = o.squeeze(1)  # Remove seq_len_q=1 dimension

        # Transpose lse from (B, N, 1) to match expected shape (B, N)
        lse = lse.squeeze(-1)  # Remove seq_len_q=1 dimension

        # .clone() is required: the Triton kernel output buffer may be reused
        # by CUDAGraph replay on subsequent iterations, so we must copy the
        # data out before returning to prevent it from being overwritten.
        return o.clone(), lse.clone()
