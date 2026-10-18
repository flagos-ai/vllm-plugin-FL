# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/attention/backends/mla/flashattn_mla.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch

from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
)
from vllm.utils.torch_utils import is_quantized_kv_cache

# from vllm.attention.ops.triton_decode_attention import decode_attention_fwd
# from vllm.attention.ops.triton_flash_attention import triton_attention
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
)

from flag_gems import flash_attn_varlen_func, flash_mla

logger = init_logger(__name__)


class MLAFLBackend(MLACommonBackend):
    @staticmethod
    def get_name() -> str:
        return "MLAFL"

    @staticmethod
    def get_impl_cls() -> type["MLAFLImpl"]:
        return MLAFLImpl


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
        # Ensure flash_attn_varlen_func is not None so parent __init__ doesn't
        # reject us. We override prefill with _flash_attn_varlen_diff_headdims.
        import vllm.model_executor.layers.attention.mla_attention as _mla_mod
        _orig_fa = _mla_mod.flash_attn_varlen_func
        if _orig_fa is None:
            _mla_mod.flash_attn_varlen_func = flash_attn_varlen_func

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

        # Restore original value
        _mla_mod.flash_attn_varlen_func = _orig_fa

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
        import flag_gems
        if flag_gems.vendor_name == "kunlunxin":
            import xtorch_ops
            xtorch_ops.concat_and_cache_mla(
                kv_c_normed, k_pe.squeeze(1), slot_mapping.flatten(), kv_cache
            )
        else:
            from vllm import _custom_ops as ops
            ops.concat_and_cache_mla(
                kv_c_normed, k_pe.squeeze(1), kv_cache,
                slot_mapping.flatten(), kv_cache_dtype, k_scale,
            )

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        """
        Modified to support Kunlunxin backend using xtorch_ops.prefill_attention.

        Kunlunxin backend uses native API which supports head_dim=576 for MLA.
        Other backends continue to use FlagGems flash_attn_varlen_func.
        """
        import flag_gems

        # DEBUG: Print to confirm this method is being called
        print(f"[MLA DEBUG] _flash_attn_varlen_diff_headdims called! vendor={flag_gems.vendor_name}, q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}", flush=True)

        # Kunlunxin-specific path: use xtorch_ops.prefill_attention
        if flag_gems.vendor_name == "kunlunxin":
            print(f"[MLA DEBUG] Using Kunlunxin xtorch_ops.prefill_attention path", flush=True)
            import xtorch_ops

            # Extract varlen parameters from kwargs
            cu_seqlens_q = kwargs.get("cu_seqlens_q")
            cu_seqlens_k = kwargs.get("cu_seqlens_k")
            causal = kwargs.get("causal", True)

            if cu_seqlens_q is None or cu_seqlens_k is None:
                raise ValueError(
                    "Kunlunxin prefill_attention requires cu_seqlens_q and cu_seqlens_k"
                )

            # Construct LOD tensors (cumulative sequence lengths)
            context_qlen_lod_cpu = cu_seqlens_q.cpu() if cu_seqlens_q.device.type != "cpu" else cu_seqlens_q
            context_qlen_lod_xpu = cu_seqlens_q.to(q.device)
            context_kvlen_lod_cpu = cu_seqlens_k.cpu() if cu_seqlens_k.device.type != "cpu" else cu_seqlens_k
            context_kvlen_lod_xpu = cu_seqlens_k.to(q.device)

            # Pad v to match q/k head_dim (same as FlagGems approach)
            v_head_dim = v.shape[-1]
            if v.shape[-1] != q.shape[-1]:
                v = torch.nn.functional.pad(
                    v, [0, q.shape[-1] - v.shape[-1]], value=0
                )

            # Prepare output tensor
            output = torch.empty_like(q)

            # Call xtorch_ops.prefill_attention (modifies output in-place)
            print(f"[MLA DEBUG] Calling xtorch_ops.prefill_attention...", flush=True)
            ret_code = xtorch_ops.prefill_attention(
                q=q,
                k=k,
                v=v,
                out=output,
                is_causal=causal,
                is_prefix_cache=False,
                alpha=softmax_scale if softmax_scale is not None else 1.0,
                context_qlen_lod_cpu=context_qlen_lod_cpu,
                context_qlen_lod_xpu=context_qlen_lod_xpu,
                context_kvlen_lod_cpu=context_kvlen_lod_cpu,
                context_kvlen_lod_xpu=context_kvlen_lod_xpu,
            )

            print(f"[MLA DEBUG] xtorch_ops.prefill_attention returned code={ret_code}", flush=True)
            if ret_code != 0:
                raise RuntimeError(f"xtorch_ops.prefill_attention failed with code {ret_code}")

            # Slice output back to original v_head_dim
            if output.shape[-1] != v_head_dim:
                output = output[..., :v_head_dim]

            # xtorch_ops.prefill_attention doesn't return LSE
            if return_softmax_lse:
                return output.clone(), None
            return output.clone()

        # Original FlagGems path for other backends
        print(f"[MLA DEBUG] Using original FlagGems flash_attn_varlen_func path", flush=True)
        maybe_padded_v = v
        if self._pad_v:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        kwargs["return_softmax_lse"] = return_softmax_lse
        attn_out = flash_attn_varlen_func(
            q=q,
            k=k,
            v=maybe_padded_v,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        # Unpack the output if there is multiple results
        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        # Remain consistent with old `flash_attn_varlen_func` where there
        # is only one output tensor if `return_softmax_lse` is False.
        if return_softmax_lse:
            return attn_out, lse
        return attn_out

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

        head_dim_v = 0
        if type(q) is tuple:
            ### q_nope & q_pe
            head_dim_v = q[0].shape[-1]
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        q_num_heads = q.shape[1]
        head_dim = q.shape[-1]
        # o = torch.zeros(B,
        #                 q_num_heads,
        #                 self.kv_lora_rank,
        #                 dtype=q.dtype,
        #                 device=q.device)
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=q.device)
        # num_kv_splits = 4  # TODO: heuristic

        # TODO(lucas) Allocate ahead of time
        # attn_logits = torch.empty(
        #     (
        #         B,
        #         q_num_heads,
        #         num_kv_splits,
        #         # NOTE(lucas) idk why the +1 is here but sglang has it so we
        #         # just mirror that
        #         self.kv_lora_rank + 1,
        #     ),
        #     dtype=torch.float32,
        #     device=q.device,
        # )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        # kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # # Run MQA
        # decode_attention_fwd(q, kv_c_and_k_pe_cache, kv_c_cache, o, lse,
        #                      attn_metadata.decode.block_table,
        #                      attn_metadata.decode.seq_lens, attn_logits,
        #                      num_kv_splits, self.scale, PAGE_SIZE)
        ### NOTE(lms): check correctness
        # flash_mla uses sm_scale = 1/sqrt(head_dim) internally, but MLA needs
        # self.scale = 1/sqrt(qk_nope_head_dim + qk_rope_head_dim).
        # Pre-scale q to correct: q * self.scale * sqrt(head_dim) makes
        # the effective scale = self.scale after the kernel applies 1/sqrt(head_dim).
        import math
        scale_correction = self.scale * math.sqrt(head_dim)
        q_scaled = q * scale_correction

        # flash_mla expects q shape [B, s_q, num_heads, d]
        q_4d = q_scaled.unsqueeze(1)
        o = flash_mla(
            q_4d,
            attn_metadata.decode.block_table,
            kv_c_and_k_pe_cache,
            None,
            PAGE_SIZE,
            B,
            1,
            attn_metadata.decode.seq_lens,
            q_num_heads,
            None,
            head_dim,
            head_dim_v,
            True,
        )
        # flash_mla returns [B, s_q=1, num_heads, dv], squeeze s_q dim
        o = o.squeeze(1)
        return o, lse
