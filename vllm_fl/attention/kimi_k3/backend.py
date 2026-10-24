"""Reuse vLLM MLA metadata with verified open decode and FP32 prefill kernels."""

from vllm_fl.dispatch import CachedOp
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.attention.prefill_mma import can_use

run = CachedOp("kimi_k3_prefill_mma")
mla_tp8_run = CachedOp("kimi_k3_mla_tp8")
attention = CachedOp("kimi_k3_prefill_attention")


import torch
from vllm_fl.ops.kimi_k3.graph_context import in_full_graph
from flaggems_vllm.ops.flash_mla_with_kvcache import FlashMLASchedMeta

flash_mla_with_kvcache = CachedOp("kimi_k3_mla_decode")
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend, TritonMLAImpl
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
import vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.attention.mla_tp8 as mla_tp8


class K3MLAImpl(TritonMLAImpl):
    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        if not attn_metadata.causal or self.dcp_world_size > 1:
            raise NotImplementedError("This candidate supports causal K3 and DCP1")
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        metadata = attn_metadata.decode
        if metadata is None:
            raise ValueError("Decode metadata is required")
        query = q[:, None].contiguous()
        cache = kv_c_and_k_pe_cache.unsqueeze(2)
        if mla_tp8.can_use(query, cache, metadata.block_table, metadata.seq_lens):
            # Tune captured decode only; mixed/eager calls retain the old kernels.
            # K3's capture wrapper uses inner NONE plus this existing FULL hint.
            full_graph = in_full_graph()
            output, lse = mla_tp8_run(
                query,
                cache,
                metadata.block_table,
                metadata.seq_lens,
                self.scale,
                qk_tile=(128 if query.shape[0] in (12, 16, 24, 32) else None)
                if full_graph
                else 64,
                qk_warps=(4 if query.shape[0] in (12, 16, 24, 32) else None)
                if full_graph
                else 2,
                pv_warps=2 if full_graph and query.shape[0] in (12, 16, 24, 32) else 4,
                short_splits=(16 if query.shape[0] in (24, 32) else 32)
                if full_graph and query.shape[0] in (24, 32, 64)
                else None,
                short_limit=None
                if full_graph
                else (1024 if query.shape[0] == 4 else 2048),
            )
            return output[:, 0], lse[:, :, 0]
        output, lse = flash_mla_with_kvcache(
            q[:, None].contiguous(),
            kv_c_and_k_pe_cache.unsqueeze(2),
            metadata.block_table,
            metadata.seq_lens,
            self.kv_lora_rank,
            FlashMLASchedMeta(),
            softmax_scale=self.scale,
            causal=False,
        )
        return output[:, 0], lse[:, :, 0]


class K3MLABackend(TritonMLABackend):
    supported_dtypes = [torch.bfloat16]
    supported_kv_cache_dtypes = ["auto", "bfloat16"]

    @classmethod
    def supports_non_causal(cls):
        return False

    @staticmethod
    def get_name():
        return "K3_OPEN_MLA"

    @staticmethod
    def get_impl_cls():
        return K3MLAImpl


class K3PrefillBackend(MLAPrefillBackend):
    supports_missing_device_capability = True

    @staticmethod
    def get_name():
        return "K3_FP32_PREFILL"

    def supports_out(self):
        return True

    def _run(self, q, k, v, cq, ck, max_q, causal, out):
        qc, kc, vc = q.contiguous(), k.contiguous(), v.contiguous()
        if (
            causal
            and cq is ck
            and q.shape[1] == 12
            and can_use(qc, kc, vc, cq, ck, max_q, self.scale)
        ):
            shape = (q.shape[0], 12, 128)
            if out is None:
                target = torch.empty(shape, dtype=q.dtype, device=q.device)
                direct = True
            else:
                target = out
                out_storage = out.untyped_storage().data_ptr()
                direct = (
                    tuple(out.shape) == shape
                    and out.is_contiguous()
                    and out.device == q.device
                    and out.dtype in (torch.bfloat16, torch.float32)
                    and all(
                        out_storage != t.untyped_storage().data_ptr()
                        for t in (qc, kc, vc, cq)
                    )
                )
            if direct:
                return run(
                    qc,
                    kc,
                    vc,
                    cq,
                    ck,
                    max_q,
                    self.scale,
                    True,
                    out=target,
                    head_major_lse=True,
                )
        result, lse = attention(
            qc,
            kc,
            vc,
            cq,
            ck,
            max_q,
            self.scale,
            causal=causal,
        )
        if out is None:
            result = result.to(q.dtype)
        else:
            out.copy_(result)
            result = out
        return result, lse.transpose(0, 1).contiguous()

    def run_prefill_new_tokens(
        self, q, k, v, return_softmax_lse, out=None, output_scale=None
    ):
        if output_scale is not None:
            raise NotImplementedError("Quantized prefill output is not enabled")
        metadata = self._prefill_metadata
        result, lse = self._run(
            q,
            k,
            v,
            metadata.query_start_loc,
            metadata.query_start_loc,
            metadata.max_query_len,
            True,
            out,
        )
        return (result, lse) if return_softmax_lse else result

    def run_prefill_context_chunk(self, chunk, q, k, v, out=None):
        return self._run(
            q,
            k,
            v,
            chunk.query_start_loc,
            chunk.cu_seq_lens,
            chunk.max_query_len,
            False,
            out,
        )


def attention_backend(use_mla=False, use_sparse=False):
    """Select K3 MLA; preserve the existing library path for other attention."""
    if use_mla:
        if not torch.cuda.is_available():
            raise RuntimeError("The open attention backend requires an available accelerator.")
        return "vllm_fl.attention.kimi_k3.backend.K3MLABackend"
    return FlagGemsBackend().attention_backend(use_mla, use_sparse)
