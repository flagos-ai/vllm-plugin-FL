# Copyright (c) 2026 BAAI. All rights reserved.

"""HY V4's 576-wide, sink-aware sparse MLA on real FlagGems kernels."""

from vllm.models.hy_v4.nvidia.flashmla_sparse import (
    HYV4FlashMLASparseBackend,
    HYV4FlashMLASparseImpl,
)


class NvidiaHYV4FlashMLASparseImpl(HYV4FlashMLASparseImpl):
    # HY's sink applies to every token, so upstream already forces sparse MQA.
    # Do not instantiate an unused dense FlashAttention backend in EMPTY.
    supports_dense_mha_prefill = False


class NvidiaHYV4FlashMLASparseBackend(HYV4FlashMLASparseBackend):
    @staticmethod
    def get_impl_cls():
        return NvidiaHYV4FlashMLASparseImpl


def flash_mla_sparse_fwd(
    q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None, out=None
):
    from flag_gems.fused.flashmla_sparse import flash_mla_sparse_fwd as gems_prefill

    if q.shape[-1] != 576 or kv.shape[-1] != 576 or d_v != 512:
        raise NotImplementedError(
            "HY V4 sparse MLA requires Q/K dimension 576 and V 512"
        )
    result, max_logits, lse = gems_prefill(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length
    )
    if out is not None:
        if (
            out.shape != result.shape
            or out.dtype != result.dtype
            or out.device != result.device
        ):
            raise ValueError("HY V4 MLA output buffer must match the result")
        out.copy_(result)
        result = out
    return result, max_logits, lse
