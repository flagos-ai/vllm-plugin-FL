# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route the BF16-cache FlashMLA call while retaining the native workspace."""

from vllm.models.hy_v4.nvidia.flashmla_sparse import HYV4FlashMLASparseImpl
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_fl.ops.hy4_flash_kernels import sparse_mla


class HYV4FlashSparseImpl(HYV4FlashMLASparseImpl):
    def _bf16_flash_mla_kernel(
        self,
        q,
        kv_c_and_k_pe_cache,
        topk_indices,
        topk_length=None,
        actual_num_heads=None,
        out=None,
    ):
        tokens = q.shape[0]
        kv = kv_c_and_k_pe_cache.view(-1, 1, kv_c_and_k_pe_cache.shape[-1])
        real_heads = q.shape[1] if actual_num_heads is None else actual_num_heads
        padding = self.prefill_padding
        padded_heads = (real_heads + padding - 1) // padding * padding
        sink = self._sinks_for_query(q[:, :real_heads], 1, padded_heads)
        if q.shape[1] < padded_heads:
            if out is None:
                padded = q.new_zeros((tokens, padded_heads, q.shape[2]))
            else:
                *_, padded, _, _ = current_workspace_manager().get_simultaneous(
                    *self.workspace_specs
                )
                padded = padded[:tokens]
                padded[:, real_heads:].zero_()
            padded[:, :real_heads].copy_(q)
            q = padded
        result, _, lse = sparse_mla(
            q,
            kv,
            topk_indices.view(tokens, 1, -1),
            self.softmax_scale,
            attn_sink=sink,
            topk_length=topk_length,
            out=out,
        )
        return result[:, :real_heads], lse[:, :real_heads]
