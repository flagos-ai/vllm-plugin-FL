# SPDX-License-Identifier: Apache-2.0
"""T-Head PPU sparse MLA backend.

The selected attention backend owns query concatenation and KV-cache writes
as well as attention. All three use PPU native kernels; helper operations do
not perform a second, independently configurable backend dispatch.
"""

from typing import TYPE_CHECKING

import torch
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
)

from .native_extensions import concat_mla_q, mla_kv_cache_update

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


class TheadMLASparseImpl(FlashMLASparseImpl):
    """Sparse MLA control flow backed by the PPU native attention kernel."""

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
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        sinks: torch.Tensor | None = mla_args.pop("sinks", None)
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
            topk_indices_buffer=topk_indices_buffer,
            indexer=indexer,
            **mla_args,
        )
        self._validate_sinks(sinks, num_heads)
        self.sinks = sinks

    @staticmethod
    def _validate_sinks(sinks: torch.Tensor | None, num_heads: int) -> None:
        if sinks is None:
            return
        if sinks.dtype != torch.float32:
            raise ValueError(
                "T-Head Sparse MLA sinks must have dtype torch.float32, "
                f"but got {sinks.dtype}."
            )
        if sinks.ndim != 1 or sinks.shape[0] != num_heads:
            raise ValueError(
                "T-Head Sparse MLA sinks must have shape "
                f"({num_heads},), but got {tuple(sinks.shape)}."
            )

    def do_kv_cache_update(
        self,
        kv_c_normed,
        k_pe,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        mla_kv_cache_update(
            kv_c_normed,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            concat_mla_q(ql_nope, q_pe, q)
        return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Invoke the PPU native sparse MLA kernel without head padding."""
        from flash_mla import flash_mla_sparse_fwd

        if q.dtype != torch.bfloat16 or kv_c_and_k_pe_cache.dtype != torch.bfloat16:
            raise TypeError("T-Head sparse MLA requires BF16 query and KV cache.")
        if q.shape[-1] != 576 or kv_c_and_k_pe_cache.shape[-1] != 576:
            raise ValueError("T-Head sparse MLA requires head_size=576.")
        if self.sinks is not None and self.sinks.device != q.device:
            raise ValueError("T-Head sparse MLA sinks and query must share a device.")

        num_tokens = q.shape[0]
        output, _, _ = flash_mla_sparse_fwd(
            q=q,
            kv=kv_c_and_k_pe_cache.view(-1, 1, 576),
            indices=topk_indices.view(num_tokens, 1, -1),
            sm_scale=self.softmax_scale,
            d_v=self.kv_lora_rank,
            attn_sink=self.sinks,
            topk_length=topk_length,
        )
        return output


class TheadMLASparseBackend(FlashMLASparseBackend):
    supported_kv_cache_dtypes = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "THEAD_FLASHMLA_SPARSE"

    @staticmethod
    def get_impl_cls():
        return TheadMLASparseImpl

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8
