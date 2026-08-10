# SPDX-License-Identifier: Apache-2.0
"""MetaX implementation of the vLLM sparse FlashMLA backend."""

from typing import Any, ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import SparseMLAAttentionImpl
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
)

from ..ops.flashmla import flash_mla_sparse_fwd


class MacaFlashMLASparseBackend(FlashMLASparseBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]

    @staticmethod
    def get_impl_cls() -> type[SparseMLAAttentionImpl[Any]]:
        return MacaFlashMLASparseImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class MacaFlashMLASparseImpl(FlashMLASparseImpl):
    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        if self.num_heads % self.prefill_padding != 0:
            assert self.prefill_padding % self.num_heads == 0
            padded_q = q.new_empty(
                (num_tokens, self.prefill_padding, q.shape[2])
            )
            padded_q[:, : self.num_heads] = q
            q = padded_q

        output = flash_mla_sparse_fwd(
            q,
            kv_cache.view(-1, 1, kv_cache.shape[-1]),
            topk_indices.view(num_tokens, 1, -1),
            self.softmax_scale,
            topk_length=topk_length,
        )[0]
        return output[:, : self.num_heads]
