# Copyright (c) 2026 BAAI. All rights reserved.

"""Keep HY V4 on its upstream sparse-only MLA path in EMPTY builds."""

from vllm.models.hy_v4.nvidia.flashmla_sparse import (
    HYV4FlashMLASparseBackend,
    HYV4FlashMLASparseImpl,
)


class NvidiaHYV4FlashMLASparseImpl(HYV4FlashMLASparseImpl):
    # HY forces sparse MQA, but upstream still constructs a dense FA backend.
    # EMPTY leaves its callable absent while CUDA availability reports true.
    supports_dense_mha_prefill = False


class NvidiaHYV4FlashMLASparseBackend(HYV4FlashMLASparseBackend):
    @staticmethod
    def get_impl_cls():
        return NvidiaHYV4FlashMLASparseImpl
