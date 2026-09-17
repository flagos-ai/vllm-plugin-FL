# Copyright (c) 2026 BAAI. All rights reserved.

"""NVIDIA platform integration for vLLM 0.28."""

import os
from typing import TYPE_CHECKING

from vllm.platforms import PlatformEnum
from vllm.platforms.cuda import CudaPlatform

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class NvidiaPlatformFL(CudaPlatform):
    """Add FL runtime hooks without changing native CUDA platform semantics."""

    # NVIDIA is an in-tree CUDA platform in vLLM. Marking it OOT makes v0.28
    # attention and MoE code enter generic accelerator branches and corrupts
    # MLA decode. FL is an operator/runtime plugin on top of native CUDA.
    _enum = PlatformEnum.CUDA
    vendor_name = "nvidia"
    dist_backend = (
        "flagcx" if "FLAGCX_PATH" in os.environ else CudaPlatform.dist_backend
    )

    @classmethod
    def _refresh_dist_backend(cls) -> bool:
        use_flagcx = "FLAGCX_PATH" in os.environ
        cls.dist_backend = "flagcx" if use_flagcx else CudaPlatform.dist_backend
        return use_flagcx

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        use_flagcx = cls._refresh_dist_backend()
        super().check_and_update_config(vllm_config)
        if use_flagcx:
            vllm_config.parallel_config.disable_custom_all_reduce = True
        vllm_config.parallel_config.worker_cls = "vllm_fl.worker.worker.NvidiaWorkerFL"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        if cls._refresh_dist_backend():
            return (
                "vllm_fl.distributed.nvidia_communicator.NvidiaCommunicatorFL"
            )
        return super().get_device_communicator_cls()

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        if cls._refresh_dist_backend():
            return False
        return super().use_custom_allreduce()

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        attn_selector_config,
        num_heads: int | None = None,
    ) -> str:
        """Keep native CUDA selection unless FlagGems attention is requested."""
        use_flaggems_attn = os.environ.get(
            "VLLM_FL_USE_FLAGGEMS_ATTN", "0"
        ).lower() in ("1", "true", "yes")
        if use_flaggems_attn:
            from vllm_fl.dispatch.backends.flaggems.flaggems import (
                FlagGemsBackend,
            )

            return FlagGemsBackend().attention_backend(
                use_mla=attn_selector_config.use_mla,
                use_sparse=attn_selector_config.use_sparse,
            )

        return super().get_attn_backend_cls(
            selected_backend,
            attn_selector_config,
            num_heads,
        )
