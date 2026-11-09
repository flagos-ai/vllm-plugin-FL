# Copyright (c) 2026 BAAI. All rights reserved.

"""FL runtime integration with the native vLLM 0.32 CUDA platform."""

import os
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.platforms import PlatformEnum
from vllm.platforms.cuda import CudaPlatform

if TYPE_CHECKING:
    from vllm.config import VllmConfig


logger = init_logger(__name__)


class NvidiaPlatformFL(CudaPlatform):
    """Keep upstream CUDA kernels, model runners and backend selection."""

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
            passes = vllm_config.compilation_config.pass_config
            if passes.fuse_allreduce_rms:
                # FlashInfer's fused collective bypasses the selected FlagCX
                # communicator. Keep the ordinary all-reduce/RMSNorm sequence.
                passes.fuse_allreduce_rms = False
                logger.warning_once(
                    "Disabling fuse_allreduce_rms to preserve the selected "
                    "FlagCX communication backend."
                )
        vllm_config.parallel_config.worker_cls = (
            "vllm_fl.worker.nvidia_worker.NvidiaWorkerFL"
        )

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        if cls._refresh_dist_backend():
            return "vllm_fl.distributed.nvidia_communicator.NvidiaCommunicatorFL"
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
        """Use native CUDA selection unless FlagGems attention is requested."""
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
