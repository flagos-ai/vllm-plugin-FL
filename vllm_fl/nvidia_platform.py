# Copyright (c) 2026 BAAI. All rights reserved.

"""NVIDIA platform integration for vLLM 0.28."""

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

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        super().check_and_update_config(vllm_config)
        vllm_config.parallel_config.worker_cls = (
            "vllm_fl.worker.worker.NvidiaWorkerFL"
        )
