# Copyright (c) 2025 BAAI. All rights reserved.
"""ARM CPU platform for the FL quantized linear integrations.

vllm-plugin-FL has no CPU vendor/backend (its PlatformFL/WorkerFL are
GPU-shaped, adapted from vLLM v0.20.2 CUDA). On ARM CPU we register this
subclass so the plugin provides a CPU backend while inheriting vLLM's native
CpuPlatform (torch.compile, CPUWorker).

Compilation mode, OpenMP policy, and affinity belong to the launcher profile.
The native CPU platform requires its MP executor to configure OpenMP correctly. An explicit
FL_CPU_UNIPROC=1 escape hatch keeps the measured in-process path available for controlled,
single-worker deployments that configure thread affinity and allocator preload themselves.

The selected W4A8 or W8 backend is installed via register_model(), which runs
in whichever process loads the model.
"""
import os

from vllm.logger import init_logger
from vllm.platforms.cpu import CpuPlatform

logger = init_logger(__name__)


class CpuPlatformFL(CpuPlatform):
    """Native vLLM CPU platform with compile enabled for ARM quantization."""

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        from vllm_fl.patches.dynamo_metrics import (
            patch_dynamo_metrics_serialization,
        )

        patch_dynamo_metrics_serialization()

        pc = vllm_config.parallel_config
        uniproc_requested = (
            os.environ.get("FL_CPU_UNIPROC", "0") == "1"
            and pc.world_size == 1
        )

        super().check_and_update_config(vllm_config)

        # Opt-in only: vLLM's CPU platform normally requires MP to configure OMP.
        if (
            uniproc_requested
            and pc.distributed_executor_backend == "mp"
        ):
            pc.distributed_executor_backend = "uni"
            logger.warning(
                "[vllm_fl] FL_CPU_UNIPROC=1: using unsupported-by-vLLM "
                "in-process CPU executor; caller must configure OpenMP"
            )

        logger.info("[vllm_fl] FL ARM CPU platform active (native-backed)")

    @classmethod
    def update_block_size_for_backend(cls, vllm_config) -> None:
        """Align hybrid attention pages without patching vLLM's CPU class."""
        model_config = vllm_config.model_config
        if not model_config or not model_config.is_hybrid:
            return
        backend_cls = cls._find_non_ssm_backend(vllm_config)
        if backend_cls is not None:
            cls._align_hybrid_block_size(vllm_config, backend_cls)
