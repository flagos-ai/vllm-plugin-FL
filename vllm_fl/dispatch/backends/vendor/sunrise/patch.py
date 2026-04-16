# Copyright (c) 2026 BAAI. All rights reserved.

import logging

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_sunrise_patches():
    """Apply Sunrise/PTPU patches that must run before model construction."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_distributed_runtime()
    patch_op_cls()


def patch_distributed_runtime():
    """Use torch.distributed pccl groups for Sunrise/PTPU."""
    try:
        from vllm.platforms import current_platform

        platform_cls = (
            current_platform
            if isinstance(current_platform, type)
            else current_platform.__class__
        )

        platform_cls.dist_backend = "pccl"
        current_platform.dist_backend = "pccl"
        platform_cls.get_device_communicator_cls = classmethod(
            lambda cls: (
                "vllm.distributed.device_communicators."
                "base_device_communicator.DeviceCommunicatorBase"
            )
        )
        platform_cls.use_custom_allreduce = classmethod(lambda cls: False)
        logger.info(
            "Configured Sunrise/PTPU to use pccl with base device communicator"
        )
    except Exception as e:
        logger.warning("Failed to configure Sunrise distributed runtime: %s", e)


def patch_op_cls():
    """Register Sunrise replacements for upstream custom ops."""
    try:
        from vllm.model_executor.custom_op import CustomOp

        from .impl.vocab_parallel_embedding import SunriseVocabParallelEmbedding

        CustomOp.register_oot(
            _decorated_op_cls=SunriseVocabParallelEmbedding,
            name="VocabParallelEmbedding",
        )
        logger.info("Patched VocabParallelEmbedding for Sunrise/PTPU")
    except Exception as e:
        logger.warning("Failed to patch VocabParallelEmbedding for Sunrise: %s", e)
