# Copyright (c) 2026 BAAI. All rights reserved.

import ctypes
import importlib
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_sunrise_patches():
    """Apply Sunrise/PTPU patches that must run before model construction."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_flagcx_stream_adapter()
    patch_distributed_runtime()
    patch_op_cls()


def patch_flagcx_stream_adapter():
    """Patch FlagCX wrapper to accept PTPU streams."""
    try:
        from vllm.platforms import current_platform

        if getattr(current_platform, "device_type", None) != "ptpu":
            return

        flagcx_path = os.getenv("FLAGCX_PATH")
        if flagcx_path and os.path.isdir(flagcx_path) and flagcx_path not in sys.path:
            sys.path.append(flagcx_path)

        flagcx_wrapper = importlib.import_module("plugin.interservice.flagcx_wrapper")
        FLAGCXLibrary = flagcx_wrapper.FLAGCXLibrary
        flagcxStream_t = flagcx_wrapper.flagcxStream_t

        if getattr(FLAGCXLibrary, "_sunrise_stream_patch_applied", False):
            return

        def _to_void_p(raw_stream_ptr):
            if isinstance(raw_stream_ptr, ctypes.c_void_p):
                return raw_stream_ptr
            if raw_stream_ptr is None:
                raise ValueError("Stream pointer is None.")
            return ctypes.c_void_p(int(raw_stream_ptr))

        def _extract_raw_stream_ptr(old_stream):
            if isinstance(old_stream, (int, ctypes.c_void_p)):
                return old_stream

            for attr in ("ptpu_stream", "cuda_stream"):
                stream_ptr = getattr(old_stream, attr, None)
                if stream_ptr is not None:
                    return stream_ptr

            stream_fn = getattr(old_stream, "stream", None)
            if callable(stream_fn):
                stream_ptr = stream_fn()
                if stream_ptr is not None:
                    return stream_ptr

            raise AttributeError(
                "Unsupported stream object: expected a raw pointer or one of "
                "`ptpu_stream`, `cuda_stream`, or callable `stream()`."
            )

        def _adaptor_stream_copy(self, old_stream):
            new_stream = flagcxStream_t()
            raw_stream_ptr = _extract_raw_stream_ptr(old_stream)
            self.FLAGCX_CHECK(
                self.handler.contents.devHandle.contents.streamCopy(
                    ctypes.byref(new_stream), _to_void_p(raw_stream_ptr)
                )
            )
            return new_stream

        FLAGCXLibrary.adaptor_stream_copy = _adaptor_stream_copy
        FLAGCXLibrary._sunrise_stream_patch_applied = True
        logger.info("Patched FlagCX stream adapter for Sunrise/PTPU")
    except Exception as e:
        logger.warning("Failed to patch FlagCX stream adapter for Sunrise: %s", e)


def patch_distributed_runtime():
    """Keep FlagCX path while mapping torch ProcessGroup backend to pccl."""
    try:
        from vllm.platforms import current_platform
        from vllm.distributed.device_communicators.base_device_communicator import (
            DeviceCommunicatorBase,
        )
        from vllm_fl.distributed.communicator import CommunicatorFL
        from vllm_fl.worker import worker as worker_mod

        platform_cls = (
            current_platform
            if isinstance(current_platform, type)
            else current_platform.__class__
        )

        if getattr(current_platform, "device_type", None) != "ptpu":
            return

        # Preserve the original Sunrise/FlagCX communicator selection logic.
        platform_cls.dist_backend = "flagcx"
        current_platform.dist_backend = "flagcx"

        if not getattr(CommunicatorFL, "_sunrise_all_gather_patched", False):
            def _all_gather(self, input_: torch.Tensor, dim: int = -1):
                world_size = self.world_size
                if world_size == 1:
                    return input_

                assert -input_.dim() <= dim < input_.dim(), (
                    f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
                )
                if dim < 0:
                    dim += input_.dim()

                pyflagcx_comm = getattr(self, "pyflagcx_comm", None)
                if pyflagcx_comm is None or pyflagcx_comm.disabled:
                    return DeviceCommunicatorBase.all_gather(self, input_, dim)

                output_tensor = self.all_gatherv(input_, dim=0, sizes=None)
                if dim == 0:
                    return output_tensor

                input_size = input_.size()
                output_tensor = output_tensor.reshape((world_size,) + input_size)
                output_tensor = output_tensor.movedim(0, dim)
                output_tensor = output_tensor.reshape(
                    input_size[:dim]
                    + (world_size * input_size[dim],)
                    + input_size[dim + 1 :]
                )
                return output_tensor

            CommunicatorFL.all_gather = _all_gather
            CommunicatorFL._sunrise_all_gather_patched = True

        init_dist = worker_mod.init_worker_distributed_environment
        if not getattr(init_dist, "_sunrise_backend_patched", False):
            def _init_worker_distributed_environment(
                vllm_config,
                rank,
                distributed_init_method=None,
                local_rank=-1,
                backend="nccl",
            ):
                backend_for_pg = backend
                if backend in ("flagcx", "nccl"):
                    backend_for_pg = "pccl"
                return init_dist(
                    vllm_config,
                    rank,
                    distributed_init_method=distributed_init_method,
                    local_rank=local_rank,
                    backend=backend_for_pg,
                )

            _init_worker_distributed_environment._sunrise_backend_patched = True
            worker_mod.init_worker_distributed_environment = (
                _init_worker_distributed_environment
            )

        logger.info(
            "Configured Sunrise/PTPU to use FlagCX communicator with pccl PGs"
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
