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
    from vllm.platforms import current_platform

    global _patches_applied
    if current_platform.device_type != "ptpu" or _patches_applied:
        return

    patch_accelerator_memory_stats()
    patch_accelerator_memory_api()
    patch_graph_capture_runtime()
    patch_flagcx_stream_adapter()
    patch_distributed_runtime()
    patch_op_cls()
    _patches_applied = True


def patch_graph_capture_runtime():
    from .impl.graph_capture import patch_graph_capture_runtime as patch_runtime

    patch_runtime()


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
        from vllm.distributed.device_communicators.base_device_communicator import (
            DeviceCommunicatorBase,
        )
        from vllm.platforms import current_platform

        from vllm_fl.distributed.communicator import CommunicatorFL

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

        # WorkerFL normalizes flagcx/nccl process groups to pccl for PTPU.
        # Do not import or wrap worker here: worker -> custom_ops -> this module
        # would create a circular import during backend registration.

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

        logger.info("Configured Sunrise/PTPU to use FlagCX communicator with pccl PGs")
    except Exception as e:
        logger.warning("Failed to configure Sunrise distributed runtime: %s", e)


def patch_op_cls():
    """Register Sunrise replacements for upstream custom ops."""
    try:
        from vllm.model_executor.custom_op import PluggableLayer

        from .impl.vocab_parallel_embedding import SunriseVocabParallelEmbedding

        PluggableLayer.register_oot(
            _decorated_layer_cls=SunriseVocabParallelEmbedding,
            name="VocabParallelEmbedding",
        )
        logger.info("Patched VocabParallelEmbedding for Sunrise/PTPU")
    except Exception as e:
        logger.warning("Failed to patch VocabParallelEmbedding for Sunrise: %s", e)


def patch_accelerator_memory_api():
    """Route vLLM 0.28 memory profiling through the PTPU runtime.

    The generic accelerator allocator interface is unavailable in torch_ptpu.
    vLLM 0.28 uses it for both its initial snapshot and KV-cache profiling.
    """
    from vllm.platforms import current_platform

    if current_platform.device_type != "ptpu":
        return

    try:
        import torch.accelerator as _accel

        if getattr(_accel, "_sunrise_memory_api_patched", False):
            return
        api_names = {
            "empty_cache": "empty_cache",
            "memory_stats": "memory_stats",
            "memory_allocated": "memory_allocated",
            "memory_reserved": "memory_reserved",
            "max_memory_allocated": "max_memory_allocated",
            "reset_peak_memory_stats": "reset_peak_memory_stats",
            "get_memory_info": "mem_get_info",
        }
        memory_api = {
            name: getattr(torch.ptpu, ptpu_name)
            for name, ptpu_name in api_names.items()
        }
        for name, fn in memory_api.items():
            setattr(_accel, name, fn)
        _accel._sunrise_memory_api_patched = True
        logger.info("Patched torch.accelerator memory APIs for Sunrise/PTPU")
    except Exception as e:
        logger.warning("Failed to patch torch.accelerator memory APIs: %s", e)


def patch_accelerator_memory_stats():
    """Provide the memory counter keys used by vLLM 0.28 snapshots.

    torch.ptpu exposes max_memory_allocated/memory_reserved but not the
    torch.cuda-style memory_stats() dict that the FL worker's KV-cache sizing
    reads. Include the current allocation so v0.28 does not count it again
    as transient peak headroom. These counters retain the runtime's own
    tracking limitations; never override a native stats implementation.
    """
    from vllm.platforms import current_platform

    if current_platform.device_type != "ptpu":
        return

    try:
        if getattr(torch.ptpu, "_sunrise_memory_stats_patched", False):
            return
        if hasattr(torch.ptpu, "memory_stats"):
            return

        def _memory_stats(device=None):
            allocated = torch.ptpu.memory_allocated(device)
            peak = max(allocated, torch.ptpu.max_memory_allocated(device))
            reserved = torch.ptpu.memory_reserved(device)
            return {
                "allocated_bytes.all.peak": peak,
                "allocated_bytes.all.current": allocated,
                "reserved_bytes.all.peak": reserved,
                "reserved_bytes.all.current": reserved,
            }

        torch.ptpu.memory_stats = _memory_stats
        torch.ptpu._sunrise_memory_stats_patched = True
        logger.info("Patched torch.ptpu.memory_stats for Sunrise/PTPU")
    except Exception as e:
        logger.warning("Failed to patch torch.ptpu.memory_stats: %s", e)
