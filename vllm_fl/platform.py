# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/platforms/cuda.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from datetime import timedelta
from functools import cache, wraps
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, Union
from typing_extensions import ParamSpec

import torch

from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.logger import init_logger

from vllm.platforms import Platform, PlatformEnum
from vllm.platforms.interface import DeviceCapability

if TYPE_CHECKING:
    from vllm.attention.selector import AttentionSelectorConfig
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
else:
    VllmConfig = None
    CacheDType = None

from vllm_fl.utils import DeviceInfo
import vllm_fl.envs as fl_envs

logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

@cache
def _get_backend(
    use_mla: bool,
    device_info: Optional[DeviceInfo] = None,
) -> list[str]:
    """Get backend priorities with lazy import to avoid circular dependency."""
    if use_mla:
        raise NotImplementedError("NOT support mla now!")
    else:
        if fl_envs.USE_FLAGGEMS:
            return [AttentionBackendEnum.TRITON_ATTN]
        return [AttentionBackendEnum.FLASH_ATTN] 
        

class PlatformFL(Platform):
    _enum = PlatformEnum.OOT
    device_info = DeviceInfo()
    device_name = device_info.device_type 
    device_type = device_info.device_type 
    dispatch_key = device_info.dispatch_key
    torch_device_fn = device_info.torch_device_fn
    ray_device_key: str = "flagos"
    dist_backend: str = "flagcx" if "FLAGCX_PATH" in os.environ else "nccl"
    ### TODO(lms): dispatch device_control_env_var
    # device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    def is_cuda_alike(self) -> bool:
        """Stateless version of [torch.cuda.is_available][]."""
        return self.device_type == "cuda"
    
    def is_cuda(self) -> bool:
        """Stateless version of [torch.cuda.is_available][]."""
        return self.device_type == "cuda"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def check_if_supports_dtype(cls, torch_dtype: torch.dtype):
        """
        Check if the dtype is supported by the current platform.
        """
        pass

    @classmethod
    def get_current_memory_usage(cls,
                                 device: Optional[torch.types.Device] = None
                                 ) -> float:
        cls.torch_device_fn.empty_cache()
        cls.torch_device_fn.reset_peak_memory_stats(device)
        return cls.torch_device_fn.max_memory_allocated(device)

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        cls.torch_device_fn.set_device(device)
    
    @classmethod
    def empty_cache(cls) -> None:
        cls.torch_device_fn.empty_cache()

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return cls.device_name
        
    ### TODO(lms): change pin_memory depend device
    @classmethod
    def is_pin_memory_available(cls):
        if cls.device_type in ["cuda", "xpu", "npu"]:
            return True
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config

        # Use vLLM's native GPU worker for v0.15.0 compatibility
        # Custom ops are registered via register_ops() entry point
        parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            # Ascend NPU requires block_size to be a multiple of 128
            # CUDA can use smaller block sizes like 16
            if cls.device_type == "npu":
                cache_config.block_size = 128
                logger.info("Setting kv cache block size to 128 for Ascend NPU.")
            else:
                cache_config.block_size = 16

        # TODO(lucas): handle this more gracefully
        # Note: model_config may be None during testing
        # Note: block_size is initialized in
        # HybridAttentionMambaModelConfig.verify_and_update_config
        # for models with both attention and mamba,
        # and doesn't need to be reinitialized here
        if (
            model_config is not None
            and model_config.use_mla
            and cache_config.block_size is not None
        ):
            if cache_config.block_size % 64 != 0:
                cache_config.block_size = 64
                logger.info("Forcing kv cache block size to 64 for FlagOSMLA backend.")

        # lazy import to avoid circular import
        from vllm.config import CUDAGraphMode

        compilation_config = vllm_config.compilation_config
        if compilation_config.compile_sizes is None:
            compilation_config.compile_sizes = []

        if (parallel_config.data_parallel_size > 1
            and compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            # TODO: Piecewise Cuda graph might be enabled
            # if torch compile cache key issue fixed
            # See https://github.com/vllm-project/vllm/pull/25093
            logger.info(
                "WideEP: Disabling CUDA Graphs since DeepEP high-throughput "
                "kernels are optimized for prefill and are incompatible with "
                "CUDA Graphs. "
                "In order to use CUDA Graphs for decode-optimized workloads, "
                "use --all2all-backend with another option, such as "
                "deepep_low_latency, pplx, or allgather_reducescatter."
            )
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
    ) -> list[str]:
        """Get the attention backend class path using the dispatch mechanism."""
        from vllm_fl.dispatch import call_op

        use_mla = attn_selector_config.use_mla

        try:
            backend_path = call_op("attention_backend", use_mla=use_mla)

            logger.info_once(
                "Using attention backend via dispatch (use_mla=%s): %s",
                use_mla, backend_path,
                scope="local",
            )
            return backend_path

        except RuntimeError as e:
            # Fallback: if dispatch fails, use device-type based selection
            logger.warning(
                "Dispatch mechanism failed for attention_backend, "
                "falling back to device-type based selection: %s", e
            )

            if cls.device_type == "npu":
                if use_mla:
                    backend_path = "vllm_fl.dispatch.backends.flaggems.impl.mla.MLAFLBackend"
                else:
                    backend_path = "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"
            else:
                # For CUDA and other devices, use vLLM native backend
                from vllm.v1.attention.backends.registry import AttentionBackendEnum
                if use_mla:
                    # MLA renamed to TRITON_MLA in vLLM 0.15.0
                    backend_path = AttentionBackendEnum.TRITON_MLA.get_path()
                else:
                    backend_path = AttentionBackendEnum.FLASH_ATTN.get_path()

            logger.info_once(
                "Using fallback attention backend (use_mla=%s): %s",
                use_mla, backend_path,
                scope="local",
            )
            return backend_path

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        return [
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.FLASH_ATTN,
        ]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: Optional["AttentionBackendEnum"] = None,
    ) -> list[str]:
        if backend is not None:
            assert backend in cls.get_supported_vit_attn_backends(), (
                f"Backend {backend} is not supported for vit attention. "
                f"Supported backends are: {cls.get_supported_vit_attn_backends()}"
            )
            logger.info_once(f"Using backend {backend} for vit attention")
            return backend

        # Try FlashAttention first
        if (cc := cls.get_device_capability()) and cc.major >= 8:
            try:
                backend_class = AttentionBackendEnum.FLASH_ATTN.get_class()
                if backend_class.supports_head_size(
                    head_size
                ) and backend_class.supports_dtype(dtype):
                    return AttentionBackendEnum.FLASH_ATTN
            except ImportError:
                pass

        return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def get_punica_wrapper(cls) -> str:
        # TODO(lms): support fl PunicaWrapper
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        if cls.dist_backend == "flagcx":
            logger.info("Using CommunicatorFL for communication.")
            return (
                "vllm_fl.distributed.communicator.CommunicatorFL"  # noqa
            )
        else:
            logger.info("Using CudaCommunicator for communication.")
            return (
                "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator" # noqa
            )

    
    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm_fl.compilation.graph.GraphWrapper"
    
    @classmethod
    def support_static_graph_mode(cls) -> bool:
        if cls.device_name in ["cuda", "npu"]:
            return True
        return False
    
    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from src_cache to dst_cache device ."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.to(dst_cache.device)

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from device to host (CPU)."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.cpu()

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    ### NOTE(lms): will effect compile result
    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True
    
    @classmethod
    def use_custom_allreduce(cls) -> bool:
        if cls.dist_backend == "flagcx":
            return False
        return True

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        # TODO(yxa): For NPU/Ascend devices, return None (no capability version like CUDA)
        if cls.device_type == "npu":
            return None
        # For CUDA devices
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)
    
    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        try:
            import pynvml
            pynvml.nvmlInit()
            """
            query if the set of gpus are fully connected by nvlink (1 hop)
            """
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
            for i, handle in enumerate(handles):
                for j, peer_handle in enumerate(handles):
                    if i < j:
                        try:
                            p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                                handle,
                                peer_handle,
                                pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                            )
                            if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                                return False
                        except pynvml.NVMLError:
                            logger.exception(
                                "NVLink detection failed. This is normal if"
                                " your machine has no NVLink equipped."
                            )
                            return False
            return True
        except:
            return False


def _register_oot_quantization_kernels():
    """Register quantization kernel mappings for OOT platform.

    vLLM's quantization kernel selection uses platform-specific mappings:
        - _POSSIBLE_FP8_KERNELS: dict[PlatformEnum, list[FP8ScaledMMLinearKernel]]
        - _POSSIBLE_INT8_KERNELS: dict[PlatformEnum, list[Int8ScaledMMLinearKernel]]
        - _POSSIBLE_KERNELS (mixed_precision): dict[PlatformEnum, list[MPLinearKernel]]

    These mappings only include CUDA, ROCM, CPU, XPU platforms by default.
    When using PlatformFL (PlatformEnum.OOT), vLLM raises KeyError.

    This function patches the global mappings to include OOT platform,
    using CUDA kernels when the OOT platform is CUDA-alike.

    Call this when PlatformFL module is loaded to fix FP8 compatibility.
    """
    try:
        from vllm.platforms import PlatformEnum

        # Import kernel mapping modules
        from vllm.model_executor.layers.quantization.kernels import scaled_mm

        # Check if OOT is already registered
        if PlatformEnum.OOT in scaled_mm._POSSIBLE_FP8_KERNELS:
            logger.debug("OOT platform FP8 kernels already registered")
            return

        # Get device info to determine which kernels to use
        device_info = DeviceInfo()

        # Register FP8 kernels
        if device_info.device_type == "cuda":
            if PlatformEnum.CUDA in scaled_mm._POSSIBLE_FP8_KERNELS:
                scaled_mm._POSSIBLE_FP8_KERNELS[PlatformEnum.OOT] = \
                    scaled_mm._POSSIBLE_FP8_KERNELS[PlatformEnum.CUDA].copy()
                logger.info(
                    "Registered OOT platform FP8 kernels (using CUDA): %s",
                    [k.__name__ for k in scaled_mm._POSSIBLE_FP8_KERNELS[PlatformEnum.OOT]]
                )
        elif PlatformEnum.CPU in scaled_mm._POSSIBLE_FP8_KERNELS:
            scaled_mm._POSSIBLE_FP8_KERNELS[PlatformEnum.OOT] = \
                scaled_mm._POSSIBLE_FP8_KERNELS[PlatformEnum.CPU].copy()
            logger.info("Registered OOT platform FP8 kernels (using CPU fallback)")
        else:
            scaled_mm._POSSIBLE_FP8_KERNELS[PlatformEnum.OOT] = []
            logger.warning("No FP8 kernels available for OOT platform")

        # Register INT8 kernels
        if PlatformEnum.OOT not in scaled_mm._POSSIBLE_INT8_KERNELS:
            if device_info.device_type == "cuda" and PlatformEnum.CUDA in scaled_mm._POSSIBLE_INT8_KERNELS:
                scaled_mm._POSSIBLE_INT8_KERNELS[PlatformEnum.OOT] = \
                    scaled_mm._POSSIBLE_INT8_KERNELS[PlatformEnum.CUDA].copy()
                logger.debug("Registered OOT platform INT8 kernels (using CUDA)")
            elif PlatformEnum.CPU in scaled_mm._POSSIBLE_INT8_KERNELS:
                scaled_mm._POSSIBLE_INT8_KERNELS[PlatformEnum.OOT] = \
                    scaled_mm._POSSIBLE_INT8_KERNELS[PlatformEnum.CPU].copy()

        # Register mixed_precision kernels (W4A8, Marlin, etc.)
        try:
            from vllm.model_executor.layers.quantization.kernels import mixed_precision
            if PlatformEnum.OOT not in mixed_precision._POSSIBLE_KERNELS:
                if device_info.device_type == "cuda" and PlatformEnum.CUDA in mixed_precision._POSSIBLE_KERNELS:
                    mixed_precision._POSSIBLE_KERNELS[PlatformEnum.OOT] = \
                        mixed_precision._POSSIBLE_KERNELS[PlatformEnum.CUDA].copy()
                    logger.debug("Registered OOT platform mixed_precision kernels (using CUDA)")
                elif PlatformEnum.CPU in mixed_precision._POSSIBLE_KERNELS:
                    mixed_precision._POSSIBLE_KERNELS[PlatformEnum.OOT] = \
                        mixed_precision._POSSIBLE_KERNELS[PlatformEnum.CPU].copy()
        except ImportError:
            pass

    except ImportError as e:
        logger.debug(f"OOT quantization kernel registration skipped (import error): {e}")
    except Exception as e:
        logger.warning(f"Failed to register OOT quantization kernels: {e}")


# Register OOT quantization kernels when this module is loaded
# This ensures FP8/INT8 quantization works with PlatformFL
_register_oot_quantization_kernels()

