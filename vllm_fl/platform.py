# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/platforms/cuda.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING, Optional, TypeVar
from typing_extensions import ParamSpec

import torch

# import custom ops, trigger op registration (CUDA only)
try:
    import vllm._C  # noqa
except (ImportError, OSError):
    pass  # NPU or other platforms may not have vllm._C

from vllm.attention.backends.registry import AttentionBackendEnum
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

from vllm_fl.utils import DeviceInfo, get_device_name, get_device_type

logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")

dist_backend_dict = {
    "npu": "hccl",
    "cuda": "nccl",
}


class PlatformFL(Platform):
    _enum = PlatformEnum.OOT
    device_info = DeviceInfo()
    vendor_name = device_info.vendor_name
    device_type = get_device_type(vendor_name)
    device_name = get_device_name(vendor_name)
    dispatch_key = device_info.dispatch_key
    torch_device_fn = device_info.torch_device_fn
    ray_device_key: str = "GPU"
    dist_backend: str = (
        "flagcx" if "FLAGCX_PATH" in os.environ else dist_backend_dict.get(device_name, "nccl")
    )
    ### TODO(lms): dispatch device_control_env_var
    # device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    def is_cuda_alike(self) -> bool:
        """Stateless version of [torch.cuda.is_available][]."""
        if self.vendor_name == "iluvatar":
            return False
        return self.device_type == "cuda"

    def is_cuda(self) -> bool:
        """Stateless version of [torch.cuda.is_available][]."""
        return self.device_type == "cuda" and self.vendor_name == "nvidia"

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
    def get_current_memory_usage(
        cls, device: Optional[torch.types.Device] = None
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
    def import_kernels(cls) -> None:
        """Import device-specific kernels."""
        logger.info(f"current vendor_name is: {cls.vendor_name}")
        if cls.vendor_name == "metax":
            try:
                import mcoplib._C  # noqa: F401
            except ImportError:
                logger.warning("Failed to import mcoplib._C")

            try:
                import mcoplib._moe_C  # noqa: F401
            except ImportError:
                logger.warning("Failed to import mcoplib._moe_C")

            try:
                import vllm_fl.dispatch.backends.vendor.metax.patches  # noqa: F401
            except Exception as e:
                logger.warning(f"Failed to import maca patches: {e}")

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config

        parallel_config.worker_cls = "vllm_fl.worker.worker.WorkerFL"

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

        if (
            parallel_config.data_parallel_size > 1
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

        # --------------------------------------------------------
        # maca specific config updates
        if cls.vendor_name == "metax":
            if model_config is not None:
                model_config.disable_cascade_attn = True
            if attention_config := vllm_config.attention_config:
                attention_config.use_cudnn_prefill = False
                attention_config.use_trtllm_ragged_deepseek_prefill = False
                attention_config.use_trtllm_attention = False
                attention_config.disable_flashinfer_prefill = True

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
    ) -> list[str]:
        """Get the attention backend class path using the dispatch mechanism."""
        from vllm_fl.dispatch import call_op

        use_mla = attn_selector_config.use_mla
        use_sparse = attn_selector_config.use_sparse

        backend_path = call_op("attention_backend", use_mla=use_mla, use_sparse=use_sparse)

        logger.info_once(
            "Using attention backend via dispatch (use_mla=%s, use_sparse=%s): %s",
            use_mla,
            use_sparse,
            backend_path,
            scope="local",
        )
        logger.info(
            "Using attention backend via dispatch (use_mla=%s): %s"
            % (use_mla, backend_path)
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
        from vllm_fl.attention.utils import patch_mm_encoder_attention

        patch_mm_encoder_attention()
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
            return "vllm_fl.distributed.communicator.CommunicatorFL"  # noqa
        else:
            logger.info("Using CudaCommunicator for communication.")
            return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm_fl.compilation.graph.GraphWrapper"

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        if cls.vendor_name in ["nvidia", "ascend", "metax"]:
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
    def pre_register_and_update(cls, parser = None) -> None:
        if cls.device_name == "npu":
            import vllm_fl.dispatch.backends.vendor.ascend


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
            handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids
            ]
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
