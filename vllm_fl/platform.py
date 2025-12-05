import os
from datetime import timedelta
from functools import cache, wraps
from typing import TYPE_CHECKING, Callable, Optional, TypeVar, Union

import torch

import vllm.envs as envs
from vllm.logger import init_logger

from vllm.platforms import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.attention.backends.registry import _Backend
    from vllm.config import VllmConfig
else:
    _Backend = None

from vllm_fl.utils import DeviceInfo

logger = init_logger(__name__)

class PlatformFL(Platform):
    _enum = PlatformEnum.OOT
    device_info = DeviceInfo()
    device_name = device_info.device_type 
    device_type = device_info.device_type 
    dispatch_key = device_info.dispatch_key
    torch_device_fn = device_info.torch_device_fn
    ray_device_key: str = "flagos"
    dist_backend: str = "flagcx"
    ### TODO(lms): dispatch device_control_env_var
    # device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    def is_cuda_alike(self) -> bool:
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
    def get_device_capability(cls, device_id: int = 0):
        pass

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return cls.device_name

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """
        Verify whether the quantization is supported by the current platform.
        """
        if cls.supported_quantization and quant not in cls.supported_quantization:
            raise ValueError(
                f"{quant} quantization is currently not supported in {cls.device_name}."
            )
        
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

        parallel_config.worker_cls = "vllm_fl.worker.worker.WorkerFL"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
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
            use_sparse = hasattr(vllm_config.model_config.hf_config, "index_topk")
            # If `VLLM_ATTENTION_BACKEND` is not set and we are using MLA,
            # then we default to FlashMLA backend for non-blackwell GPUs,
            # else we default to CutlassMLA. For each case, we force the
            # required block_size.
            use_flashmla = False
            use_cutlass_mla = False
            use_flashinfer_mla = False

            if envs.VLLM_ATTENTION_BACKEND is None:
                # Default case
                use_flashmla = True
            else:
                # Forced case
                use_flashmla = envs.VLLM_ATTENTION_BACKEND == "FLASHMLA"
                use_cutlass_mla = envs.VLLM_ATTENTION_BACKEND == "CUTLASS_MLA"
                use_flashinfer_mla = envs.VLLM_ATTENTION_BACKEND == "FLASHINFER_MLA"

            from vllm.attention.ops.flashmla import is_flashmla_dense_supported

            if (
                use_flashmla
                and is_flashmla_dense_supported()[0]
                and cache_config.block_size % 64 != 0
            ):
                cache_config.block_size = 64
                logger.info("Forcing kv cache block size to 64 for FlashMLA backend.")

            if use_cutlass_mla and cache_config.block_size % 128 != 0:
                cache_config.block_size = 128
                logger.info(
                    "Forcing kv cache block size to 128 for CUTLASS_MLA backend."
                )

            if (
                use_flashinfer_mla
                and cache_config.block_size != 32
                and cache_config.block_size % 64 != 0
            ):
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashInferMLA backend."
                )

            # TODO(Chen): remove this hacky code
            if use_sparse and cache_config.block_size != 64:
                cache_config.block_size = 64
                logger.info(
                    "Forcing kv cache block size to 64 for FlashMLASparse backend."
                )
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
        selected_backend,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_v1,
        use_mla,
        has_sink,
        use_sparse,
    ) -> str:

        ### TODO(lms): support int8 kv cache
        # use_fp8_kv_cache = kv_cache_dtype is not None and kv_cache_dtype.startswith(
        #     "fp8"
        # )

        if use_mla:
            ### TODO(lms): support mla
            raise NotImplementedError
            # logger.info_once("Using FL MLA Attention backend.")
            # return (
            #         "vllm_fl.attention.backends.mla.MLAFLBackend"
            #     )
        else:
            logger.info_once("Using FL Attention backend.")
            return (
                    "vllm_fl.attention.attention.AttentionFLBackend"
                )

    @classmethod
    def get_punica_wrapper(cls) -> str:
        # TODO(lms): support fl PunicaWrapper
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return (
            "vllm_fl.distributed.communicator.CommunicatorFL"  # noqa
        )

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm_fl.compilation.graph.GraphWrapper"
    
    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True
    
    ### TODO(lms): support hybrid kv cache
    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return False

    ### NOTE(lms): will effect compile result
    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True
    
