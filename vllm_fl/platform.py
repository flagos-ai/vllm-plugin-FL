# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.20.2/vllm/platforms/cuda.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from typing import TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

import torch

# import custom ops, trigger op registration (CUDA only)
try:
    import vllm._C  # noqa
    import vllm._C_stable_libtorch  # noqa
except (ImportError, OSError):
    pass  # NPU or other platforms may not have vllm._C

from vllm.logger import init_logger
from vllm.platforms import Platform, PlatformEnum
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm_fl.dispatch import CachedOp

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None
    CacheDType = None

from vllm_fl.utils import DeviceInfo, get_device_name, get_device_type

logger = init_logger(__name__)

_attention_backend = CachedOp("attention_backend")

_P = ParamSpec("_P")
_R = TypeVar("_R")

dist_backend_dict = {
    "npu": "hccl",
    "cuda": "nccl",
    "gcu": "eccl",
    "musa": "mccl",
}


def _ascend_npugraph_ex_enabled(vllm_config: "VllmConfig") -> bool:
    """Return whether the opt-in Ascend npugraph_ex compiler is enabled.

    Keep this behind the same nested additional-config shape used by
    vLLM-Ascend.  The default remains the existing eager-FX path so a runtime
    without npugraph_ex is unaffected.
    """
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    ascend_compilation_config = additional_config.get(
        "ascend_compilation_config", {}
    )
    if not isinstance(ascend_compilation_config, dict):
        logger.warning(
            "Ignoring non-dict ascend_compilation_config: %r",
            ascend_compilation_config,
        )
        return False
    return bool(ascend_compilation_config.get("enable_npugraph_ex", False))

def _resolve_flagcx_backend() -> bool:
    """Check whether the flagcx torch distributed backend is available."""
    flagcx_path = os.environ.get("FLAGCX_PATH")
    if not flagcx_path:
        return False
    try:
        if flagcx_path not in sys.path:
            sys.path.insert(0, flagcx_path)
        import flagcx  # triggers _C.so load and backend registration
        return torch.distributed.is_backend_available("flagcx")
    except Exception:
        logger.warning(
            "FLAGCX_PATH=%s is set but flagcx torch backend could not be loaded.",
            flagcx_path,
        )
        return False


class PlatformFL(Platform):
    _enum = PlatformEnum.OOT
    device_info = DeviceInfo()
    vendor_name = device_info.vendor_name
    device_type = get_device_type(vendor_name)
    device_name = get_device_name(vendor_name)
    # cuda_alike (nvidia/metax): device_name = vendor_name (not used in torch.device)
    # non-cuda_alike (iluvatar/ascend): device_name = device_type (used in torch.device)
    device_name = device_info.vendor_name if (
        device_info.device_type == "cuda"
        and device_info.vendor_name not in ("iluvatar", "hygon")
    ) else device_info.device_type
    device_type = device_info.device_type
    dispatch_key = device_info.dispatch_key
    torch_device_fn = device_info.torch_device_fn
    # Upstream uses this backend for small, independently decorated helpers
    # such as sampler logprob ranking.  NPU Inductor is not part of FL's
    # Ascend graph path; keep these helpers eager just like vLLM-Ascend does.
    simple_compile_backend: str = "eager" if device_type == "npu" else "inductor"
    ray_device_key: str = "GPU"
    dist_backend: str = (
        "flagcx"
        if _resolve_flagcx_backend()
        else dist_backend_dict.get(device_name, "nccl")
    )
    ### TODO(lms): dispatch device_control_env_var
    # device_control_env_var: str = "CUDA_VISIBLE_DEVICES"

    @classmethod
    def get_compile_backend(cls) -> str:
        # vLLM asks the platform for the import path only when
        # CompilationConfig.backend is neither "eager" nor "inductor".  The
        # default Ascend path remains EagerAdaptor; the custom path is selected
        # only by the explicit npugraph_ex opt-in in check_and_update_config().
        if cls.device_type == "npu":
            return "vllm_fl.compilation.compiler_interface.AscendCompiler"
        return super().get_compile_backend()

    @classmethod
    def _get_default_max_cudagraph_capture_size(
        cls, vllm_config: "VllmConfig"
    ) -> int | None:
        """Return vLLM-Ascend's workload-bound default capture maximum."""
        compilation_config = vllm_config.compilation_config
        if compilation_config.max_cudagraph_capture_size is not None:
            return None
        if compilation_config.cudagraph_capture_sizes is not None:
            return None

        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        max_num_seqs = getattr(scheduler_config, "max_num_seqs", None)
        if max_num_seqs is None:
            return None

        decode_query_len = 1
        speculative_config = getattr(vllm_config, "speculative_config", None)
        if speculative_config and speculative_config.num_speculative_tokens:
            decode_query_len += speculative_config.num_speculative_tokens

        return min(max_num_seqs * decode_query_len, 512)

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: "VllmConfig") -> None:
        if cls.device_type != "npu":
            return

        default_max_capture_size = cls._get_default_max_cudagraph_capture_size(
            vllm_config
        )
        if default_max_capture_size is not None:
            vllm_config.compilation_config.max_cudagraph_capture_size = (
                default_max_capture_size
            )

    def is_cuda_alike(self) -> bool:
        """Stateless version of [torch.cuda.is_available][]."""
        if self.vendor_name == "iluvatar":
            return False
        if self.device_type == "musa":
            return True
        if self.vendor_name == "hygon":
            return False
        if self.vendor_name == "gcu":
            return True
        return self.device_type == "cuda"

    def is_cuda(self) -> bool:
        """Stateless version of [torch.cuda.is_available][]."""
        return self.device_type == "cuda" and self.vendor_name == "nvidia"

    def is_musa(self) -> bool:
        if hasattr(torch, 'musa') and torch.musa.is_available():
            return True
        return False

    def is_gcu(self) -> bool:
        if hasattr(torch, 'gcu') and torch.gcu.is_available():
            return True
        return False

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
        cls, device: torch.types.Device | None = None
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
        if cls.device_type in ["cuda", "xpu", "npu", "musa", "txda", "gcu"]:
            return True
        return False

    @classmethod
    def import_kernels(cls) -> None:
        """Import device-specific kernels."""
        logger.info(f"current vendor_name is: {cls.vendor_name}")

        if cls.device_type == "npu":
            try:
                from vllm_fl.ascend_custom_ops import bootstrap_custom_op_env

                bootstrap_custom_op_env()
            except (ImportError, OSError) as exc:
                logger.warning("Ascend custom OPP is unavailable: %s", exc)

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
        else:
            super().import_kernels()

        if cls.device_type == "musa":
            try:
                from vllm_fl.dispatch.backends.vendor.musa.patch import apply_musa_patches
                apply_musa_patches()
            except Exception as e:
                logger.warning(f"Failed to apply MUSA patches: {e}")

    @classmethod
    def import_ir_kernels(cls) -> None:
        """Import IR kernel modules. OOT platforms override to import their own."""
        import vllm.kernels  # noqa: F401

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> None:
        # vLLM-Ascend chooses the NPU block size during config validation and
        # intentionally skips upstream's post-load backend realignment.  The
        # latter treats the combined Mamba state as one page and changes this
        # model's correctly aligned 2048-token page back to 1152 tokens.
        if cls.device_type == "npu":
            return
        super().update_block_size_for_backend(vllm_config)

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
            elif cls.vendor_name == "kunlunxin":
                cache_config.block_size = 128
                logger.info("Setting kv cache block size to 128 for Kunlunxin.")
            elif cls.device_type == "musa":
                cache_config.block_size = 64
                logger.info("Setting kv cache block size to 64 for MUSA.")
            else:
                cache_config.block_size = 16
        if cls.device_type == "npu":
            # vLLM-Ascend does not globally replace ATen operators with
            # FlagGems.  Keep Ascend on its vendor/torch_npu implementations
            # by default as well; global FlagGems registration changes model
            # numerics and also intercepts sampler reductions such as
            # sum(bool), which its Ascend Triton backend cannot compile.
            os.environ["USE_FLAGGEMS"] = "0"
            logger.info(
                "Disabling global FlagGems on Ascend; using FL-local vendor "
                "and torch_npu implementations."
            )

            from vllm_fl.dispatch.backends.vendor.ascend.patch import refresh_block_size

            refresh_block_size(vllm_config)
            # vLLM-Ascend's NPU communicator uses the HCCL process group and
            # does not provide CUDA's custom all-reduce implementation.
            parallel_config.disable_custom_all_reduce = True

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

        # Ascend full graph uses vLLM's eager FX backend plus an outer NPUGraph.
        # Inductor and piecewise graph modes are intentionally out of scope.
        if cls.device_type == "npu":
            from vllm.config import CompilationMode

            enforce_eager = bool(
                model_config is not None
                and getattr(model_config, "enforce_eager", False)
            )
            if enforce_eager or compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
                compilation_config.mode = CompilationMode.NONE
                compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            elif compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
                compilation_config.mode = CompilationMode.VLLM_COMPILE
                enable_npugraph_ex = _ascend_npugraph_ex_enabled(vllm_config)
                if enable_npugraph_ex:
                    # Keep the compiler implementation local to FL: the
                    # standalone wheel must not import vllm-ascend at runtime.
                    compilation_config.backend = (
                        "vllm_fl.compilation.compiler_interface.AscendCompiler"
                    )
                else:
                    # Preserve the validated compatibility path by default.
                    compilation_config.backend = "eager"
                compilation_config.use_inductor = False
                compilation_config.splitting_ops = []
                # These post-grad quantization fusions are CUDA-only in
                # upstream vLLM.  The NPU path imports neither pass class and
                # this BF16 graph does not need them, so keep the eager FX
                # backend free of CUDA-specific post passes.
                compilation_config.pass_config.fuse_norm_quant = False
                compilation_config.pass_config.fuse_act_quant = False
                compilation_config.pass_config.fuse_attn_quant = False
                compilation_config.cudagraph_num_of_warmups = 1
                if enable_npugraph_ex:
                    logger.info(
                        "Enabling Ascend FULL_DECODE_ONLY with FL-local "
                        "AscendCompiler/npugraph_ex and NPUGraph."
                    )
                else:
                    logger.info(
                        "Enabling Ascend FULL_DECODE_ONLY with eager FX and "
                        "NPUGraph."
                    )
            else:
                logger.warning(
                    "Ascend FL currently supports only NONE or "
                    "FULL_DECODE_ONLY graph mode; falling back to NONE from %s.",
                    compilation_config.cudagraph_mode,
                )
                compilation_config.mode = CompilationMode.NONE
                compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        if (
            cls.device_type == "musa"
            and compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            logger.info(
                "MUSA: Downgrading cudagraph_mode from %s to PIECEWISE because "
                "FULL cudagraphs require musaStreamCaptureModeThreadLocal which "
                "is not yet supported by torch_musa. PIECEWISE graphs still "
                "provide graph capture benefits for non-TP-communication regions.",
                compilation_config.cudagraph_mode,
            )
            compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

        if (
            parallel_config.all2all_backend == "deepep_high_throughput"
            and parallel_config.data_parallel_size > 1
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

        if cls.vendor_name == "gcu":
            parallel_config.disable_custom_all_reduce = True

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum | None",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        """Get the attention backend class path using the dispatch mechanism."""
        if cls.device_type == "npu":
            # Keep attention and its metadata builder from the same current
            # vLLM-Ascend release as the reused model operators.
            return (
                "vllm_fl.dispatch.backends.vendor.ascend.impl."
                "native_attention.AscendAttentionBackendFL"
            )
        use_mla = attn_selector_config.use_mla
        use_sparse = attn_selector_config.use_sparse

        backend_path = _attention_backend(use_mla=use_mla, use_sparse=use_sparse)

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
        backend: "AttentionBackendEnum | None" = None,
    ) -> "AttentionBackendEnum":
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
        if cls.device_type == "npu":
            logger.info("Using vLLM-Ascend-compatible NPU communicator.")
            return (
                "vllm_fl.distributed.device_communicators."
                "npu_communicator.NPUCommunicator"
            )
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
        # Current detection reports vendor_name="enflame" and
        # device_type="gcu". Keep the legacy "gcu" vendor alias for images
        # built with the earlier detector.
        if cls.vendor_name in {
            "nvidia",
            "ascend",
            "metax",
            "hygon",
            "mthreads",
            "iluvatar",
            "thead",
            "gcu",
            "enflame",
            "kunlunxin",
        }:
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

    @classmethod
    def set_additional_forward_context(cls, *args, **kwargs):
        """Keep the base context portable; FL populates NPU extras locally."""
        return super().set_additional_forward_context(*args, **kwargs)

    ### NOTE(lms): will effect compile result
    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        if cls.vendor_name == "hygon":
            return False
        if cls.dist_backend == "flagcx":
            return False
        return True

    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        if cls.device_name == "npu":
            import vllm_fl.dispatch.backends.vendor.ascend

            # Match vLLM-Ascend's platform-patch lifecycle.  This hook runs
            # after platform resolution but before VllmConfig construction,
            # which is early enough for hybrid Mamba/attention cache sizing
            # without recursively importing vLLM while the plugin itself is
            # still being resolved.
            from vllm_fl.dispatch.backends.vendor.ascend.patch import (
                patch_mamba_config,
            )

            patch_mamba_config()
        elif cls.device_name == "gcu":
            import vllm_fl.dispatch.backends.vendor.gcu  # noqa: F401

    @classmethod
    def supports_fp8(cls) -> bool:
        """Return whether the current device architecture supports FP8."""
        if cls.vendor_name == "mthreads":
            return True

        if cls.vendor_name == "gcu":
            cc = cls.get_device_capability()
            return cc is not None and cc.major >= 4

        if cls.vendor_name != "nvidia":
            return False
        try:
            capability = cls.get_device_capability()
        except (AttributeError, RuntimeError):
            return False
        return capability is not None and capability >= DeviceCapability(8, 9)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        if cls.device_type == "cuda":
            import pynvml
            pynvml.nvmlInit()
            physical_device_id = cls.device_id_to_physical_device_id(device_id)
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            pynvml.nvmlShutdown()
            return uuid
        elif cls.device_type == "npu":
            if os.getenv("ASCEND_RT_VISIBLE_DEVICES") is not None:
                npu_visible_devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
                return "NPU-" + npu_visible_devices[device_id]
            return f"NPU-{device_id}"
        else:
            return f"{cls.device_type}-{device_id}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return cls.torch_device_fn.get_device_properties(
            device_id
        ).total_memory

    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        return cls.vendor_name in ("nvidia", "thead", "iluvatar")

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        return cls.torch_device_fn.get_device_properties(device_id).multi_processor_count


    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        # TODO(yxa): For NPU/Ascend devices, return None (no capability version like CUDA)
        if cls.device_type == "npu":
            return None
        if cls.device_type == "musa":
            major, minor = torch.musa.get_device_capability(device_id)
            return DeviceCapability(major=major, minor=minor)
        if cls.device_type == "txda":
            major, minor = torch.txda.get_device_capability(device_id)
            return DeviceCapability(major=major, minor=minor)
        # TODO: For PTPU/Sunrise devices, return None
        if cls.device_type == "ptpu":
            return None
        if cls.device_type == "gcu":
            gcu = getattr(torch, "gcu", None)
            if gcu is None:
                return None
            major, minor = gcu.get_device_capability(device_id)
            return DeviceCapability(major=major, minor=minor)
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def support_deep_gemm(cls) -> bool:
        """Currently, only Hopper and Blackwell GPUs are supported."""
        if cls.device_type == "cuda" and cls.vendor_name == "nvidia":
            return cls.is_device_capability(90) or cls.is_device_capability_family(100)
        return False

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

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        """Set RNG seed across all devices for the current platform."""
        # torch_ptpu.ptpu doesn't have manual_seed_all, implement it manually
        if hasattr(cls.torch_device_fn, 'manual_seed_all'):
            cls.torch_device_fn.manual_seed_all(seed)
        else:
            # Fallback for devices without manual_seed_all (e.g., ptpu)
            torch.manual_seed(seed)
            if hasattr(cls.torch_device_fn, 'device_count') and hasattr(cls.torch_device_fn, '_get_or_create_default_generator'):
                # Set seed for each device's default generator
                for device_id in range(cls.torch_device_fn.device_count()):
                    generator = cls.torch_device_fn._get_or_create_default_generator(device_id)
                    generator.manual_seed(seed)

    @classmethod
    def is_integrated_gpu(cls, device_id: int = 0) -> bool:
        """Returns whether the GPU is an integrated (UMA) device."""
        return False
