# Copyright (c) 2026 BAAI. All rights reserved.

"""NVIDIA platform integration for vLLM 0.32, including empty builds."""

import os
from dataclasses import fields
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import PlatformEnum
from vllm.platforms.cuda import CudaPlatform

from vllm_fl.nvidia_compat import has_native_vllm_kernels

if TYPE_CHECKING:
    from vllm.config import VllmConfig


logger = init_logger(__name__)

# These passes replace otherwise valid graphs with fused vLLM CUDA ops.
_NATIVE_C_FUSION_PASSES = (
    "fuse_norm_quant",
    "fuse_act_quant",
    "enable_qk_norm_rope_fusion",
    "fuse_rope_kvcache_cat_mla",
)

_FLASH_ATTN_BACKENDS = frozenset(
    {
        "FLASH_ATTN",
        "FLASH_ATTN_DIFFKV",
        "FLASH_ATTN_MLA",
        "FLASH_ATTN_MLA_SPARSE",
        "TRITON_FLASH_ATTN",
    }
)


def _has_native_flash_attention() -> bool:
    # Upstream's is_flash_attn_varlen_func_available() assumes every CUDA
    # platform has compiled FA kernels. An empty wheel intentionally does not.
    try:
        from vllm.v1.attention.backends import fa_utils

        return callable(getattr(fa_utils, "flash_attn_varlen_func", None)) and (
            fa_utils.get_flash_attn_version() is not None
        )
    except (ImportError, OSError):
        return False


# These backends write KV caches directly through _C_cache_ops. Composite
# TRITON_FLASHINFER delegates its cache write to Triton and is not excluded.
_NATIVE_CACHE_BACKENDS = frozenset({"FLASHINFER", "FLEX_ATTENTION"})


def _has_cuda_cache_writer() -> bool:
    # Both native wheels and the FL empty-build bridge register real CUDA
    # implementations. A schema by itself is not a usable cache writer.
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            "_C_cache_ops::reshape_and_cache_flash", "CUDA"
        )
    except RuntimeError:
        return False


def _attention_backend_unavailable_reason(backend_name: str) -> str | None:
    if backend_name in _FLASH_ATTN_BACKENDS and not _has_native_flash_attention():
        return "FlashAttention kernels are unavailable in this vLLM build"
    if backend_name in _NATIVE_CACHE_BACKENDS and not _has_cuda_cache_writer():
        return "CUDA cache kernels are unavailable in this vLLM build"
    return None


class NvidiaPlatformFL(CudaPlatform):
    """Add FL runtime hooks without changing native CUDA platform semantics."""

    # NVIDIA is an in-tree CUDA platform in vLLM. Marking it OOT makes upstream
    # attention and MoE code enter generic accelerator branches and corrupts
    # MLA decode. FL is an operator/runtime plugin on top of native CUDA.
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
        cls._configure_compilation_passes(vllm_config, use_flagcx)
        if use_flagcx or not has_native_vllm_kernels():
            vllm_config.parallel_config.disable_custom_all_reduce = True
        vllm_config.parallel_config.worker_cls = (
            "vllm_fl.worker.nvidia_worker.NvidiaWorkerFL"
        )

    @classmethod
    def _configure_compilation_passes(
        cls, vllm_config: "VllmConfig", use_flagcx: bool
    ) -> None:
        passes = vllm_config.compilation_config.pass_config
        if not has_native_vllm_kernels():
            # vLLM resolves O-level defaults before this hook and does not retain
            # whether a pass was explicitly selected. Warn for every disabled
            # fusion, while preserving compilation and CUDA graph settings.
            disabled = [
                name for name in _NATIVE_C_FUSION_PASSES if getattr(passes, name, False)
            ]
            if disabled:
                for name in disabled:
                    setattr(passes, name, False)
                logger.warning_once(
                    "Disabling compilation passes %s: the empty vLLM build "
                    "does not provide their fused CUDA kernels. "
                    "Unfused FL/native operations remain enabled.",
                    ", ".join(disabled),
                )
        if use_flagcx and passes.fuse_allreduce_rms:
            # FlashInfer's fusion creates its own communication workspace and
            # collective, bypassing the explicitly selected FlagCX backend.
            passes.fuse_allreduce_rms = False
            logger.warning_once(
                "Disabling fuse_allreduce_rms to preserve the selected "
                "FlagCX communication backend."
            )

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        if cls._refresh_dist_backend():
            return "vllm_fl.distributed.nvidia_communicator.NvidiaCommunicatorFL"
        return super().get_device_communicator_cls()

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        if cls._refresh_dist_backend() or not has_native_vllm_kernels():
            return False
        return super().use_custom_allreduce()

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        # vLLM treats pinned-memory support as its UVA capability gate. Empty
        # builds have no get_cuda_view_from_cpu_tensor kernel, so select the
        # native V2 runner's explicit-copy NonUvaBuffer path instead.
        if not has_native_vllm_kernels():
            return False
        return super().is_pin_memory_available()

    @classmethod
    def get_default_ir_op_priority(cls, vllm_config):
        priorities = super().get_default_ir_op_priority(vllm_config)
        if not has_native_vllm_kernels():
            # CUDA's eager defaults prefer vllm_c, but schemas alone do not
            # supply those kernels. Keep real native/Triton IR providers.
            for field in fields(priorities):
                providers = getattr(priorities, field.name)
                setattr(
                    priorities,
                    field.name,
                    [provider for provider in providers if provider != "vllm_c"],
                )
        return priorities

    @classmethod
    def get_valid_backends(
        cls, device_capability, attn_selector_config, num_heads: int | None = None
    ):
        candidates, invalid_reasons = super().get_valid_backends(
            device_capability, attn_selector_config, num_heads
        )
        available = []
        for candidate in candidates:
            reason = _attention_backend_unavailable_reason(candidate.backend.name)
            if reason is not None:
                invalid_reasons[candidate.backend] = (candidate.priority, [reason])
            else:
                available.append(candidate)
        return available, invalid_reasons

    @classmethod
    def get_supported_vit_attn_backends(cls):
        backends = super().get_supported_vit_attn_backends()
        if _has_native_flash_attention():
            return backends
        return [backend for backend in backends if backend.name != "FLASH_ATTN"]

    @classmethod
    def get_vit_attn_backend(cls, head_size, dtype, backend=None):
        if (
            backend is not None
            and backend.name == "FLASH_ATTN"
            and not _has_native_flash_attention()
        ):
            raise ValueError(
                "FLASH_ATTN vision attention requires FlashAttention kernels; "
                "choose TRITON_ATTN or TORCH_SDPA for an empty vLLM build"
            )
        return super().get_vit_attn_backend(head_size, dtype, backend)

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        attn_selector_config,
        num_heads: int | None = None,
    ) -> str:
        """Keep native CUDA selection unless FlagGems attention is requested."""
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

        if selected_backend is not None:
            reason = _attention_backend_unavailable_reason(selected_backend.name)
            if reason is not None:
                raise ValueError(
                    f"{selected_backend.name} requires "
                    + reason.replace(" are unavailable", ", which are unavailable")
                )
        return super().get_attn_backend_cls(
            selected_backend,
            attn_selector_config,
            num_heads,
        )
