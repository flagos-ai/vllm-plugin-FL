# Copyright (c) 2026 BAAI. All rights reserved.
"""Runtime registration for the Qwen3.8-Flash-Next / Qwen4Exp Day0 model.

The checkpoint keeps its original ``qwen4_exp`` model types and architecture
name.  This module maps those public names to the plugin-owned implementation
without modifying either the checkpoint or the installed vLLM package.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = logging.getLogger(__name__)


def _strip_mrope(model_config: "ModelConfig") -> None:
    configs = {
        id(config): config
        for config in (
            getattr(model_config, "hf_config", None),
            model_config.hf_text_config,
        )
        if config is not None
    }
    for config in configs.values():
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_parameters.pop("mrope_section", None)
            rope_parameters.pop("mrope_interleaved", None)


class Qwen3_8FlashNextForConditionalGenerationConfig(
    Qwen3_5ForConditionalGenerationConfig
):
    """Apply the hybrid-cache and unsupported-feature contract."""

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        text_config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config

        # vLLM 0.24's Qwen3.5 verifier is intentionally empty. Preserve the
        # checkpoint's FP32 recurrent-state contract here.
        mamba_ssm_dtype = getattr(text_config, "mamba_ssm_dtype", None)
        if cache_config.mamba_ssm_cache_dtype == "auto":
            if mamba_ssm_dtype is not None:
                cache_config.mamba_ssm_cache_dtype = mamba_ssm_dtype
        elif (
            mamba_ssm_dtype is not None
            and cache_config.mamba_ssm_cache_dtype != mamba_ssm_dtype
        ):
            logger.warning(
                "Qwen4Exp config requests mamba_ssm_dtype=%s, but the runtime "
                "override is %s; preserving the explicit runtime value.",
                mamba_ssm_dtype,
                cache_config.mamba_ssm_cache_dtype,
            )

        if int(text_config.hc_count) <= 1:
            raise ValueError("Qwen4Exp requires hc_count > 1")

        parallel_config = vllm_config.parallel_config
        uses_ple_or_qsa = bool(text_config.ple_layer_ids) or (
            getattr(text_config, "indexer_n_heads", None) is not None
        )
        if uses_ple_or_qsa and (
            parallel_config.enable_dbo or parallel_config.ubatch_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp PLE/QSA does not support dual-batch overlap or "
                "microbatching in the Day0 path"
            )
        if bool(text_config.ple_layer_ids) and parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Qwen4Exp PLE requires pipeline_parallel_size=1 because raw "
                "token n-gram context is not broadcast between PP stages"
            )

        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is not None and multimodal_config.language_model_only:
            _strip_mrope(vllm_config.model_config)

        spec_config = vllm_config.speculative_config
        if spec_config is not None:
            raise NotImplementedError(
                "Qwen4Exp Day0 serves normal next-token generation first; "
                "native MTP/speculative decoding is a separate follow-up gate"
            )


class Qwen3_8FlashNextForCausalLMConfig(
    Qwen3_8FlashNextForConditionalGenerationConfig
):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        Qwen3_8FlashNextForConditionalGenerationConfig.verify_and_update_config(
            vllm_config
        )
        _strip_mrope(vllm_config.model_config)


_ARCHITECTURES = {
    "Qwen3_8FlashNextForCausalLM": (
        "Qwen3_8FlashNextForCausalLM",
        Qwen3_8FlashNextForCausalLMConfig,
        "text",
    ),
    "Qwen3_8FlashNextForConditionalGeneration": (
        "Qwen3_8FlashNextForConditionalGeneration",
        Qwen3_8FlashNextForConditionalGenerationConfig,
        "multimodal",
    ),
    "Qwen4ExpForCausalLM": (
        "Qwen4ExpForCausalLM",
        Qwen3_8FlashNextForCausalLMConfig,
        "text",
    ),
    "Qwen4ExpForConditionalGeneration": (
        "Qwen4ExpForConditionalGeneration",
        Qwen3_8FlashNextForConditionalGenerationConfig,
        "multimodal",
    ),
}

# vLLM 0.24 may expose either the nested text config or the outer multimodal
# config through ``hf_text_config`` while the platform policy is installed.
_TEXT_MODEL_TYPES = {
    "qwen3_8_flash_next",
    "qwen3_8_flash_next_text",
    "qwen4_exp",
    "qwen4_exp_text",
}
_MODEL_ARCHITECTURES = {
    "Qwen3_8FlashNextForCausalLM",
    "Qwen3_8FlashNextForConditionalGeneration",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
}


def needs_native_index_select(vllm_config: "VllmConfig") -> bool:
    """Return whether FlagGems PLE state row I/O must be disabled.

    FlagGems 5.3/5.4 materializes a contiguous copy of non-contiguous inputs.
    Qwen3.8-Flash-Next exposes its multi-gigabyte PLE state cache as a
    transposed view.  FlagGems index_select is slower than native ATen and can
    OOM.  This check is deliberately model-scoped.
    """

    model_config = vllm_config.model_config
    text_config = model_config.hf_text_config
    if getattr(text_config, "model_type", None) in _TEXT_MODEL_TYPES:
        return True
    hf_config = getattr(model_config, "hf_config", None)
    if getattr(hf_config, "model_type", None) in _TEXT_MODEL_TYPES:
        return True
    architectures = (
        getattr(model_config, "architectures", None)
        or getattr(hf_config, "architectures", None)
        or ()
    )
    return any(name in _MODEL_ARCHITECTURES for name in architectures)


def apply_native_index_select_policy(
    vllm_config: "VllmConfig",
    whitelist: Optional[list[str]],
    blacklist: Optional[list[str]],
    *,
    vendor_name: Optional[str] = None,
) -> tuple[Optional[list[str]], Optional[list[str]]]:
    """Merge model-scoped native-op exclusions.

    The convolution and padding entries are included because the
    FlagGems 5.3/5.4 PLE profile-run path either overflows the padding launch
    grid or faults while autotuning Conv2d at c=64 with a 16K dummy token
    batch. Each accelerator's PyTorch backend supplies the safe fallback.

    On NVIDIA only, keep decode ``repeat_interleave`` and dense ``linear`` on
    native ATen.  The generic FlagGems repeat-interleave implementation drops
    vLLM's host-known ``output_size`` and performs ``cumsum[-1].item()``, which
    introduces one stream synchronization per GDN layer.  Matched H100 nsys
    also shows the generic FlagGems linear path at roughly 3x the native dense
    GEMM time.  Other accelerator vendors retain their FlagGems paths until
    they have vendor-specific measurements; this is a performance policy, not
    a CUDA-only model requirement.
    """

    if not needs_native_index_select(vllm_config):
        return whitelist, blacklist
    required_native_ops = [
        "index_select",
        "conv1d",
        "_conv_depthwise2d",
        "conv2d",
        "pad",
        "constant_pad_nd",
    ]
    if vendor_name == "nvidia":
        # FlagGems ``unused`` entries are implementation function names, not
        # ATen overload names (for example, not repeat_interleave.self_Tensor).
        required_native_ops.extend(
            (
                "repeat_interleave_tensor",
                "repeat_interleave_self_tensor",
                # aten::linear is CompositeImplicitAutograd.  Once its direct
                # FlagGems implementation is disabled it decomposes to mm or
                # addmm, so those leaf implementations must stay native too.
                "linear",
                "mm",
                "mm_out",
                "addmm",
                "addmm_out",
                "addmm_",
                "addmm_dtype",
                "addmm_dtype_out",
            )
        )
    if whitelist:
        conflicts = [op_name for op_name in required_native_ops if op_name in whitelist]
        if conflicts:
            raise ValueError(
                "Qwen3.8-Flash-Next requires native PLE runtime primitives; "
                "remove these operators from the FlagGems whitelist: "
                + ", ".join(conflicts)
            )
        return whitelist, blacklist
    merged = list(blacklist or [])
    for op_name in required_native_ops:
        if op_name not in merged:
            merged.append(op_name)
    return whitelist, merged


def should_skip_generic_flaggems_aten(
    vllm_config: "VllmConfig",
    *,
    vendor_name: Optional[str],
    whitelist: Optional[list[str]],
) -> bool:
    """Keep generic NVIDIA ATen native while retaining explicit FlagOS ops.

    Matched H100 c=64 profiling found that QSA metadata preparation spent 47%
    of rank-0 GIL samples in FlagGems Python pointwise dispatch and shape
    heuristics. Explicit fused/OOT QSA, HC and MoE kernels are registered by a
    separate path and remain enabled. Other accelerators keep FlagGems because
    their native PyTorch backends may not offer an equivalent fast path. An
    explicit whitelist remains a user override. An explicitly requested ATen
    plan-cache also retains generic FlagGems dispatch so it can cache plans.
    """

    return (
        vendor_name == "nvidia"
        and whitelist is None
        and needs_native_index_select(vllm_config)
        and os.getenv(
            "VLLM_FL_FLAGGEMS_ATEN_PLAN_CACHE",
            os.getenv("FLAGGEMS_ATEN_PLAN_CACHE", "0"),
        ).strip().lower() in {"0", "false", "off", "no"}
    )


def _patch_common_attention_token_to_req_cache() -> None:
    """Backport vLLM's shared per-token request-index cache to v0.24."""
    import torch
    from vllm.v1.attention.backend import CommonAttentionMetadata

    if hasattr(CommonAttentionMetadata, "token_to_req_indices"):
        return

    def token_to_req_indices(self, buffer: torch.Tensor) -> torch.Tensor:
        num_tokens = self.num_actual_tokens
        cached = getattr(self, "_token_to_req_indices_cache", None)
        if cached is not None:
            assert cached.device == buffer.device
            assert cached.dtype == torch.int32
            assert cached.shape[0] >= num_tokens
            return cached[:num_tokens]

        num_mapped_tokens = int(self.query_start_loc_cpu[-1])
        query_lens = self.query_start_loc[1:] - self.query_start_loc[:-1]
        assert buffer.shape[0] >= max(num_mapped_tokens, num_tokens)
        mapping = torch.repeat_interleave(
            torch.arange(
                query_lens.shape[0], dtype=torch.int32, device=buffer.device
            ),
            query_lens,
            output_size=num_mapped_tokens,
        )
        buffer[:num_mapped_tokens].copy_(mapping)
        if num_mapped_tokens < num_tokens:
            buffer[num_mapped_tokens:num_tokens].zero_()
        cached = buffer[: max(num_mapped_tokens, num_tokens)]
        self._token_to_req_indices_cache = cached
        return cached[:num_tokens]

    setattr(CommonAttentionMetadata, "token_to_req_indices", token_to_req_indices)


def _patch_ple_metadata_bridge() -> None:
    """Teach the v0.24 hybrid state to pass spec fields to PLE metadata."""
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata

    from vllm_fl.models.qwen3_8_flash_next.common.short_conv_attn import (
        PleShortConvAttentionMetadataBuilder,
    )

    original = MambaHybridAttnMetadata.get_extra_attn_kwargs
    if getattr(original, "_vllm_fl_qwen38_ple", False):
        return

    def get_extra_attn_kwargs(self, attn_metadata_builder, num_reqs):
        if isinstance(attn_metadata_builder, PleShortConvAttentionMetadataBuilder):
            return {
                "num_accepted_tokens": None
                if self.num_accepted_tokens is None
                else self.num_accepted_tokens[:num_reqs],
                "num_decode_draft_tokens_cpu": None
                if self.num_decode_draft_tokens_cpu is None
                else self.num_decode_draft_tokens_cpu[:num_reqs],
            }
        return original(self, attn_metadata_builder, num_reqs)

    get_extra_attn_kwargs._vllm_fl_qwen38_ple = True
    MambaHybridAttnMetadata.get_extra_attn_kwargs = get_extra_attn_kwargs


def _register_compilation_boundaries() -> None:
    from vllm.config.compilation import CompilationConfig

    for op in (
        "vllm::qwen3_8_flash_next_ple_short_conv",
        "vllm::qwen3_8_flash_next_qsa_with_output",
    ):
        if op not in CompilationConfig._attention_ops:
            CompilationConfig._attention_ops.append(op)


def apply_qwen3_8_flash_next_patches() -> bool:
    """Install idempotent config, model and v0.24 metadata registrations."""
    from vllm.model_executor.models import config as model_config
    from vllm.model_executor.models import registry as model_registry
    from vllm.transformers_utils import config as transformers_config

    from vllm_fl.models.qwen3_8_flash_next.config import (
        Qwen3_8FlashNextConfig,
        Qwen3_8FlashNextTextConfig,
    )
    from vllm_fl.patches.gdn_packed_decode import patch_vllm_packed_gdn_beta

    config_registry = transformers_config._CONFIG_REGISTRY
    config_registry.setdefault("qwen3_8_flash_next", Qwen3_8FlashNextConfig)
    config_registry.setdefault("qwen3_8_flash_next_text", Qwen3_8FlashNextTextConfig)
    # Reuse the concrete base classes for checkpoint aliases. Transformers 5
    # serializes inherited composite sub-configs back to dictionaries when an
    # alias subclass changes ``sub_configs``; the base classes preserve typed
    # text/vision configs while retaining the checkpoint's instance
    # ``model_type`` values.
    config_registry.setdefault("qwen4_exp", Qwen3_8FlashNextConfig)
    config_registry.setdefault("qwen4_exp_text", Qwen3_8FlashNextTextConfig)

    module = "vllm_fl.models.qwen3_8_flash_next"
    for architecture, (class_name, verifier, family) in _ARCHITECTURES.items():
        model_config.MODELS_CONFIG_MAP[architecture] = verifier
        target_map = (
            model_registry._MULTIMODAL_MODELS
            if family == "multimodal"
            else model_registry._TEXT_GENERATION_MODELS
        )
        target_map.setdefault(architecture, (module, class_name))
        model_registry._VLLM_MODELS.setdefault(architecture, (module, class_name))
        model_registry.ModelRegistry.register_model(
            architecture, f"{module}:{class_name}"
        )

    _patch_common_attention_token_to_req_cache()
    _patch_ple_metadata_bridge()
    _register_compilation_boundaries()
    patch_vllm_packed_gdn_beta()
    logger.info("Installed Qwen3.8-Flash-Next / Qwen4Exp Day0 runtime support")
    return True


__all__ = [
    "Qwen3_8FlashNextForCausalLMConfig",
    "Qwen3_8FlashNextForConditionalGenerationConfig",
    "apply_native_index_select_policy",
    "apply_qwen3_8_flash_next_patches",
    "needs_native_index_select",
    "should_skip_generic_flaggems_aten",
]
