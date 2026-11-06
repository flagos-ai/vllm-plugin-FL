# Copyright (c) 2026 BAAI. All rights reserved.
"""Runtime registration for the Qwen3.8-Flash-Next / Qwen4Exp Day0 model.

The checkpoint keeps its original ``qwen4_exp`` model types and architecture
name.  This module maps those public names to the plugin-owned implementation
without modifying either the checkpoint or the installed vLLM package.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = logging.getLogger(__name__)


def _strip_mrope(model_config: ModelConfig) -> None:
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
    def get_flag_gems_blacklist(*, vendor_name: str | None) -> list[str]:
        if vendor_name != "nvidia":
            return []
        # Keep history gather and QSA selection native. The generic
        # repeat-interleave path synchronizes to infer output_size; dense
        # GEMM and stable sorting also regress on the measured H100 workload.
        return [
            "index_select",
            "repeat_interleave_tensor",
            "repeat_interleave_self_tensor",
            "linear",
            "mm",
            "mm_out",
            "addmm",
            "addmm_out",
            "addmm_",
            "addmm_dtype",
            "addmm_dtype_out",
            "sort",
            "sort_stable",
        ]

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
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
        if (
            bool(text_config.ple_layer_ids)
            and parallel_config.pipeline_parallel_size > 1
        ):
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


class Qwen3_8FlashNextForCausalLMConfig(Qwen3_8FlashNextForConditionalGenerationConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
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
            torch.arange(query_lens.shape[0], dtype=torch.int32, device=buffer.device),
            query_lens,
            output_size=num_mapped_tokens,
        )
        buffer[:num_mapped_tokens].copy_(mapping)
        if num_mapped_tokens < num_tokens:
            buffer[num_mapped_tokens:num_tokens].zero_()
        cached = buffer[: max(num_mapped_tokens, num_tokens)]
        self._token_to_req_indices_cache = cached
        return cached[:num_tokens]

    CommonAttentionMetadata.token_to_req_indices = token_to_req_indices


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
    from vllm.model_executor.models import (
        config as model_config,
        registry as model_registry,
    )
    from vllm.transformers_utils import config as transformers_config

    from vllm_fl.models.qwen3_8_flash_next.config import (
        Qwen3_8FlashNextConfig,
        Qwen3_8FlashNextTextConfig,
    )

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
    logger.info("Installed Qwen3.8-Flash-Next / Qwen4Exp Day0 runtime support")
    return True


__all__ = [
    "Qwen3_8FlashNextForCausalLMConfig",
    "Qwen3_8FlashNextForConditionalGenerationConfig",
    "apply_qwen3_8_flash_next_patches",
]
