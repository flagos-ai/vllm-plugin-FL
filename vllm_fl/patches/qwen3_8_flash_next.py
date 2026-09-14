# Copyright (c) 2026 BAAI. All rights reserved.
"""Runtime registration for the Qwen3.8-Flash-Next / Qwen4Exp Day0 model.

The checkpoint keeps its original ``qwen4_exp`` model types and architecture
name.  This module maps those public names to the plugin-owned implementation
without modifying either the checkpoint or the installed vLLM package.
"""

from __future__ import annotations

import logging
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

_TEXT_MODEL_TYPES = {"qwen3_8_flash_next_text", "qwen4_exp_text"}


def needs_native_index_select(vllm_config: "VllmConfig") -> bool:
    """Return whether FlagGems PLE state row I/O must be disabled.

    FlagGems 5.3/5.4 materializes a contiguous copy of non-contiguous inputs.
    Qwen3.8-Flash-Next exposes its multi-gigabyte PLE state cache as a
    transposed view.  FlagGems index_select is slower than native ATen and can
    OOM.  This check is deliberately model-scoped.
    """

    text_config = vllm_config.model_config.hf_text_config
    return getattr(text_config, "model_type", None) in _TEXT_MODEL_TYPES


def apply_native_index_select_policy(
    vllm_config: "VllmConfig",
    whitelist: Optional[list[str]],
    blacklist: Optional[list[str]],
) -> tuple[Optional[list[str]], Optional[list[str]]]:
    """Merge the model exclusion without overriding explicit whitelists."""

    if not needs_native_index_select(vllm_config):
        return whitelist, blacklist
    required_native_ops = ("index_select",)
    if whitelist:
        conflicts = [op_name for op_name in required_native_ops if op_name in whitelist]
        if conflicts:
            raise ValueError(
                "Qwen3.8-Flash-Next requires native PLE state row I/O; "
                "remove these operators from the FlagGems whitelist: "
                + ", ".join(conflicts)
            )
        return whitelist, blacklist
    merged = list(blacklist or [])
    for op_name in required_native_ops:
        if op_name not in merged:
            merged.append(op_name)
    return whitelist, merged


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


def _patch_mamba_group_uniformity() -> None:
    """Allow heterogeneous MambaSpecs (vLLM 0.26 compatibility).

    vLLM 0.26's ``get_mamba_groups`` asserts that every MambaSpec in the model
    is identical::

        assert all(mamba_specs[0] == spec for spec in mamba_specs)

    That contract does not hold for this architecture, and did not exist in the
    0.24 tree this port was written against. The model deliberately exposes two
    kinds of Mamba state (see ``get_mamba_specs_from_config``):

      * the GDN spec shared by the 36 linear_attention layers
      * a separate short_conv spec owned by the PLE layer

    They differ only in ``shapes``/``dtypes``. Every consumer of the returned
    spec (``MambaCopyBuffers.create``, ``MambaSpecDecodeGPUContext.create``, and
    the block-size helper) reads only ``block_size`` and, on the speculative
    path, ``num_speculative_blocks`` -- never ``shapes`` or ``dtypes``. So the
    assertion is stricter than the code requires.

    Relax it to check exactly the fields that are actually consumed, and keep
    returning the spec with the largest page so any size-derived bound stays
    conservative.
    """
    from vllm.v1.worker import mamba_utils
    from vllm.v1.kv_cache_interface import MambaSpec

    if getattr(mamba_utils.get_mamba_groups, "_vllm_fl_qwen4_heterogeneous", False):
        return

    def get_mamba_groups(kv_cache_config):
        group_ids: list[int] = []
        specs: list[MambaSpec] = []
        for i, group in enumerate(kv_cache_config.kv_cache_groups):
            if isinstance(group.kv_cache_spec, MambaSpec):
                group_ids.append(i)
                specs.append(group.kv_cache_spec)
        assert len(group_ids) > 0, "no mamba layers in the model"

        # Only the consumed fields must agree. If these ever diverge the
        # downstream buffer math really would be wrong, so still assert.
        for spec in specs:
            assert spec.block_size == specs[0].block_size, (
                "Mamba specs disagree on block_size: "
                f"{[s.block_size for s in specs]}"
            )
            assert (
                spec.num_speculative_blocks == specs[0].num_speculative_blocks
            ), (
                "Mamba specs disagree on num_speculative_blocks: "
                f"{[s.num_speculative_blocks for s in specs]}"
            )

        representative = max(specs, key=lambda s: s.page_size_bytes)
        return group_ids, representative

    get_mamba_groups._vllm_fl_qwen4_heterogeneous = True
    mamba_utils.get_mamba_groups = get_mamba_groups


def _patch_qsa_page_size_padding() -> None:
    """Let the QSA side caches be page-padded (vLLM 0.26 compatibility).

    0.26 added ``unify_kv_cache_spec_page_size``, which reconciles differing
    page sizes across layers. It offers three routes: exact divisibility,
    MambaSpec padding, or -- for AttentionSpecs -- padding when the spec's
    ``indexes_kv_by_block_stride`` is set. The QSA side caches take none of
    them, and 0.24 had no such unification step at all.

    Divisibility is unreachable by tuning ``block_size``. When the checkpoint
    uses MRoPE the raw key cache packs three int64 position axes after the
    128-wide key, giving ``head_size = 128 + 3*4 = 140`` versus 256 (512 for
    K+V) for full attention. The ratio 512/140 = 128/35 is independent of
    ``block_size``, so no block size makes it divide.

    vLLM copies this flag off the backend class inside ``get_kv_cache_spec``
    (see ``v1/worker/gpu/attn_utils.py``), which runs well after import, so
    setting the class attribute is sufficient. Padding only reserves extra
    bytes per page; it does not change the
    ``[pages, block_size, 1, head_dim]`` view the QSA kernels index through,
    which is why this is safe even though the cache is not laid out
    block-major in the vLLM sense.

    Only ``QSAStateBackend`` is touched here -- ``gpu/qsa.py`` is still
    mid-import when the registration hook runs (the package has intentional
    model -> qsa -> model ownership), so it is handled by
    ``_patch_qsa_gpu_backend_when_loaded`` below.
    """
    from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import QSAStateBackend

    if not getattr(QSAStateBackend, "_vllm_fl_qwen4_page_padded", False):
        QSAStateBackend.indexes_kv_by_block_stride = classmethod(lambda cls: True)
        QSAStateBackend._vllm_fl_qwen4_page_padded = True


def _patch_qsa_gpu_backend_when_loaded() -> None:
    """Apply the same page-padding fix to the GPU QSA attention backend.

    That backend defines its own ``indexes_kv_by_block_stride`` override, so it
    does not inherit the base-class fix, and its module cannot be imported from
    the registration hook (circular import). Hook ``AttentionSpec`` creation
    instead: by the time any spec is built every plugin module is loaded, so
    resolve and patch the class then.
    """
    from vllm.v1.worker.gpu import attn_utils

    original = attn_utils.get_kv_cache_spec
    if getattr(original, "_vllm_fl_qwen4_page_padded", False):
        return

    def get_kv_cache_spec(*args, **kwargs):
        import sys as _sys

        mod = _sys.modules.get("vllm_fl.models.qwen3_8_flash_next.gpu.qsa")
        backend = (
            getattr(mod, "Qwen3_8FlashNextQSAAttentionBackend", None) if mod else None
        )
        if backend is not None and not getattr(
            backend, "_vllm_fl_qwen4_page_padded", False
        ):
            backend.indexes_kv_by_block_stride = classmethod(lambda cls: True)
            backend._vllm_fl_qwen4_page_padded = True
        return original(*args, **kwargs)

    get_kv_cache_spec._vllm_fl_qwen4_page_padded = True
    attn_utils.get_kv_cache_spec = get_kv_cache_spec


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

    _patch_ple_metadata_bridge()
    _patch_mamba_group_uniformity()
    _patch_qsa_page_size_padding()
    _patch_qsa_gpu_backend_when_loaded()
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
]
