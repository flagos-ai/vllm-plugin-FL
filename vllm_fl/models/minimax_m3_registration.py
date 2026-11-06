# Copyright (c) 2026 BAAI. All rights reserved.
"""Lazy MiniMax M3 registration and fused projection metadata."""

from threading import RLock

from vllm.model_executor.models.config import VerifyAndUpdateConfig

_ARCHITECTURES = (
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
)
_METAX_ARCHITECTURES = (*_ARCHITECTURES, "MiniMaxM3MTP")
_FUSED_PROJECTIONS = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
_REGISTRATION_LOCK = RLock()


def add_fused_quantization_mapping(model_cls: type) -> None:
    """Fill missing fusion metadata without replacing upstream mappings."""
    existing = getattr(model_cls, "packed_modules_mapping", {})
    missing = {
        name: list(parts)
        for name, parts in _FUSED_PROJECTIONS.items()
        if name not in existing
    }
    if missing:
        model_cls.packed_modules_mapping = {**existing, **missing}


class MiniMaxM3MetaXConfig(VerifyAndUpdateConfig):
    """Validate the opt-in BF16 profile without replacing the FL platform."""

    @staticmethod
    def verify_and_update_config(vllm_config) -> None:
        import torch

        from vllm.platforms import current_platform

        from vllm_fl.minimax_m3_config import load

        profile = load()
        if getattr(current_platform, "vendor_name", None) != "metax":
            raise ValueError("The MiniMax M3 optimization profile requires MetaX")
        model = vllm_config.model_config
        if model is not None:
            architectures = set(model.hf_config.architectures or [])
            if not architectures or not architectures <= set(_METAX_ARCHITECTURES):
                raise ValueError(
                    "This optimization profile is restricted to MiniMax M3"
                )
            if model.dtype != torch.bfloat16 or model.quantization is not None:
                raise ValueError("The MiniMax M3 optimization profile requires BF16")
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != profile.framework.tensor_parallel_size:
            raise ValueError("The MiniMax M3 optimization profile requires TP16")
        parallel.worker_cls = "vllm_fl.worker.minimax_m3.MiniMaxWorkerFL"


def _register_metax_models(registry) -> None:
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from vllm_fl.models.minimax_m3_metax.bindings import install

    supported = registry.get_supported_archs()
    if not set(_METAX_ARCHITECTURES) <= set(supported):
        raise RuntimeError(
            "The pinned vLLM build must provide MiniMax M3 architectures"
        )
    missing = object()
    with _REGISTRATION_LOCK:
        previous_models = {
            name: registry.models.get(name, missing) for name in _METAX_ARCHITECTURES
        }
        previous_configs = {
            name: MODELS_CONFIG_MAP.get(name, missing) for name in _METAX_ARCHITECTURES
        }
        try:
            for name in _METAX_ARCHITECTURES:
                module = (
                    "vllm.models.minimax_m3"
                    if name == "MiniMaxM3MTP"
                    else "vllm_fl.models.minimax_m3"
                )
                current = registry.models.get(name)
                if (
                    getattr(current, "module_name", None) != module
                    or getattr(current, "class_name", None) != name
                ):
                    registry.register_model(name, f"{module}:{name}")
                MODELS_CONFIG_MAP[name] = MiniMaxM3MetaXConfig
            # Model-module bindings are explicit, version checked and atomic.
            # Lazy class export no longer installs hooks as an import side effect.
            install()
        except Exception:
            for mapping, previous in (
                (registry.models, previous_models),
                (MODELS_CONFIG_MAP, previous_configs),
            ):
                for name, value in previous.items():
                    if value is missing:
                        mapping.pop(name, None)
                    else:
                        mapping[name] = value
            raise


def register_minimax_m3_support() -> None:
    """Extend the shared CUDA registration with the explicit MetaX profile."""
    from vllm.model_executor.models import ModelRegistry
    from vllm.platforms import current_platform

    from vllm_fl.minimax_m3_config import is_enabled

    if is_enabled():
        if getattr(current_platform, "vendor_name", None) != "metax":
            raise ValueError("The MiniMax M3 optimization profile requires MetaX")
        _register_metax_models(ModelRegistry)
        return

    # Preserve the upstream registration and quantization metadata path.
    if not current_platform.is_cuda():
        return

    supported = ModelRegistry.get_supported_archs()
    for architecture in _ARCHITECTURES:
        if architecture in supported:
            ModelRegistry.register_model(
                architecture, f"vllm_fl.models.minimax_m3:{architecture}"
            )
