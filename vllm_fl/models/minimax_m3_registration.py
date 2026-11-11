# Copyright (c) 2026 BAAI. All rights reserved.
"""Lazy MiniMax M3 registration and fused projection metadata."""

_ARCHITECTURES = (
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
)
_FUSED_PROJECTIONS = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


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


def register_minimax_m3_support() -> None:
    """Keep vendor model imports lazy and register available architectures."""
    from vllm.model_executor.models import ModelRegistry
    from vllm.platforms import current_platform

    if (
        getattr(current_platform, "vendor_name", None) == "ascend"
        and current_platform.device_type == "npu"
    ):
        module = "vllm_fl.models.minimax_m3_ascend"
    elif current_platform.is_cuda():
        module = "vllm_fl.models.minimax_m3"
    else:
        return

    supported = ModelRegistry.get_supported_archs()
    for architecture in _ARCHITECTURES:
        if architecture in supported:
            ModelRegistry.register_model(
                architecture, f"{module}:{architecture}"
            )
