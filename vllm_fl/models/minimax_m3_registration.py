# Copyright (c) 2026 BAAI. All rights reserved.
"""Register MiniMax M3 quantization metadata with the paired NVIDIA runtime."""


def add_fused_quantization_mapping(model_cls):
    """Preserve named checkpoint targets after QKV and gate/up fusion."""
    mapping = dict(getattr(model_cls, "packed_modules_mapping", {}))
    mapping.setdefault("qkv_proj", ["q_proj", "k_proj", "v_proj"])
    mapping.setdefault("gate_up_proj", ["gate_proj", "up_proj"])
    model_cls.packed_modules_mapping = mapping


def register_minimax_m3_support() -> bool:
    from vllm.model_executor.models import ModelRegistry
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return False
    supported = ModelRegistry.get_supported_archs()
    registered = False
    for architecture in (
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    ):
        if architecture in supported:
            ModelRegistry.register_model(
                architecture, f"vllm_fl.models.minimax_m3:{architecture}"
            )
            registered = True
    return registered
