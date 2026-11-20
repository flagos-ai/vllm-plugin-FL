# Copyright (c) 2026 BAAI. All rights reserved.
"""Lazy BGE-M3 model and IO entry-point registration."""


def register_bge_m3_support():
    from vllm.model_executor.models import ModelRegistry

    architecture = "BgeM3EmbeddingModelFL"
    if architecture not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            architecture, "vllm_fl.models.bge_m3.model:BgeM3EmbeddingModelFL"
        )


def register_bge_m3_processor():
    return "vllm_fl.models.bge_m3.io_processor.BgeM3IOProcessor"
