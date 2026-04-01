# Copyright (c) 2025 BAAI. All rights reserved.

"""
Test that MoE model classes can be successfully imported from
vllm.model_executor.models.

These model modules and classes are required by downstream training
frameworks for weight-loader patching. If any import fails, the model
is silently skipped, which may cause subtle runtime issues. This test
ensures all expected model modules and classes are accessible.
"""

import importlib

import pytest

# Each entry: (module_path, class_names)
MODEL_IMPORTS = [
    (
        "vllm.model_executor.models.deepseek_v2",
        ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"],
    ),
    (
        "vllm.model_executor.models.mixtral",
        ["MixtralForCausalLM"],
    ),
    (
        "vllm.model_executor.models.qwen2_moe",
        ["Qwen2MoeForCausalLM"],
    ),
    (
        "vllm.model_executor.models.qwen3_moe",
        ["Qwen3MoeForCausalLM"],
    ),
    (
        "vllm.model_executor.models.qwen3_vl_moe",
        ["Qwen3MoeLLMForCausalLM"],
    ),
    (
        "vllm.model_executor.models.qwen3_next",
        ["Qwen3NextForCausalLM"],
    ),
    (
        "vllm.model_executor.models.kimi_vl",
        ["KimiVLForConditionalGeneration"],
    ),
]


def _build_test_params():
    """Flatten (module, class) pairs for parametrize."""
    params = []
    for module_path, class_names in MODEL_IMPORTS:
        for class_name in class_names:
            params.append(
                pytest.param(
                    module_path,
                    class_name,
                    id=f"{module_path.split('.')[-1]}.{class_name}",
                )
            )
    return params


class TestMoeModelImports:
    """Verify that all expected MoE model classes are importable."""

    @pytest.mark.parametrize("module_path, class_name", _build_test_params())
    def test_import_model_class(self, module_path, class_name):
        """Each model class should be importable and be a valid class."""
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        assert cls is not None, (
            f"{class_name} not found in {module_path}. "
            f"Available attrs: {[a for a in dir(mod) if not a.startswith('_')]}"
        )
        assert isinstance(cls, type), (
            f"{module_path}.{class_name} is {type(cls).__name__}, expected a class"
        )
