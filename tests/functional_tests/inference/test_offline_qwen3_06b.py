# Copyright (c) 2025 BAAI. All rights reserved.

"""
Offline inference tests for vllm_fl.
Tests basic model generation with VllmRunner in eager and graph mode.
"""

import os

import pytest
import vllm  # noqa: F401
from .vllm_runner import VllmRunner
import vllm_fl  # noqa: F401

MODELS = [
    # "Qwen/Qwen3-0.6B",
    #"/share/project/lms/Qwen3-4B",
    "/flagops/models/Qwen/Qwen3-0.6B"
]
os.environ["VLLM_USE_MODELSCOPE"] = "True"

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_models(
    model: str,
    dtype: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    prompt = "The following numbers of the sequence " + ", ".join(
        str(i) for i in range(1024)) + " are:"
    example_prompts = [prompt]

    with VllmRunner(model,
                    max_model_len=8192,
                    dtype=dtype,
                    enforce_eager=enforce_eager,
                    gpu_memory_utilization=0.7) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
