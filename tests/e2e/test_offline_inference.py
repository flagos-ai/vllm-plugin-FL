# Copyright (c) 2025 BAAI. All rights reserved.
import os

import pytest
import vllm  # noqa: F401
from conftest import VllmRunner
import vllm_flagos

MODELS = [
    # "Qwen/Qwen3-0.6B",
    "/share/project/lms/Qwen3-4B",
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

