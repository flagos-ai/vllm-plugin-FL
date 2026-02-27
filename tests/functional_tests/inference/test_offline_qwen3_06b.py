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

MODEL_PATH = "/data/models/Qwen/Qwen3-0.6B"
pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason=f"Model not found: {MODEL_PATH}"
)


@pytest.fixture(autouse=True)
def modelscope_env():
    """Set VLLM_USE_MODELSCOPE only for tests in this module."""
    old = os.environ.get("VLLM_USE_MODELSCOPE")
    os.environ["VLLM_USE_MODELSCOPE"] = "True"
    yield
    if old is not None:
        os.environ["VLLM_USE_MODELSCOPE"] = old
    else:
        os.environ.pop("VLLM_USE_MODELSCOPE", None)


@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_models(
    dtype: str,
    max_tokens: int,
    enforce_eager: bool,
) -> None:
    prompt = "The following numbers of the sequence " + ", ".join(
        str(i) for i in range(1024)) + " are:"
    example_prompts = [prompt]

    with VllmRunner(MODEL_PATH,
                    max_model_len=8192,
                    dtype=dtype,
                    enforce_eager=enforce_eager,
                    gpu_memory_utilization=0.7) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
