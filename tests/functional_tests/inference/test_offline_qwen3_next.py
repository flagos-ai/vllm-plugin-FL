# Copyright (c) 2025 BAAI. All rights reserved.

"""
Offline inference tests for Qwen3-Next model.
Tests basic generation and knowledge validation with tensor parallelism.
"""

import os

import pytest
from vllm import LLM, SamplingParams

MODEL_PATH = "/data/models/Qwen/Qwen3-Next-80B-A3B-Instruct"
pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason=f"Model not found: {MODEL_PATH}"
)


@pytest.fixture(scope="module")
def llm_instance():
    return LLM(
        model=MODEL_PATH,
        max_num_batched_tokens=16384,
        max_num_seqs=1024,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=8,
        trust_remote_code=True,
    )


@pytest.fixture
def default_params():
    return SamplingParams(max_tokens=10, temperature=0.0)


def test_basic_generation(llm_instance, default_params):
    prompt = "Hello, my name is"
    outputs = llm_instance.generate([prompt], default_params)

    assert len(outputs) > 0
    generated_text = outputs[0].outputs[0].text
    assert len(generated_text) > 0
    print(f"\n[Output]: {generated_text}")


@pytest.mark.parametrize(
    "prompt, expected_part",
    [
        ("The capital of France is", "Paris"),
    ],
)
def test_specific_logic(llm_instance, default_params, prompt, expected_part):
    outputs = llm_instance.generate([prompt], default_params)
    generated_text = outputs[0].outputs[0].text
    assert expected_part in generated_text
