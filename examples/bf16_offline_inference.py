# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/examples/offline_inference/basic/basic.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.platforms import current_platform

print(f"Current Platform: {current_platform}")
print(f"Platform Type: {type(current_platform)}")

# Check if FlagGems is being used
if "USE_FLAGGEMS" in os.environ:
    print(f"USE_FLAGGEMS: {os.environ['USE_FLAGGEMS']}")

if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "What is 2+2? The answer is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=30, temperature=0.0)
    # Create an LLM.
    llm = LLM(
        model="/mine/DeepSeek-V4-Flash-BF16/",
        max_num_batched_tokens=16384,
        max_num_seqs=2048,
        tensor_parallel_size=8,
        kv_cache_dtype="bfloat16",
        enable_expert_parallel=True,
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, cudagraph_mode="FULL_DECODE_ONLY"
        ),
        tokenizer_mode="deepseek_v4",
        safetensors_load_strategy="prefetch",
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
        gpu_memory_utilization=0.8,
    )

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm
    torch.cuda.empty_cache()

    print("\n Reasoning complete, resources cleared.")
