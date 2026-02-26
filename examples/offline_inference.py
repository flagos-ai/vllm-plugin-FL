# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/examples/offline_inference/basic/basic.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
import os
import torch
from vllm.config.compilation import CompilationConfig

# Check Platform
from vllm.platforms import current_platform
print(f"Current Platform: {current_platform}")
print(f"Platform Type: {type(current_platform)}")

# Check if FlagGems is being used
if "USE_FLAGGEMS" in os.environ:
    print(f"USE_FLAGGEMS: {os.environ['USE_FLAGGEMS']}")

if __name__ == '__main__':
    prompts = [
        "Hello, my name is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
    # Create an LLM.
    llm = LLM(model="Qwen/Qwen3-4B", max_num_batched_tokens=16384, max_num_seqs=2048)

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm
    torch.cuda.empty_cache()
    
    print("\n Reasoning complete, resources cleared.")
