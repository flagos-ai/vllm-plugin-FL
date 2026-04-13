# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/examples/offline_inference/basic/basic.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

# Enable FlagGems operators
os.environ["USE_FLAGGEMS"] = "1"
# Blacklist FlagGems operators that are incompatible with MoE models
os.environ["VLLM_FL_FLAGOS_BLACKLIST"] = (
    "one_hot,_to_copy,moe_align_block_size,topk,gather,sort,sort_stable"
)

from vllm import LLM, SamplingParams

if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
    # Create an LLM.
    llm = LLM(
        model="google/gemma-4-26B-A4B-it",
        tensor_parallel_size=2,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,
    )

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
