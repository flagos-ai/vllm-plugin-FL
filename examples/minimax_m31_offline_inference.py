# Copyright (c) 2026 BAAI. All rights reserved.

"""MiniMax-M3.1-Flash W8A8 offline inference on T-Head (Zhenwu).

Run in the platform's configured FlagOS environment:
    VLLM_PLUGINS=fl python examples/minimax_m31_offline_inference.py

See docs/minimax-m31.md for model downloads and platform requirements.
"""

from vllm import LLM, SamplingParams


if __name__ == "__main__":
    prompts = ["Please introduce MiniMax-M3.1-Flash."]
    sampling_params = SamplingParams(max_tokens=512, temperature=0.0)

    # T-Head W8A8 Express example from the official FlagOS deployment guide.
    # For local weights, replace model with the downloaded model directory.
    llm = LLM(
        model="FlagRelease/MiniMax-M3.1-Flash-W8A8-zhenwu-FlagOS-Express",
        tensor_parallel_size=8,
        trust_remote_code=True,
    )

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
