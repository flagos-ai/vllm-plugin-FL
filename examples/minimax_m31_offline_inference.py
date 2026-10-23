# Copyright (c) 2026 BAAI. All rights reserved.

"""MiniMax-M3.1-Flash W8A8 offline inference using existing local weights.

See docs/minimax-m31.md for model paths and environment requirements.
"""

from vllm import LLM, SamplingParams


if __name__ == "__main__":
    # Existing FlagOS W8A8 checkpoint; change this to your local model path.
    llm = LLM(
        model="/models/MiniMax-M3.1-Flash-W8A8-FlagOS-20261019",
        tensor_parallel_size=8,
        trust_remote_code=True,
        block_size=128,
        dtype="bfloat16",
        disable_hybrid_kv_cache_manager=True,
        max_model_len=2048,
        max_num_seqs=1,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )

    messages = [{"role": "user", "content": "What is 2 + 2? Answer briefly."}]
    outputs = llm.chat(messages, SamplingParams(max_tokens=64, temperature=0.0))
    for output in outputs:
        print(f"Generated: {output.outputs[0].text!r}")
