# Copyright (c) 2026 BAAI. All rights reserved.

"""MiniMax-M3.1-Flash offline text generation with official weights.

Targets NVIDIA H100/H200 TP8 with an M3.1-capable vLLM runtime.
See docs/minimax-m31.md for version requirements and validation status.
"""

from vllm import LLM, SamplingParams


if __name__ == "__main__":
    sampling_params = SamplingParams(max_tokens=512, temperature=1.0, top_p=0.95)

    # Official checkpoint: MXFP8 dense layers and NVFP4 MoE experts.
    # Hopper uses FP8 KV/indexer caches and the Marlin MoE backend.
    llm = LLM(
        model="MiniMaxAI/MiniMax-M3.1-Flash",
        tensor_parallel_size=8,
        trust_remote_code=True,
        block_size=128,
        kv_cache_dtype="fp8",
        attention_config={"indexer_kv_dtype": "fp8"},
        moe_backend="marlin",
        disable_hybrid_kv_cache_manager=True,
        language_model_only=True,
        max_model_len=8192,
        max_num_seqs=1,
        enforce_eager=True,
    )

    messages = [{"role": "user", "content": "Please introduce MiniMax-M3.1-Flash."}]
    outputs = llm.chat(messages, sampling_params)
    for output in outputs:
        print(f"Generated: {output.outputs[0].text!r}")
