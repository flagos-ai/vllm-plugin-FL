# Copyright (c) 2026 BAAI. All rights reserved.
"""Offline inference example for Xing4.1-Preview with vllm-plugin-FL.

Xing4.1-Preview is a DeepSeek-V3.2 style model (MLA + DSA) quantised with
Compressed-Tensors W8A8. vLLM already implements the architecture, so enabling
this model only requires:

* the ``Xing4.1-Preview`` -> ``DeepseekV3Config`` mapping, and
* the ``xing4_0`` reasoning / tool parsers,

both provided by ``vllm_fl.models.xing4_1``.

Usage::

    VLLM_PLUGINS=fl python examples/xing4_1_offline_inference.py \
        --model /path/to/Xing4.1-Preview-W8A8

Note: the chat template injects the `` thinking`` marker into the prompt, so
``enable_thinking=True`` must be requested per conversation; otherwise the
reasoning parser leaves ``content`` empty.
"""

from __future__ import annotations

import argparse

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Local path or repository id of the Xing4.1-Preview checkpoint.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        # The checkpoint advertises model_type "Xing4.1-Preview"; the plugin maps
        # it onto vLLM's DeepSeek config. Overriding it here keeps the flag
        # explicit for readers of this example.
        hf_overrides={"model_type": "deepseek_v32"},
    )

    sampling = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    conversations = [
        [
            {
                "role": "user",
                "content": "What is 27 times 27? Answer with the number only.",
            }
        ],
        [{"role": "user", "content": "用一句话介绍你自己。"}],
    ]

    outputs = llm.chat(
        conversations,
        sampling_params=sampling,
        chat_template_kwargs={"enable_thinking": True},
    )

    for i, output in enumerate(outputs, 1):
        text = output.outputs[0].text
        print(f"--- sample {i} (finish_reason={output.outputs[0].finish_reason}) ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
