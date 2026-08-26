# Copyright (c) 2025 BAAI. All rights reserved.

"""CLI smoke test for image-capable P0 models."""

from __future__ import annotations

import argparse

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.multimodal.utils import encode_image_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-runner", choices=("v1", "v2"), required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--asset", default="stop_sign")
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--expected-text", default="")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--speculative-method", default="")
    parser.add_argument("--num-speculative-tokens", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )
    if args.speculative_method:
        llm_kwargs["speculative_config"] = {
            "method": args.speculative_method,
            "num_speculative_tokens": args.num_speculative_tokens,
            "max_model_len": args.max_model_len,
        }

    llm = LLM(**llm_kwargs)
    try:
        actual_runner = (
            "v2" if llm.llm_engine.vllm_config.use_v2_model_runner else "v1"
        )
        print(f"P0_MODEL_RUNNER={actual_runner}")
        assert actual_runner == args.expected_runner, (
            f"Expected {args.expected_runner}, got {actual_runner}"
        )

        image_url = encode_image_url(ImageAsset(args.asset).pil_image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        outputs = llm.chat(
            messages,
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
            ),
        )
        text = outputs[0].outputs[0].text.strip()
        print(f"P0_MULTIMODAL_OUTPUT={text!r}")
        assert text, "Model returned an empty multimodal response"
        if args.expected_text:
            assert args.expected_text.casefold() in text.casefold(), (
                f"Expected {args.expected_text!r} in output: {text!r}"
            )
    finally:
        del llm
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
