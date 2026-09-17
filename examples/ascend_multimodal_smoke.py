# Copyright (c) 2026 BAAI. All rights reserved.
"""Check text and generated-image responses from a local Qwen3.5/3.6 model.

Run inside the prepared Ascend empty-vLLM environment with assigned NPUs.
This is a correctness smoke check, not a performance benchmark.
"""

import argparse
import base64
import json
import time
from io import BytesIO
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory", type=float, default=0.5)
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()
    from PIL import Image, ImageDraw

    from vllm import LLM, SamplingParams

    started = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=2,
        max_model_len=args.max_model_len,
        max_num_seqs=2,
        max_num_batched_tokens=args.max_model_len,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.memory,
        limit_mm_per_prompt={"image": 1, "video": 0},
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        async_scheduling=False,
        trust_remote_code=True,
    )
    cases = [
        (
            "text",
            [
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer with the city name.",
                }
            ],
            ["paris"],
        ),
    ]
    image = Image.new("RGB", (300, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 50, 250, 150), fill="blue")
    draw.text((90, 80), "Hello VLM", fill="yellow")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    cases.append(
        (
            "image",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": uri}},
                        {
                            "type": "text",
                            "text": "Describe all visible text, colors, and shapes in English.",
                        },
                    ],
                }
            ],
            ["hello vlm", "blue", "rectangle"],
        )
    )
    results = []
    for name, messages, expected in cases:
        outputs = llm.chat(
            messages,
            SamplingParams(temperature=0.0, max_tokens=256),
            chat_template_kwargs={"enable_thinking": False},
        )
        result = outputs[0].outputs[0]
        record = {
            "case": name,
            "text": result.text,
            "tokens": list(result.token_ids),
            "expected": expected,
            "passed": all(x in result.text.lower() for x in expected),
        }
        results.append(record)
        print("CASE_RESULT", json.dumps(record, ensure_ascii=False), flush=True)
        Path(args.output).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "tp": 2,
                    "seconds": time.time() - started,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        assert record["passed"], record


if __name__ == "__main__":
    main()
