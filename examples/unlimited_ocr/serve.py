# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve Unlimited-OCR BF16 on the FL Ascend platform."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

CHAT_TEMPLATE = """\
{{- bos_token }}
{%- for message in messages %}
    {%- if message['content'] is string %}
        {{- message['content'] }}
    {%- else %}
        {%- for content in message['content'] | selectattr('type', 'equalto', 'image') %}
            {{- '<image>' }}
        {%- endfor %}
        {%- for content in message['content'] | selectattr('type', 'equalto', 'text') %}
            {{- content['text'] }}
        {%- endfor %}
    {%- endif %}
    {%- if message['role'] == 'assistant' %}
        {{- eos_token }}
    {%- endif %}
{%- endfor %}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--tp", type=int, choices=(1, 2), default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--rswa-mask-mib", type=int, default=2048)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", args.devices):
        parser.error("--devices must contain comma-separated NPU device IDs")
    devices = tuple(map(int, args.devices.split(",")))
    if len(set(devices)) != len(devices) or len(devices) != args.tp:
        parser.error("--devices must specify distinct IDs, one per TP rank")
    seqs, budget, mask = (
        args.max_num_seqs,
        args.max_num_batched_tokens,
        args.rswa_mask_mib,
    )
    if not (1 <= seqs <= budget and 1 <= mask <= 8192 and seqs * budget <= mask * 32):
        parser.error(
            "R-SWA requires 1 <= sequences <= budget and sequences * budget <= mask MiB * 32"
        )
    if not args.eager and os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
        parser.error("FULL_DECODE_ONLY requires ASCEND_LAUNCH_BLOCKING=0")

    vllm = Path(sys.executable).with_name("vllm")
    if not vllm.is_file():
        parser.error(f"Install the pinned vLLM in the current environment ({vllm})")
    env = {
        **os.environ,
        "ASCEND_RT_VISIBLE_DEVICES": args.devices,
        "VLLM_PLUGINS": "fl",
        "VLLM_FL_VENDOR": "ascend",
        "USE_FLAGGEMS": os.environ.get("USE_FLAGGEMS", "0"),
        "VLLM_FL_PREFER_ENABLED": "1",
        "VLLM_FL_STRICT": "1",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_FL_RSWA_MASK_MAX_MIB": str(mask),
    }
    env.setdefault("OMP_NUM_THREADS", "1")
    argv = [
        str(vllm),
        "serve",
        args.model,
        "--served-model-name=Unlimited-OCR",
        f"--host={args.host}",
        f"--port={args.port}",
        "--dtype=bfloat16",
        f"--tensor-parallel-size={args.tp}",
        "--disable-custom-all-reduce",
        "--max-model-len=32768",
        f"--max-num-seqs={seqs}",
        f"--max-num-batched-tokens={budget}",
        "--enable-chunked-prefill",
        "--no-async-scheduling",
        "--gpu-memory-utilization=0.80",
        "--no-enable-prefix-caching",
        f"--chat-template={CHAT_TEMPLATE}",
        "--mm-processor-cache-gb=0",
        '--limit-mm-per-prompt={"image":1}',
        '--mm-processor-kwargs={"image_size":640,"base_size":1024,"crop_mode":true}',
        "--logits-processors=vllm_fl.models.ocr_ngram:NGramPerReqLogitsProcessorFL",
    ]
    if args.eager:
        argv.append("--enforce-eager")
    else:
        config = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": sorted(
                {seqs, *(2**k for k in range(seqs.bit_length()))}
            ),
        }
        argv.extend(("--compilation-config", json.dumps(config)))
    os.execve(vllm, argv, env)


if __name__ == "__main__":
    main()
