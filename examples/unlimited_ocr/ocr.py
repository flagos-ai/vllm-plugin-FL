# Copyright (c) 2026 BAAI. All rights reserved.
"""Transcribe one or more ordered pages with an Unlimited-OCR server."""

import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images", nargs="+", type=Path, help="Page images in reading order"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Unlimited-OCR")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    content = []
    for path in args.images:
        with Image.open(path) as image:
            mime_type = Image.MIME[image.format]
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64," + encoded},
            }
        )
    multiple_pages = len(args.images) > 1
    instruction = "Multi page parsing." if multiple_pages else "document parsing."
    content.append({"type": "text", "text": instruction})
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "add_special_tokens": True,
        "skip_special_tokens": False,
        "temperature": 0,
        "seed": 42,
        "max_tokens": args.max_tokens,
        "vllm_xargs": {
            "ngram_size": 35,
            "window_size": 1024 if multiple_pages else 128,
        },
    }
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    sys.stdout.write(result["choices"][0]["message"]["content"] + "\n")


if __name__ == "__main__":
    main()
