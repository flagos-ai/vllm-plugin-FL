# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/examples/offline_inference/basic/basic.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal evas (LPU) offline inference example.

The four engine knobs below are not style choices. On LPU each of the
defaults costs either a failed init or minutes of wasted warmup:

* ``max_model_len`` -- left unset, the engine reserves KV for the
  checkpoint's native window; an 8B model then needs ~5 GiB per request and
  init fails at a 0.5-0.6 GPU memory utilisation.
* ``max_num_batched_tokens`` -- also the size of the profiling dummy
  forward at init; 16384 burns minutes before the first real token.
* ``enable_flashinfer_autotune`` -- vLLM's O2 default turns FlashInfer
  autotuning on, and LPU has no FlashInfer; it is pure warmup cost.
* ``enforce_eager`` -- graph capture is not part of this backend yet.

Run with the LPU environment sourced first, e.g.
``python examples/evas_offline_inference.py``.  On an offline host, point
``FL_MODEL_ROOT`` at the local checkpoint directory and the hub id resolves
to ``<FL_MODEL_ROOT>/Qwen3-0.6B``.
"""

import os

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform


def _resolve_model(name: str) -> str:
    """Map a bare checkpoint id onto a local tree when ``FL_MODEL_ROOT`` is set.

    LPU hosts are normally offline and keep their checkpoints in one directory,
    so ``Qwen/Qwen3-0.6B`` has to resolve to ``<FL_MODEL_ROOT>/Qwen3-0.6B``.
    An absolute path, an unset ``FL_MODEL_ROOT``, or a name with no matching
    directory is returned unchanged, so the plain hub path still works.
    """
    root = os.environ.get("FL_MODEL_ROOT")
    if not root or os.path.isabs(name):
        return name
    local = os.path.join(root, name.split("/")[-1])
    return local if os.path.isdir(local) else name


print(f"Current Platform: {current_platform}")
print(f"Platform Type: {type(current_platform)}")

# Check if FlagGems is being used
if "USE_FLAGGEMS" in os.environ:
    print(f"USE_FLAGGEMS: {os.environ['USE_FLAGGEMS']}")

if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
    # Create an LLM.
    llm = LLM(
        model=_resolve_model("Qwen/Qwen3-0.6B"),
        max_model_len=2048,
        max_num_batched_tokens=512,
        enable_flashinfer_autotune=False,
        enforce_eager=True,
    )

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm

    print("\nReasoning complete, resources cleared.")
