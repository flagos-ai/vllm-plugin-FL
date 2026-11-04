# Copyright (c) 2026 BAAI. All rights reserved.

"""Check the Sunrise empty-wheel runtime; optionally run offline inference."""

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from runtime_constraints import verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-dir", type=Path, default=Path("/opt/sunrise-stack"))
    parser.add_argument("--vllm-version", default="0.32.0")
    parser.add_argument("--min-devices", type=int, default=1)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1073741824)
    args = parser.parse_args()
    if args.min_devices < 1 or args.tensor_parallel_size < 1:
        parser.error("Device counts must be positive")
    verify(args.stack_dir, args.vllm_version)
    if sys.version_info[:2] == (3, 10):
        receipt = args.stack_dir / "python310-compat.json"
        if not receipt.is_file():
            raise RuntimeError(
                "Python 3.10 requires the explicit compatibility-build receipt"
            )
        print("Third-party Python 3.10 compatibility build:")
        print(receipt.read_text(encoding="utf-8"))

    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "1") != "1":
        raise RuntimeError("Sunrise acceptance requires VLLM_USE_V2_MODEL_RUNNER=1")

    import torch

    importlib.import_module("torch_ptpu")  # Register PTPU before FlagGems detection.

    import flag_gems
    import triton

    from vllm.platforms import current_platform

    import vllm_fl

    if current_platform.device_type != "ptpu":
        raise RuntimeError(f"Expected Sunrise/PTPU, got {current_platform.device_type}")
    if not torch.ptpu.is_available():
        raise RuntimeError("No Sunrise device is available")
    required = max(args.min_devices, args.tensor_parallel_size)
    count = torch.ptpu.device_count()
    if count < required:
        raise RuntimeError(f"Need {required} visible devices, found {count}")
    for index in range(required):
        device = f"ptpu:{index}"
        left = torch.ones((16,), device=device)
        result = (left + left).cpu()
        torch.testing.assert_close(result, torch.full((16,), 2.0))
    print(
        json.dumps(
            {
                "python": sys.version,
                "torch": torch.__version__,
                "torch_ptpu": importlib.metadata.version("torch-ptpu"),
                "vllm": importlib.metadata.version("vllm"),
                "triton_path": triton.__file__,
                "flag_gems_path": flag_gems.__file__,
                "plugin_path": vllm_fl.__file__,
                "visible_devices": os.environ.get("TANG_VISIBLE_DEVICES"),
                "device_count": count,
            },
            indent=2,
        )
    )
    if args.model is not None:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(args.model),
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            max_num_seqs=2,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        )
        if not llm.llm_engine.vllm_config.use_v2_model_runner:
            raise RuntimeError("The constructed engine did not select Model Runner V2")
        print("Sunrise acceptance selected Model Runner V2")
        outputs = llm.generate(
            ["The capital of France is", "1 + 1 ="],
            SamplingParams(temperature=0, max_tokens=24),
        )
        if len(outputs) != 2 or any(
            not output.outputs or not output.outputs[0].text.strip()
            for output in outputs
        ):
            raise RuntimeError("Inference returned missing or empty text")
        for output in outputs:
            print(json.dumps({"prompt": output.prompt, "text": output.outputs[0].text}))
    print("Sunrise runtime checks passed")


if __name__ == "__main__":
    main()
