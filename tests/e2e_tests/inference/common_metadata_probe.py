# Copyright (c) 2026 BAAI. All rights reserved.
"""Compare metadata policies in separate full worker processes, without downloads.

Run against an installed plugin wheel; this script does not add the checkout's
package root to PYTHONPATH. Artifacts include the local checkpoint, exact token
IDs/logprobs, producer counters and one log per policy.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


class MetadataProbe:
    """Named worker RPC avoids enabling pickle-based function serialization."""

    def metadata_state(self):
        runner = self.model_runner
        helper = runner.common_attention_metadata_graph
        return {
            "policy": runner.common_metadata_policy.mode,
            "async_scheduling": runner.use_async_scheduling,
            "captures": helper.captures if helper else 0,
            "replays": helper.replays if helper else 0,
            "eager_calls": helper.eager_calls if helper else 0,
        }


def make_model(path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    torch.manual_seed(17)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    LlamaForCausalLM(config).to(torch.float16).save_pretrained(path)
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({f"w{i}": i for i in range(4, 128)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    ).save_pretrained(path)


def run_worker(args):
    from vllm import LLM, SamplingParams

    root = args.output_dir
    graph_mode = args.cudagraph_mode
    llm = LLM(
        worker_extension_cls="common_metadata_probe.MetadataProbe",
        model=str(root / "model"),
        dtype="half",
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.12,
        enable_prefix_caching=True,
        async_scheduling=True,
        disable_custom_all_reduce=True,
        enforce_eager=graph_mode == "NONE",
        seed=11,
        compilation_config={
            "mode": "VLLM_COMPILE" if graph_mode == "PIECEWISE" else "NONE",
            "cudagraph_mode": graph_mode,
            "cudagraph_capture_sizes": [1, 2, 4],
            "max_cudagraph_capture_size": 4,
        },
    )
    try:
        params = SamplingParams(
            temperature=0, max_tokens=12, ignore_eos=True, logprobs=5
        )
        prompts = [
            {"prompt_token_ids": [1] + [4 + (i + j) % 100 for j in range(n)]}
            for i, n in enumerate([3, 8, 17, 41, 3, 32, 8, 17])
        ]
        results = []
        for batch in (prompts, list(reversed(prompts))):
            outputs = llm.generate(batch, params, use_tqdm=False)
            results.append(
                [
                    {
                        "prompt": output.prompt_token_ids,
                        "tokens": output.outputs[0].token_ids,
                        "logprobs": [
                            {str(k): v.logprob for k, v in step.items()}
                            for step in output.outputs[0].logprobs
                        ],
                    }
                    for output in outputs
                ]
            )
        states = llm.collective_rpc("metadata_state")
        (root / f"{args.worker_mode}.json").write_text(
            json.dumps({"results": results, "states": states}, indent=2)
        )
        for state in states:
            assert state["policy"] == args.worker_mode, state
            if args.worker_mode == "graph" and graph_mode != "NONE":
                assert state["captures"] > 0 and state["replays"] > 0, state
            if args.worker_mode == "eager":
                assert state["replays"] == 0 and state["eager_calls"] > 0, state
    finally:
        llm.llm_engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cudagraph-mode",
        choices=["NONE", "FULL_DECODE_ONLY", "PIECEWISE", "FULL"],
        default="FULL_DECODE_ONLY",
    )
    parser.add_argument("--worker-mode", choices=["stock", "eager", "graph"])
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.worker_mode:
        run_worker(args)
        return

    # Refuse to overwrite an earlier validation or its checkpoint.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    make_model(args.output_dir / "model")
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = "fl"
    env["USE_FLAGGEMS"] = "0"  # Isolate metadata from ATen MM registration.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")]
    )
    runs = {}
    for mode in ("stock", "eager", "graph"):
        env["VLLM_FL_COMMON_ATTENTION_METADATA"] = mode
        with (args.output_dir / f"{mode}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(args.output_dir),
                    "--cudagraph-mode",
                    args.cudagraph_mode,
                    "--worker-mode",
                    mode,
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        runs[mode] = json.loads((args.output_dir / f"{mode}.json").read_text())
    # Separate Inductor compilations can differ by a few FP32 log-softmax
    # ULPs. Keep token IDs and top-k identities exact; only PIECEWISE permits
    # this small absolute logprob tolerance. Metadata tensor tests stay exact.
    atol = 1e-5 if args.cudagraph_mode == "PIECEWISE" else 0.0
    max_delta = 0.0
    for mode in ("eager", "graph"):
        for batch, reference in zip(
            runs[mode]["results"], runs["stock"]["results"], strict=True
        ):
            for actual, expected in zip(batch, reference, strict=True):
                assert actual["prompt"] == expected["prompt"], mode
                assert actual["tokens"] == expected["tokens"], mode
                for step, ref_step in zip(
                    actual["logprobs"], expected["logprobs"], strict=True
                ):
                    assert step.keys() == ref_step.keys(), mode
                    for token, value in step.items():
                        assert math.isfinite(value) and math.isfinite(ref_step[token])
                        delta = abs(value - ref_step[token])
                        assert delta <= atol, (mode, token, delta, atol)
                        max_delta = max(max_delta, delta)
    summary = {
        "cudagraph_mode": args.cudagraph_mode,
        "requests_per_policy": 16,
        "token_ids_and_top5_identities_exactly_equal": True,
        "logprob_atol": atol,
        "max_logprob_delta": max_delta,
        "states": {mode: run["states"] for mode, run in runs.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
