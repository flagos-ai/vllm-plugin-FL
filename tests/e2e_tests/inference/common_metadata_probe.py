# Copyright (c) 2026 BAAI. All rights reserved.
"""Compare metadata policies in separate full worker processes, without downloads.

Run against an installed plugin wheel; this script does not add the checkout's
package root to PYTHONPATH. Artifacts include the local checkpoint, exact token
IDs/logprobs, producer counters and one log per policy.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


class MetadataProbe:
    """Named worker RPC avoids enabling pickle-based function serialization."""

    def metadata_state(self):
        runner = self.model_runner
        helper = runner.common_attention_metadata_graph
        from vllm_fl import flaggems_runtime

        identity = runtime_identity()
        expected = json.loads(Path(os.environ["COMMON_PROBE_MANIFEST"]).read_text())
        verify_identity(
            identity, Path(os.environ["COMMON_PROBE_INSTALLED_ROOT"]), expected
        )
        state = flaggems_runtime._STATE
        mm_state = state.mm_state if state else None
        return {
            "identity": identity,
            "shape_aware_mm": mm_state.result.status if mm_state else "disabled",
            "mm_backends": asdict(mm_state.result) if mm_state else None,
            "mm_threshold": state.config.mm_threshold if state else None,
            "policy": runner.common_metadata_policy.mode,
            "async_scheduling": runner.use_async_scheduling,
            "captures": helper.captures if helper else 0,
            "replays": helper.replays if helper else 0,
            "eager_calls": helper.eager_calls if helper else 0,
        }


RUNTIME_MODULES = (
    "vllm_fl.worker.common_attention_metadata",
    "vllm_fl.worker.model_runner",
    "vllm_fl.worker.worker",
    "vllm_fl.patches.flaggems_mm_shape_aware",
    "vllm_fl.flaggems_runtime",
)


def runtime_identity():
    import vllm_fl

    files = {}
    for name in RUNTIME_MODULES:
        path = Path(importlib.import_module(name).__file__).resolve()
        files[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    versions = {}
    for package in ("vllm-plugin-FL", "vllm", "torch", "flag-gems", "triton"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {
        "package_path": str(Path(vllm_fl.__file__).resolve()),
        "files": files,
        "versions": versions,
    }


def verify_identity(identity, installed_root, expected):
    root = installed_root.resolve()
    assert Path(identity["package_path"]).parent == root / "vllm_fl", identity
    for module, record in identity["files"].items():
        relative = module.replace(".", "/") + ".py"
        assert Path(record["path"]) == root / relative, record
        assert record["sha256"] == expected[relative], record


def compare_outputs(actual, expected, *, label, atol):
    assert actual["prompt"] == expected["prompt"], label
    assert actual["tokens"] == expected["tokens"], label
    maximum = 0.0
    for step, reference in zip(actual["logprobs"], expected["logprobs"], strict=True):
        assert step.keys() == reference.keys(), label
        for token, value in step.items():
            assert math.isfinite(value) and math.isfinite(reference[token]), label
            delta = abs(value - reference[token])
            assert delta <= atol, (label, token, delta, atol)
            maximum = max(maximum, delta)
    return maximum


def compare_runs(runs, atol):
    maximum = 0.0
    for mode, run in runs.items():
        first, second = run["results"]
        by_prompt = {tuple(output["prompt"]): output for output in first}
        assert len(by_prompt) == len(first) == len(second)
        assert {tuple(x["prompt"]) for x in second} == by_prompt.keys()
        for output in second:
            maximum = max(
                maximum,
                compare_outputs(
                    output,
                    by_prompt[tuple(output["prompt"])],
                    label=f"{mode}: request reuse",
                    atol=atol,
                ),
            )
        for batch, reference in zip(
            run["results"], runs["stock"]["results"], strict=True
        ):
            for actual, expected in zip(batch, reference, strict=True):
                maximum = max(
                    maximum,
                    compare_outputs(
                        actual, expected, label=f"{mode}: policy parity", atol=atol
                    ),
                )
    return maximum


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
            if args.combination:
                assert state["shape_aware_mm"] == "installed", state
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
    parser.add_argument(
        "--installed-root",
        type=Path,
        required=True,
        help="Explicit installed site-packages directory (not a checkout)",
    )
    parser.add_argument(
        "--expected-runtime-manifest",
        type=Path,
        required=True,
        help="Build-time relative-path -> SHA256 JSON from the wheel",
    )
    parser.add_argument(
        "--dependency-path",
        type=Path,
        action="append",
        default=[],
        help="Explicit dependency source path, e.g. validated FlagGems",
    )
    parser.add_argument(
        "--combination",
        action="store_true",
        help="Enable real worker FlagGems policy and shape-aware MM",
    )
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
    env["USE_FLAGGEMS"] = "1" if args.combination else "0"
    env["VLLM_FL_FLAGOS_MM_SHAPE_AWARE"] = "1" if args.combination else "0"
    env["VLLM_FL_FLAGOS_MM_DECODE_MAX_M"] = "2"
    isolated = args.output_dir / "probe"
    isolated.mkdir()
    script = isolated / "common_metadata_probe.py"
    shutil.copy2(Path(__file__).resolve(), script)
    installed = args.installed_root.resolve()
    # Require wheel metadata, then independently verify each worker's imports.
    assert any(installed.glob("vllm_plugin_fl-*.dist-info")), installed
    env["PYTHONPATH"] = os.pathsep.join(
        map(str, [isolated, installed] + [p.resolve() for p in args.dependency_path])
    )
    env["COMMON_PROBE_INSTALLED_ROOT"] = str(installed)
    env["COMMON_PROBE_MANIFEST"] = str(args.expected_runtime_manifest.resolve())
    runs = {}
    for mode in ("stock", "eager", "graph"):
        env["VLLM_FL_COMMON_ATTENTION_METADATA"] = mode
        with (args.output_dir / f"{mode}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--output-dir",
                    str(args.output_dir),
                    "--cudagraph-mode",
                    args.cudagraph_mode,
                    "--worker-mode",
                    mode,
                    "--installed-root",
                    str(installed),
                    "--expected-runtime-manifest",
                    env["COMMON_PROBE_MANIFEST"],
                ]
                + (["--combination"] if args.combination else []),
                env=env,
                cwd=isolated,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        runs[mode] = json.loads((args.output_dir / f"{mode}.json").read_text())
    # The combined probe intentionally switches between native and FlagGems
    # GEMM as M changes across reordered batches. Those FP16 kernels need not
    # round identically. Bound logprob drift by one FP16 epsilon while keeping
    # token IDs and top-k identities exact. Without MM routing, only separate
    # PIECEWISE compilations allow a few FP32 log-softmax ULPs. Metadata tensor
    # comparisons remain exact in all modes.
    atol = (
        2**-10
        if args.combination
        else (1e-5 if args.cudagraph_mode == "PIECEWISE" else 0.0)
    )
    max_delta = compare_runs(runs, atol)
    summary = {
        "cudagraph_mode": args.cudagraph_mode,
        "requests_per_policy": 16,
        "request_reuse_checked": True,
        "installed_worker_identity_checked": True,
        "combination": args.combination,
        "accepted": args.cudagraph_mode != "FULL",
        "token_ids_and_top5_identities_exactly_equal": True,
        "logprob_atol": atol,
        "max_logprob_delta": max_delta,
        "states": {mode: run["states"] for mode, run in runs.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if args.cudagraph_mode == "FULL":
        raise SystemExit(
            "Mixed FULL is experimental and cannot receive an acceptance PASS"
        )


if __name__ == "__main__":
    main()
