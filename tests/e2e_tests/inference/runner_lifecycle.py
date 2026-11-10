"""Real R6 worker/runner probe; no mocks or upstream modifications."""

import argparse
import json
import os
from pathlib import Path


def inspect_worker(worker):
    from vllm.platforms import current_platform

    runner = worker.model_runner
    return {
        "runner": f"{type(runner).__module__}.{type(runner).__name__}",
        "use_v2": worker.use_v2_model_runner,
        "config_use_v2": worker.vllm_config.use_v2_model_runner,
        "download_dir": worker.load_config.download_dir,
        "config_shared": runner.load_config is worker.vllm_config.load_config,
        "sleep_available": current_platform.is_sleep_mode_available(),
        "rank": worker.rank,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", choices=["default", "v1", "v2"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--sleep", action="store_true")
    parser.add_argument("--external-launcher", action="store_true")
    args = parser.parse_args()
    if args.runner == "default":
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    else:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = str(int(args.runner == "v2"))
    from importlib.metadata import version

    from vllm import LLM, SamplingParams

    import vllm_fl

    assert version("vllm") in ("0.32.0", "0.32.0+empty")
    result = {
        "args": vars(args),
        "plugin": vllm_fl.__file__,
        "versions": {p: version(p) for p in ("vllm", "torch", "flag-gems", "flagtree")},
    }
    extra = {"async_scheduling": False} if args.pp > 1 else {}
    if args.external_launcher:
        extra["distributed_executor_backend"] = "external_launcher"
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        dtype="bfloat16",
        max_model_len=256,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.5,
        enforce_eager=not args.graph,
        enable_sleep_mode=args.sleep,
        enable_prefix_caching=True,
        **extra,
    )
    result["before"] = llm.collective_rpc(inspect_worker)
    if args.external_launcher and args.runner == "default" and args.pp > 1:
        assert all(not w["use_v2"] for w in result["before"]), result["before"]
    if args.runner != "default":
        assert all(w["use_v2"] == (args.runner == "v2") for w in result["before"])
    assert all(w["config_use_v2"] == w["use_v2"] for w in result["before"])
    if not result["before"][0]["use_v2"]:
        assert all(
            w["runner"] == "vllm_fl.worker.model_runner.ModelRunnerFL"
            for w in result["before"]
        )
    prompts = ["The capital of France is", "One plus one equals"]
    sampling = SamplingParams(temperature=0, max_tokens=16)

    def generate():
        outputs = llm.generate(prompts, sampling)
        texts = [o.outputs[0].text for o in outputs]
        assert len(texts) == 2 and all(t.strip() for t in texts), texts
        assert "paris" in texts[0].lower(), texts
        return {"texts": texts, "tokens": [o.outputs[0].token_ids for o in outputs]}

    result["initial"] = generate()
    llm.collective_rpc(
        "update_config",
        args=({"load_config": {"download_dir": "/tmp/r6-config-updated"}},),
    )
    result["after_update"] = llm.collective_rpc(inspect_worker)
    assert all(
        w["download_dir"] == "/tmp/r6-config-updated" and w["config_shared"]
        for w in result["after_update"]
    )
    result["updated_generation"] = generate()
    assert result["updated_generation"]["tokens"] == result["initial"]["tokens"]
    assert llm.reset_prefix_cache()
    result["after_cache_reset"] = generate()
    assert result["after_cache_reset"]["tokens"] == result["initial"]["tokens"]
    if args.sleep:
        llm.sleep(level=1)
        assert llm.wake_up()
        result["after_wake"] = generate()
        assert result["after_wake"]["tokens"] == result["initial"]["tokens"]
    output = args.output
    if args.external_launcher:
        output = f"{output}.rank{os.environ['RANK']}.json"
    Path(output).write_text(json.dumps(result, indent=2))
    llm.llm_engine.engine_core.shutdown(timeout=30)
    print("R6_RUNTIME_PASS", output, flush=True)


if __name__ == "__main__":
    main()
