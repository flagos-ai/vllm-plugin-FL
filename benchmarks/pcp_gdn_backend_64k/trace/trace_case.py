"""PCP 64K×1: warm up, then capture three annotated first-token requests."""

import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("flashinfer", "triton"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    from vllm import LLM, SamplingParams

    settings = dict(
        model="/data/model",
        tensor_parallel_size=4,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        seed=42,
        max_model_len=66560,
        max_num_batched_tokens=65536,
        max_num_seqs=64,
        kv_cache_memory_bytes=4294967296,
        gpu_memory_utilization=0.85,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 3},
        kernel_config={"moe_backend": "triton"},
        enable_chunked_prefill=True,
        language_model_only=True,
        generation_config="vllm",
        disable_log_stats=False,
        additional_config={"gdn_prefill_backend": "flashinfer"},
        worker_extension_cls="trace_worker.TraceWorker",
    )
    ids = json.loads((ROOT.parent / "qwen36_64k_20260928/prompt_ids.json").read_text())
    assert len(ids) == 65536
    params = SamplingParams(temperature=0, seed=42, max_tokens=1, ignore_eos=True)
    result = {
        "backend": args.backend,
        "settings": settings,
        "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "warmups": [],
        "captures": [],
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    llm = LLM(**settings)

    def generate():
        start = time.perf_counter()
        outputs = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)
        elapsed = (time.perf_counter() - start) * 1000
        assert len(outputs) == 1 and list(outputs[0].prompt_token_ids) == ids
        out = outputs[0].outputs[0]
        assert len(out.token_ids) == 1 and not outputs[0].metrics.is_corrupted
        return {"wall_ms": elapsed, "tokens": list(out.token_ids), "text": out.text}

    try:
        result["identity"] = llm.collective_rpc(
            "configure", args=("fused", args.backend)
        )
        assert len(result["identity"]) == 4
        assert all(
            len(r["gdn"]) == 30
            and r["fa_versions"] == [3] * 10
            and r["layer_forward"] == "vllm_fl.cp.prstyle"
            for r in result["identity"]
        )
        for i in range(5):
            result["warmups"].append(generate())
            print("WARMUP", args.backend, i, result["warmups"][-1], flush=True)
        result["annotations"] = llm.collective_rpc(
            "install_trace", args=(args.backend,)
        )
        llm.collective_rpc("synchronize_trace")
        llm.collective_rpc("start_trace")
        try:
            for sample in range(3):
                llm.collective_rpc("set_trace_sample", args=(sample,))
                row = generate()
                result["captures"].append(row)
                print("CAPTURE", args.backend, sample, row, flush=True)
        finally:
            llm.collective_rpc("synchronize_trace")
            llm.collective_rpc("stop_trace")
        result["trace_metadata"] = llm.collective_rpc("trace_metadata")
        assert all(
            len(r["calls"]) == 90 and len(r["core_meta"]) == 30
            for r in result["trace_metadata"]
        )
        assert (
            len({tuple(r["tokens"]) for r in result["warmups"] + result["captures"]})
            == 1
        )
        result["completed"] = True
        save()
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
