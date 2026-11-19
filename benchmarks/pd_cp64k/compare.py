"""Compare completed, audited PD deployments with matching workload/resources."""

import argparse
import json
import re
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text())


def deployment(path):
    assert not (path / "INVALID.json").exists(), path
    assert read(path / "complete.json")["complete"], path
    assert read(path / "source_hashes.json") == read(
        path / "source_hashes_after.json"
    ), path
    settings = read(path / "settings.json")
    runs = sorted(path.glob("run_*/result.json"))
    assert len(runs) == settings["repeats"], path
    for result in runs:
        validation = read(result.parent / "validation.json")
        assert validation["valid"] and validation["deployment"] == "pd", result
    return settings, [read(p) for p in runs]


def server_latency(path):
    def metric(text, name):
        return sum(
            float(v)
            for v in re.findall(
                r"^vllm:" + name + r"(?:\{[^\n]*\})? ([0-9.eE+\-]+)$", text, re.M
            )
        )

    runs = []
    for folder in sorted(path.glob("run_*")):
        result = {}
        for role in ("prefill", "decode"):
            a = (folder / f"{role}_metrics_before.txt").read_text()
            b = (folder / f"{role}_metrics_after.txt").read_text()
            result[role] = {}
            for name in (
                "request_queue_time_seconds",
                "request_prefill_time_seconds",
                "time_to_first_token_seconds",
                "request_inference_time_seconds",
            ):
                count = metric(b, name + "_count") - metric(a, name + "_count")
                result[role][name] = (
                    metric(b, name + "_sum") - metric(a, name + "_sum")
                ) / count
        runs.append(result)
    return runs


def compare(baseline, candidate):
    a, aa = deployment(baseline)
    b, bb = deployment(candidate)
    matched = (
        "input_tokens",
        "num_prompts",
        "concurrency",
        "output_tokens",
        "batch_tokens",
        "decode_batch_tokens",
        "server_limit",
        "kv_gib",
        "gpu_memory_utilization",
        "warmup_requests",
        "prefill_gpus",
        "decode_gpus",
    )
    for key in matched:
        default = {"input_tokens": 65536, "gpu_memory_utilization": 0.90}.get(key)
        assert a.get(key, default) == b.get(key, default), (key, a, b)
        a.setdefault(key, default)
    assert a["mode"] == "tp" and b["mode"] == "cp"
    metrics = {}
    for key in (
        "mean_ttft_ms",
        "mean_e2el_ms",
        "output_throughput",
        "p99_ttft_ms",
        "p99_e2el_ms",
        "p99_itl_ms",
    ):
        x, y = mean(r[key] for r in aa), mean(r[key] for r in bb)
        gain = y / x - 1 if key == "output_throughput" else 1 - y / x
        metrics[key] = {
            "baseline_mean": x,
            "candidate_mean": y,
            "benefit_pct": 100 * gain,
            "baseline_runs": [r[key] for r in aa],
            "candidate_runs": [r[key] for r in bb],
        }
    acceptance = ("mean_ttft_ms", "mean_e2el_ms", "output_throughput")
    return {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "matched_settings": {k: a[k] for k in matched},
        "metrics": metrics,
        "server_latency_seconds": {
            "baseline": server_latency(baseline),
            "candidate": server_latency(candidate),
        },
        "preferred_ttft_target_met": metrics["mean_ttft_ms"]["benefit_pct"] >= 25,
        "any_target_met": any(metrics[k]["benefit_pct"] >= 25 for k in acceptance),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(compare(args.baseline.resolve(), args.candidate.resolve()), indent=2)
    )
