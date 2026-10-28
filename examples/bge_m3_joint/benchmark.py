# SPDX-License-Identifier: Apache-2.0
"""Full-response BGE-M3 HTTP benchmark; the server is managed separately."""

import argparse
import gc
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import orjson
import requests

MODES = {"dense": 1, "sparse": 2, "colbert": 4, "joint": 7}
FIELDS = ("dense_vecs", "lexical_weights", "colbert_vecs")


def decode_response(response, parser):
    """Decode bytes directly; requests.json is retained for A/B measurements."""
    response.raise_for_status()
    if parser == "stdlib":
        return response.json()["data"]
    return orjson.loads(response.content)["data"]


def validate_output(result, mask, length):
    if set(result) != set(FIELDS):
        raise ValueError("Unexpected output fields")
    for bit, field in enumerate(FIELDS):
        value = result[field]
        if not mask & (1 << bit):
            if value is not None:
                raise ValueError(f"{field} must be null when disabled")
            continue
        if not isinstance(value, list) or len(value) != 1:
            raise ValueError(f"{field} must retain the one-document batch")
        if bit == 1:
            if not isinstance(value[0], dict):
                raise ValueError("Sparse output must be a token-ID dictionary")
            for key, weight in value[0].items():
                if not key.isdecimal() or not np.isfinite(weight) or weight <= 0:
                    raise ValueError("Invalid sparse token/weight")
        else:
            array = np.asarray(value[0])
            shape = (1024,) if bit == 0 else (length - 1, 1024)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid {field} shape or non-finite values")


def http_peak(rows):
    # Finish precedes start for tied timestamps.
    events = [(row["start_ns"], 1) for row in rows]
    events += [(row.get("body_received_ns", row["end_ns"]), -1) for row in rows]
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


def run_round(url, model, texts, length, mask, concurrency, parser, timeout):
    barrier = threading.Barrier(concurrency + 1)
    local = threading.local()
    sessions = []
    lock = threading.Lock()

    def one(index):
        if not hasattr(local, "session"):
            local.session = requests.Session()
            local.session.trust_env = False
            with lock:
                sessions.append(local.session)
        payload = {
            "model": model,
            "task": "plugin",
            "data": {
                "sentences": [texts[index]],
                "max_length": 8192,
                "return_dense": bool(mask & 1),
                "return_sparse": bool(mask & 2),
                "return_colbert_vecs": bool(mask & 4),
            },
        }
        if index < concurrency:
            barrier.wait(timeout=60)
        row = {
            "index": index,
            "start_ns": time.perf_counter_ns(),
            "start_wall_ns": time.time_ns(),
            "passed": False,
        }
        try:
            with local.session.post(
                url.rstrip("/") + "/pooling", json=payload, timeout=(30, timeout)
            ) as response:
                row["body_received_ns"] = time.perf_counter_ns()
                row["status"] = response.status_code
                row["response_bytes"] = len(response.content)
                result = decode_response(response, parser)
                validate_output(result, mask, length)
                row["passed"] = True
        except Exception as error:
            row["error"] = repr(error)
        finally:
            row["end_ns"] = time.perf_counter_ns()
            row["latency_ms"] = (row["end_ns"] - row["start_ns"]) / 1e6
        return row

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one, i) for i in range(len(texts))]
            start_wall_ns = time.time_ns()
            start = time.perf_counter_ns()
            barrier.wait(timeout=60)
            rows = [future.result() for future in futures]
        end = time.perf_counter_ns()
    finally:
        for session in sessions:
            session.close()
    successful = [row for row in rows if row["passed"]]
    latency = [row["latency_ms"] for row in successful]
    seconds = (end - start) / 1e9
    peak = http_peak(rows)
    summary = {
        "length": length,
        "mask": mask,
        "requests": len(texts),
        "documents": len(texts),
        "concurrency": concurrency,
        "observed_http_inflight_peak": peak,
        "start_wall_ns": start_wall_ns,
        "end_wall_ns": time.time_ns(),
        "seconds": seconds,
        "docs_per_s": len(successful) / seconds,
        "effective_input_tokens_per_s": len(successful) * length / seconds,
        "response_bytes": sum(row.get("response_bytes", 0) for row in rows),
        "failures": len(texts) - len(successful),
        "retries": 0,
        "passed": len(successful) == len(texts) and peak == concurrency,
        "latency_ms": {
            "mean": float(np.mean(latency)),
            **{f"p{p}": float(np.percentile(latency, p)) for p in (50, 95, 99)},
        }
        if latency
        else {},
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help='JSON object containing a nonempty "texts" list',
    )
    parser.add_argument("--length", type=int, choices=(1024, 4096), required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--json-parser", choices=("orjson", "stdlib"), default="orjson")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= args.requests or args.timeout <= 0:
        parser.error("Require 1 <= concurrency <= requests and a positive timeout")

    from transformers import AutoTokenizer

    raw = args.input.read_bytes()
    original = json.loads(raw)["texts"]
    if not original or not all(isinstance(text, str) for text in original):
        parser.error("Input must contain a nonempty list of strings")
    texts = [original[i % len(original)] for i in range(args.requests)]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    lengths = [len(ids) for ids in tokenizer(texts)["input_ids"]]
    if lengths != [args.length] * args.requests:
        parser.error("Every fixture must tokenize to the requested length")

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source": str(args.input),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_count": len(original),
        "text_sha256": [hashlib.sha256(text.encode()).hexdigest() for text in texts],
        "lengths": lengths,
        "json_parser": args.json_parser,
        "orjson_version": orjson.__version__,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "warmup_rounds": 1,
        "formal_rounds": 1,
        "timing": "HTTP send through complete body, JSON parse and shape/finite check",
    }
    (args.output / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
    rounds = []
    for mode in args.modes:
        for phase in ("warmup", "formal"):
            print(f"START {args.length} {mode} {phase}", flush=True)
            summary, rows = run_round(
                args.url,
                args.model,
                texts,
                args.length,
                MODES[mode],
                args.concurrency,
                args.json_parser,
                args.timeout,
            )
            summary.update(mode=mode, round=phase, json_parser=args.json_parser)
            rounds.append(summary)
            (args.output / f"{mode}-{phase}-requests.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            with (args.output / "rounds.jsonl").open("a") as handle:
                handle.write(json.dumps(summary) + "\n")
            print(json.dumps(summary), flush=True)
            if not summary["passed"]:
                raise RuntimeError(
                    "Failed requests or requested HTTP concurrency not reached"
                )
            gc.collect()
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "passed": True,
                "length": args.length,
                "formal": [row for row in rounds if row["round"] == "formal"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
