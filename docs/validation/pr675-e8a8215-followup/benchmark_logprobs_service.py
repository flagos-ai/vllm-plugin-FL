"""Measure API token arrivals and server-tree CPU; never infer tokens from text."""

import argparse
import concurrent.futures as cf
import json
import statistics
import time
from pathlib import Path

import psutil
import requests


def request_one(url, enabled, tokens):
    body = dict(
        model="qwen",
        messages=[
            {"role": "user", "content": "Explain how a computer executes instructions."}
        ],
        stream=True,
        temperature=0,
        max_tokens=tokens,
        ignore_eos=True,
        return_token_ids=True,
        stream_interval=1,
        include_reasoning=True,
    )
    if enabled:
        body.update(logprobs=True, top_logprobs=5)
    start = time.perf_counter()
    arrivals, ids, widths = [], [], []
    with requests.post(
        url + "/chat/completions", json=body, stream=True, timeout=600
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            data = json.loads(line[6:])
            if "error" in data:
                raise RuntimeError(data["error"])
            for choice in data.get("choices", []):
                token_ids = choice.get("token_ids") or []
                if token_ids:
                    arrivals.append(time.perf_counter())
                    ids.extend(token_ids)
                    widths.append(len(token_ids))
    assert len(ids) == tokens, (len(ids), tokens)
    assert len(arrivals) >= 2
    intervals = [b - a for a, b in zip(arrivals, arrivals[1:])]
    return dict(
        seconds=time.perf_counter() - start,
        ttft=arrivals[0] - start,
        token_ids=ids,
        widths=widths,
        inter_chunk=intervals,
        itl=intervals if all(w == 1 for w in widths) else None,
        tpot=(arrivals[-1] - arrivals[0]) / (len(ids) - widths[0]),
    )


def cpu_snapshot(pid):
    root = psutil.Process(pid)
    result = {}
    for proc in [root] + root.children(recursive=True):
        try:
            t = proc.cpu_times()
            result[(proc.pid, proc.create_time())] = t.user + t.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def percentile(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-pid", required=True, type=int)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for n in args.concurrency:
        for enabled in (False, True):
            with cf.ThreadPoolExecutor(max_workers=n) as pool:
                list(
                    pool.map(
                        lambda _, enabled=enabled: request_one(
                            args.base_url, enabled, args.max_tokens
                        ),
                        range(n),
                    )
                )
        for enabled in (False, True, True, False):
            before = cpu_snapshot(args.server_pid)
            start = time.perf_counter()
            with cf.ThreadPoolExecutor(max_workers=n) as pool:
                samples = list(
                    pool.map(
                        lambda _, enabled=enabled: request_one(
                            args.base_url, enabled, args.max_tokens
                        ),
                        range(n),
                    )
                )
            elapsed = time.perf_counter() - start
            after = cpu_snapshot(args.server_pid)
            cpu_seconds = sum(
                max(0, after[key] - before[key]) for key in before.keys() & after.keys()
            )
            intervals = [v for sample in samples for v in sample["inter_chunk"]]
            itl = [v for sample in samples for v in (sample["itl"] or [])]
            exact = all(sample["itl"] is not None for sample in samples)
            record = dict(
                concurrency=n,
                logprobs=enabled,
                elapsed=elapsed,
                output_tokens_per_second=sum(len(s["token_ids"]) for s in samples)
                / elapsed,
                cpu_seconds=cpu_seconds,
                cpu_core_equivalents=cpu_seconds / elapsed,
                cpu_snapshot_processes=[len(before), len(after)],
                exact_token_itl=exact,
                median_itl=statistics.median(itl) if exact else None,
                p95_itl=percentile(itl, 0.95) if exact else None,
                median_inter_chunk=statistics.median(intervals),
                p95_inter_chunk=percentile(intervals, 0.95),
                median_tpot=statistics.median(s["tpot"] for s in samples),
                samples=samples,
            )
            records.append(record)
            (output / "measurements.json").write_text(json.dumps(records, indent=2))
            print(
                json.dumps({k: v for k, v in record.items() if k != "samples"}),
                flush=True,
            )


if __name__ == "__main__":
    main()
