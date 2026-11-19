"""Audited real 4P+4D benchmark through the FlagCX router.

The installed official bench client is unchanged. The validation support
directory supplies the same worker selectors and transfer counters used by
the October 16 campaign; every dependency and launch command is recorded.
"""

import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def metric(text, name):
    values = re.findall(
        r"^vllm:" + re.escape(name) + r"(?:\{[^\n]*\})? ([0-9.eE+\-]+)$", text, re.M
    )
    if not values:
        raise ValueError(f"Missing metric: {name}")
    return sum(map(float, values))


def manifest(source):
    paths = sorted((source / "vllm_fl").rglob("*.py"))
    paths.append(source / "examples/disaggregated_serving_xpyd/router.py")
    return {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def launch(args, support, role):
    mode = args.mode if role == "prefill" else "tp"
    template = args.support_dir / "pd" / mode / f"{role}_launch.json"
    original = json.loads(template.read_text())
    command = original["command"]
    command[0] = sys.executable
    replacements = {
        "--port": str(support.PORTS[role]),
        "--max-num-batched-tokens": str(
            args.batch_tokens if role == "prefill" else args.decode_batch_tokens
        ),
        "--max-num-seqs": str(args.server_limit),
        "--gpu-memory-utilization": str(args.gpu_memory_utilization),
        "--worker-extension-cls": args.worker_extension,
    }
    for key, value in replacements.items():
        command[command.index(key) + 1] = value
    # Historical launch templates contain a fixed KV budget. Remove it to
    # enable vLLM's memory profiling unless the caller requests a fixed budget.
    if "--kv-cache-memory-bytes" in command:
        index = command.index("--kv-cache-memory-bytes")
        del command[index : index + 2]
    if args.kv_gib is not None:
        command.extend(["--kv-cache-memory-bytes", str(args.kv_gib * 1024**3)])
    environment = dict(original["environment"])
    environment.update(
        PYTHONPATH=f"{args.support_dir}:{Path(__file__).resolve().parent}:{args.source}",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        CUDA_VISIBLE_DEVICES=(
            args.prefill_gpus if role == "prefill" else args.decode_gpus
        ),
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("VLLM_FL_", "FLAGCX_", "NCCL_"))
    }
    env.update(environment)
    log = (args.out / f"{role}.log").open("x")
    process = subprocess.Popen(
        command,
        env=env,
        cwd=args.support_dir,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    support.PROCESSES.append((role, process, log))
    save(
        args.out / f"{role}_launch.json",
        {"pid": process.pid, "command": command, "environment": environment},
    )
    print("START", role, mode, process.pid, flush=True)


async def snapshot(args, support, client, phase, folder):
    result = {}
    for role in ("prefill", "decode"):
        response = await client.get(support.URLS[role] + "/metrics")
        response.raise_for_status()
        (folder / f"{role}_metrics_{phase}.txt").write_text(response.text)
        result[role] = {
            "parallelism": await support.rpc(client, role, "inspect_qwen_parallelism"),
            "transfer": await support.rpc(client, role, "inspect_transfer"),
        }
    save(folder / f"inspection_{phase}.json", result)
    return result


def launch_router(args, support):
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("router_trace.py")),
        str(args.router_source),
        "--host",
        "127.0.0.1",
        "--port",
        str(support.PORTS["router"]),
        "--prefill",
        support.URLS["prefill"],
        "21898",
        "--decode",
        support.URLS["decode"],
    ]
    log = (args.out / "router.log").open("x")
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    support.PROCESSES.append(("router", process, log))
    save(args.out / "router_launch.json", {"pid": process.pid, "command": command})


async def checked(coroutine, support):
    task = asyncio.create_task(coroutine)
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=2)
            for role, process, _ in support.PROCESSES:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{role} exited during request processing: {process.returncode}"
                    )
        return await task
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def warmup(args, support, client):
    prompt_directory = (
        "qwen36_128k_20260928" if args.input_tokens > 65536 else "qwen36_64k_20260928"
    )
    prompt = json.loads(
        (args.support_dir.parent / prompt_directory / "prompt_ids.json").read_text()
    )[: args.input_tokens]
    assert len(prompt) == args.input_tokens
    body = {
        "model": "qwen36",
        "prompt": prompt,
        "max_tokens": 2,
        "temperature": 0,
        "ignore_eos": True,
        "seed": 42,
        "stream": False,
    }

    async def one(index):
        start = time.monotonic()
        response = await client.post(
            support.URLS["router"] + "/v1/completions", json=body
        )
        response.raise_for_status()
        usage = response.json()["usage"]
        assert (
            usage["prompt_tokens"] == args.input_tokens
            and usage["completion_tokens"] == 2
        ), usage
        return {"index": index, "duration_s": time.monotonic() - start, "usage": usage}

    print("WARMUP_START", args.warmup_requests, flush=True)
    rows = await asyncio.gather(*(one(i) for i in range(args.warmup_requests)))
    save(args.out / "warmup.json", rows)
    print("WARMUP_DONE", flush=True)


async def monitor(args, support, client):
    stalled_since = None
    captured = False
    with (args.out / "monitor.jsonl").open("x") as log:
        while True:
            row = {"time": time.time(), "gpu": support.gpu_state(), "roles": {}}
            for role in ("prefill", "decode"):
                try:
                    response = await client.get(
                        support.URLS[role] + "/metrics", timeout=3
                    )
                    response.raise_for_status()
                    row["roles"][role] = {
                        name: metric(response.text, name)
                        for name in (
                            "num_requests_running",
                            "num_requests_waiting",
                            "kv_cache_usage_perc",
                            "num_preemptions_total",
                            "request_success_total",
                        )
                    }
                except (httpx.HTTPError, ValueError) as exc:
                    row["roles"][role] = {"error": str(exc)}
            log.write(json.dumps(row) + "\n")
            log.flush()
            p, d = (row["roles"].get(r, {}) for r in ("prefill", "decode"))
            stalled = (
                p.get("num_requests_running") == 0
                and p.get("num_requests_waiting") == 0
                and d.get("num_requests_running") == 0
                and d.get("num_requests_waiting", 0) > 0
            )
            if not stalled:
                stalled_since = None
                captured = False
            elif stalled_since is None:
                stalled_since = time.monotonic()
            elif time.monotonic() - stalled_since >= 30 and not captured:
                details = {"time": time.time()}
                for role in ("prefill", "decode"):
                    details[role] = await support.rpc(
                        client, role, "inspect_pd_pending"
                    )
                response = await client.get(support.URLS["router"] + "/pd_pending")
                details["router"] = response.json()
                save(args.out / f"stall_{int(time.time())}.json", details)
                print("PD_STALL_CAPTURED", args.out, flush=True)
                captured = True
            await asyncio.sleep(5)


def audit(args, before, after, folder):
    result = json.loads((folder / "result.json").read_text())
    assert result["completed"] == args.num_prompts and result["failed"] == 0, result[
        "errors"
    ]
    assert not any(result["errors"])
    assert result["input_lens"] == [args.input_tokens] * args.num_prompts
    assert result["output_lens"] == [args.output_tokens] * args.num_prompts
    assert result["max_concurrency"] == args.concurrency
    checks = {
        "valid": True,
        "deployment": "pd",
        "mode": args.mode,
        "input_output_exact": True,
        "num_prompts": args.num_prompts,
        "roles": {},
    }
    ids = {}
    for role in ("prefill", "decode"):
        a = (folder / f"{role}_metrics_before.txt").read_text()
        b = (folder / f"{role}_metrics_after.txt").read_text()
        preemptions = metric(b, "num_preemptions_total") - metric(
            a, "num_preemptions_total"
        )
        assert preemptions >= 0
        role_check = {
            "preemption_delta": preemptions,
            "transfers": [],
            "parallelism": [],
        }
        role_ids = []
        for x, y in zip(before[role]["transfer"], after[role]["transfer"], strict=True):
            assert all(
                m["page_contains_ssm"] and m["shared_storage"] for m in y["mamba"]
            )
            changes = {
                key: y[key] - x[key]
                for key in (
                    "successful_send_calls",
                    "successful_write_calls",
                    "successful_write_bytes",
                    "receive_calls",
                    "sent_requests",
                    "received_requests",
                )
            }
            sent = role == "prefill"
            key = "sent_transfer_ids" if sent else "received_transfer_ids"
            transfer_ids = y[key][len(x[key]) :]
            assert len(transfer_ids) == len(set(transfer_ids)) == args.num_prompts
            assert (
                changes["sent_requests" if sent else "received_requests"]
                == args.num_prompts
            )
            assert changes["successful_write_bytes" if sent else "receive_calls"] > 0
            role_ids.append(set(transfer_ids))
            role_check["transfers"].append(changes)
        assert len(role_ids) == 4 and all(i == role_ids[0] for i in role_ids)
        ids[role] = role_ids[0]
        for x, y in zip(
            before[role]["parallelism"], after[role]["parallelism"], strict=True
        ):
            if role == "decode" or args.mode == "tp":
                assert not y["pr_layers"]
            else:
                deltas = [
                    {
                        key: v[key] - u[key]
                        for key in v
                        if key.endswith("calls") or key.endswith("tokens")
                    }
                    for u, v in zip(x["pr_layers"], y["pr_layers"], strict=True)
                ]
                assert len(deltas) == 10 and all(d["prefill_calls"] > 0 for d in deltas)
                if "prefill_tokens" in deltas[0]:
                    for d in deltas:
                        total = (
                            d["prefill_tokens"]
                            + d["fallback_tokens"]
                            + d["decode_tokens"]
                        )
                        assert total == args.num_prompts * (args.input_tokens - 1), d
                        d["cp_token_fraction"] = d["prefill_tokens"] / total
                role_check["parallelism"].append(deltas)
        checks["roles"][role] = role_check
    assert ids["prefill"] == ids["decode"]
    checks["matched_transfer_requests_per_rank"] = args.num_prompts
    save(folder / "validation.json", checks)
    keys = (
        "duration",
        "completed",
        "failed",
        "output_throughput",
        "mean_e2el_ms",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_itl_ms",
    )
    summary = {
        "deployment": "pd",
        "mode": args.mode,
        **{key: result[key] for key in keys},
    }
    save(folder / "summary.json", summary)
    return summary


async def bench(args, support, client, index):
    folder = args.out / f"run_{index}"
    folder.mkdir()
    before = await snapshot(args, support, client, "before", folder)
    command = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        support.URLS["router"],
        "--model",
        "qwen36",
        "--tokenizer",
        "/data/model",
        "--endpoint",
        "/v1/completions",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.input_tokens),
        "--random-output-len",
        str(args.output_tokens),
        "--random-range-ratio",
        "0.0",
        "--max-concurrency",
        str(args.concurrency),
        "--num-prompts",
        str(args.num_prompts),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--temperature",
        "0",
        "--seed",
        "42",
        "--num-warmups",
        "0",
        "--ready-check-timeout-sec",
        "0",
        "--disable-tqdm",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(folder),
        "--result-filename",
        "result.json",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--extra-body",
        '{"seed":42}',
    ]
    env = dict(
        os.environ,
        VLLM_PLUGINS="",
        PYTHONPATH=str(args.support_dir / "bench_deps"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    save(
        folder / "command.json",
        {
            "command": command,
            "environment": {
                k: env[k]
                for k in (
                    "VLLM_PLUGINS",
                    "PYTHONPATH",
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "NO_PROXY",
                    "no_proxy",
                )
            },
        },
    )
    (folder / "command.sh").write_text(shlex.join(command) + "\n")
    print("BENCH_START", args.mode, index, flush=True)
    with (folder / "bench.log").open("x") as log:
        process = await asyncio.create_subprocess_exec(
            *command, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            code = await asyncio.wait_for(process.wait(), 3600)
        except BaseException:
            if process.returncode is None:
                process.terminate()
                await process.wait()
            raise
    assert code == 0, folder
    after = await snapshot(args, support, client, "after", folder)
    summary = audit(args, before, after, folder)
    print("BENCH_DONE", json.dumps(summary), flush=True)
    return summary


async def main(args, support):
    support.require_idle()
    args.out.mkdir(parents=True)
    save(
        args.out / "settings.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    before = manifest(args.source)
    router_before = hashlib.sha256(args.router_source.read_bytes()).hexdigest()
    save(
        args.out / "router_hash.json",
        {"source": str(args.router_source), "sha256": router_before},
    )
    save(args.out / "source_hashes.json", before)
    save(
        args.out / "support_hashes.json",
        {
            str(p.relative_to(args.support_dir)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(args.support_dir.glob("*.py"))
        },
    )
    save(
        args.out / "driver_hashes.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )
    save(
        args.out / "revision.json",
        {
            "commit": subprocess.check_output(
                ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "diff": subprocess.check_output(
                ["git", "-C", str(args.source), "diff", "HEAD"], text=True
            ),
            "model_config_sha256": hashlib.sha256(
                Path("/data/model/config.json").read_bytes()
            ).hexdigest(),
        },
    )
    monitor_task = None
    try:
        for role in ("prefill", "decode"):
            launch(args, support, role)
        support.SOURCE = args.source
        launch_router(args, support)
        async with httpx.AsyncClient(
            timeout=3600, trust_env=False, limits=httpx.Limits(max_connections=160)
        ) as client:
            await support.ready(client, ["prefill", "decode"], router=True)
            for _ in range(120):
                if (
                    await client.get(support.URLS["router"] + "/pd_health")
                ).status_code == 200:
                    break
                await asyncio.sleep(1)
            else:
                raise TimeoutError("PD router did not become ready")
            for role in ("prefill", "decode"):
                mode = args.mode if role == "prefill" else "tp"
                await support.configure(client, role, mode, args.out, pd=True)
                log = (args.out / f"{role}.log").read_text()
                capacity = re.search(r"GPU KV cache size: ([0-9,]+) tokens", log)
                assert capacity, role
                save(
                    args.out / f"{role}_capacity.json",
                    {"tokens": int(capacity[1].replace(",", ""))},
                )
            monitor_task = asyncio.create_task(monitor(args, support, client))
            if args.audit_reference:
                assert args.mode == "cp"
                await support.rpc(
                    client,
                    "prefill",
                    "begin_pd_layout_audit",
                    str(args.audit_reference),
                )
            await checked(warmup(args, support, client), support)
            if args.audit_reference:
                rows = await support.rpc(client, "prefill", "end_pd_layout_audit")
                assert len(rows) == 4 and all(r for r in rows)
                save(args.out / "layout_audit.json", rows)
            summaries = [
                await checked(bench(args, support, client, index), support)
                for index in range(1, args.repeats + 1)
            ]
            save(args.out / "summary.json", summaries)
            save(args.out / "complete.json", {"complete": True})
    finally:
        # A second termination must not interrupt cleanup of owned GPU workers.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if monitor_task:
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await monitor_task
        owned_groups = [process.pid for _, process, _ in support.PROCESSES]
        support.stop()
        # An API process can exit before its workers. Reap any remaining
        # members of these deployment-owned groups as well.
        for group in owned_groups:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group, signal.SIGKILL)
        after = manifest(args.source)
        save(args.out / "source_hashes_after.json", after)
        assert after == before, "Source changed during the measured deployment"
        assert (
            hashlib.sha256(args.router_source.read_bytes()).hexdigest() == router_before
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["tp", "cp"], required=True)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--router-source",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "examples/disaggregated_serving_xpyd/router.py"
        ),
    )
    parser.add_argument(
        "--support-dir",
        type=Path,
        default=Path("/vllm-workspace/validation/qwen36_benchserve_o1024_20261016"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--input-tokens", type=int, default=65536)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--batch-tokens", type=int, default=65536)
    parser.add_argument("--decode-batch-tokens", type=int, default=65536)
    parser.add_argument("--server-limit", type=int, default=64)
    parser.add_argument(
        "--kv-gib",
        type=int,
        help="Fixed KV GiB per rank; omit to use vLLM memory profiling.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--warmup-requests", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--worker-extension", default="pd_worker.PDWorker")
    parser.add_argument("--audit-reference", type=Path)
    parser.add_argument("--prefill-gpus", default="0,1,2,3")
    parser.add_argument("--decode-gpus", default="4,5,6,7")
    args = parser.parse_args()
    if args.kv_gib is not None and args.kv_gib <= 0:
        parser.error("--kv-gib must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    args.out = args.out.resolve()
    args.source = args.source.resolve()
    sys.path.insert(0, str(args.support_dir))
    import support

    # FlagCX reserves bootstrap_port + TP rank (21898..21901/21998..22001).
    support.PORTS.update(prefill=22903, decode=22904, router=22900)
    support.URLS.update(
        {
            r: f"http://127.0.0.1:{support.PORTS[r]}"
            for r in ("prefill", "decode", "router")
        }
    )
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    lock_path = args.support_dir.parent / "qwen36_fused_layout_64k_20260929/.gpu.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(main(args, support))
