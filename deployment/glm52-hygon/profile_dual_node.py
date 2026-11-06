"""Two-node native-mp HIP profile: trace-off launch, health, warmup, measure,
stop/flush, exact-group shutdown, offline export. Separate from single-node Qwen.
Run from node0; only the specified node1 container is controlled over SSH.
"""
import argparse
import csv
from collections import defaultdict
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request


def shutdown_group_command(pgid, grace_seconds):
    """Bounded cleanup of a verified owned group, allowing profiler DB drain."""
    if not 0 < grace_seconds <= 1800:
        raise ValueError("shutdown grace must be in 1..1800 seconds")
    code = """import os,signal,sys,time
g=int(sys.argv[1]); grace=int(sys.argv[2])
def members():
 out=[]
 for x in os.listdir('/proc'):
  if not x.isdigit(): continue
  try:
   if os.getpgid(int(x))!=g: continue
   cmd=open('/proc/'+x+'/cmdline','rb').read()
   if not cmd: continue
   assert any(k in cmd for k in (b'hipprof',b'dual_node_smoke',b'VLLM::',b'multiprocessing',b'torch')), (x,cmd)
   out.append(int(x))
  except (ProcessLookupError,FileNotFoundError): pass
 return out
live=members()
print('verified group',g,live,flush=True)
if live: os.killpg(g,signal.SIGINT)
end=time.monotonic()+grace
while members() and time.monotonic()<end: time.sleep(2)
live=members()
if live:
 print('forced cleanup after SIGINT grace',g,live,flush=True)
 os.killpg(g,signal.SIGKILL)
 for _ in range(15):
  if not members(): break
  time.sleep(1)
assert not members(),members()
"""
    return ["python3", "-c", code, str(pgid), str(grace_seconds)], grace_seconds + 25


def profile_index_window(source, seconds, device):
    """Read-only window; retain whole launch indices across the time boundary."""
    if not seconds > 0:
        raise ValueError("window seconds must be positive")
    with sqlite3.connect(Path(source).resolve().as_uri() + "?mode=ro", uri=True) as c:
        tables = [row[0] for row in c.execute(
            "select name from sqlite_master where type='table' and name like 'HIPOPS_%'")]
        selected = []
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            start = c.execute(f"select min(BeginNs) from {quoted} where dev_id=?",
                              (device,)).fetchone()[0]
            if start is not None:
                selected.append((quoted, start))
        if len(selected) != 1:
            raise ValueError("offline window requires exactly one GPU process table")
        table, start = selected[0]
        low, high = c.execute(
            f"select min(_Index),max(_Index) from {table} where dev_id=? and BeginNs<?",
            (device, start + int(seconds * 1e9))).fetchone()
        if low is None or low < 0:
            raise ValueError("window has no valid launch indices")
        count, duration, first, last, pid, max_pid = c.execute(
            f"select count(*),sum(DurationNs),min(BeginNs),max(EndNs),min(pid),max(pid) "
            f"from {table} where dev_id=? and _Index between ? and ?",
            (device, low, high)).fetchone()
        assert count and pid == max_pid
        return dict(index_range=f"{low}:{high}", pid=pid, kernel_calls=count,
                    kernel_total_ns=duration, first_gpu_ns=first, last_gpu_ns=last,
                    requested_seconds=seconds, actual_span_ns=last-first)


def scheduler_metrics(payload):
    """Keep raw histogram labels; gauges describe scheduling, not kernel shapes."""
    prefixes = ("vllm:num_requests_running", "vllm:num_requests_waiting",
                "vllm:kv_cache_usage_perc", "vllm:iteration_tokens_total",
                "vllm:num_preemptions")
    samples = [line for line in payload.splitlines()
               if line.startswith(prefixes)]
    result = {"samples": samples}
    for key, name in (("running", prefixes[0]), ("waiting", prefixes[1]),
                      ("kv_usage", prefixes[2])):
        values = [float(line.rsplit(None, 1)[1]) for line in samples
                  if line.split("{", 1)[0].split()[0] == name]
        if not values:
            raise ValueError(f"missing scheduler metric {name}")
        result[key] = sum(values)
    return result


def analyze_node(root):
    """Offline only. Per-rank summed durations are not critical-path wall time."""
    root = Path(root)
    rank_map = {}
    for line in (root / "server.log").read_text(errors="replace").splitlines():
        match = re.search(r"\(Worker pid=(\d+)\).*world_size=16 rank=(\d+) local_rank=", line)
        if match:
            rank_map[int(match[1])] = int(match[2])
    assert len(rank_map) == 8, rank_map
    def read_stats(path):
        rows, pid = defaultdict(list), None
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("# PID:"):
                pid = int(line.split(":", 1)[1])
            elif pid is not None and line.startswith('"'):
                row = next(csv.reader([line]))
                if row[0] not in ("Name", "Total"):
                    rows[pid].append((row[0], int(row[1]), float(row[2]), float(row[3])))
        return rows
    def bucket(name):
        name = name.lower()
        if any(x in name for x in ("nccl", "rccl", "allreduce", "all_reduce", "allgather", "reduce_scatter")):
            return "Communication"
        if any(x in name for x in ("mqa", "mla", "attn", "attention", "indexer", "quant_and_cache")):
            return "Attention/indexer"
        if any(x in name for x in ("gdn", "gated_delta", "recurrent_gated")):
            return "Linear Attention"
        if "moe" in name:
            return "MoE"
        if any(x in name for x in ("gemm", "matmul", "cijk_")):
            return "GEMM (projection/MoE unresolved)"
        if any(x in name for x in ("norm", "silu", "gelu", "activation")):
            return "Norm/activation"
        if any(x in name for x in ("topk", "top_k", "sort")):
            return "Selection (indexer/router unresolved)"
        return "Other"
    kernels = read_stats(root / "tmp/trace.hipkernel.csv")
    apis = read_stats(root / "tmp/trace.hiptrace.csv")
    lines = ["trace_analysis=PASS (8 local ranks); request/shutdown status see run_metadata.txt",
        "Scope: trim10 TP16 Graph, 4096->128 C64 N64; not full-model/H100 performance or accuracy.",
        "Sources: tmp/trace.hipkernel.csv, tmp/trace.hiptrace.csv, tmp/trace.db, tmp/trace_stream.json, server.log",
        "Columns: Name, Calls, TotalDurationNs, AverageNs; percentages recomputed per PID.",
        "Buckets are heuristic keyword classifications, NOT exact phase or source attribution.",
        "Summed kernel/API durations can overlap; do not sum 16 ranks as request wall time.",
        "Communication kernels include device waits; their percentage is not automatically recoverable speedup."]
    result = {}
    for pid, rank in sorted(rank_map.items(), key=lambda item: item[1]):
        rows = kernels[pid]
        assert rows and apis[pid], (rank, pid)
        total = sum(row[2] for row in rows)
        buckets = defaultdict(float)
        for row in rows:
            buckets[bucket(row[0])] += row[2]
        graphs = sum(row[1] for row in apis[pid] if row[0] == "hipGraphLaunch")
        assert graphs > 0, (rank, pid, "no hipGraphLaunch")
        top = sorted(rows, key=lambda row: row[2], reverse=True)
        top_api = sorted(apis[pid], key=lambda row: row[2], reverse=True)
        result[rank] = dict(pid=pid, kernel_total_s=total / 1e9,
            unique_kernels=len(rows), kernel_calls=sum(row[1] for row in rows),
            graph_launches=graphs,
            buckets_pct={k: v / total * 100 for k, v in buckets.items()},
            top=[dict(name=row[0], calls=row[1], total_s=row[2]/1e9,
                      pct=row[2]/total*100, bucket=bucket(row[0])) for row in top[:15]],
            top_api=[dict(name=row[0], calls=row[1], total_s=row[2]/1e9) for row in top_api[:12]])
        lines += ["", f"[TP{rank}] pid={pid} kernel_total_s={total/1e9:.6f} unique_kernels={len(rows)} hipGraphLaunch={graphs}"]
        for name, value in sorted(buckets.items(), key=lambda item: -item[1]):
            lines.append(f"bucket={name} seconds={value/1e9:.6f} pct={value/total*100:.4f}")
        lines.append("Top 20 kernels:")
        for row in top[:20]:
            lines.append(f"{row[2]/total*100:.4f}% calls={row[1]} total_s={row[2]/1e9:.6f} avg_us={row[3]/1e3:.3f} category={bucket(row[0])} name={row[0]}")
        lines.append("Top 12 HIP APIs (summed inclusive time; not CPU compute time):")
        for row in top_api[:12]:
            lines.append(f"calls={row[1]} total_s={row[2]/1e9:.6f} avg_us={row[3]/1e3:.3f} name={row[0]}")
    (root / "kernel_bucket_analysis.txt").write_text("\n".join(lines) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--run-root", type=Path, required=True, help="Same name, LOCAL storage on each node")
    p.add_argument("--node0", default="10.232.2.18")
    p.add_argument("--node1", default="10.232.2.10")
    p.add_argument("--container1", default="yxa-plugin-glm5_2-10")
    p.add_argument("--ssh-host", default="bastion.aiops.baai.ac.cn")
    p.add_argument("--ssh-user", default="yuxinan@root@10.232.2.10")
    p.add_argument("--ssh-port", type=int, default=2224)
    p.add_argument("--port", type=int, default=18052)
    p.add_argument("--master-port", type=int, default=29652)
    p.add_argument("--interface", default="bond1")
    p.add_argument("--alias", default="glm-5.2-trim10")
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--memory", type=float, default=.9)
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--kv-cache-memory-bytes", type=int)
    p.add_argument("--reuse-server-cache", type=Path)
    p.add_argument("--rdma-library-path", default="")
    p.add_argument("--network-backend", choices=("auto", "shca"), default="auto",
                   help="shca explicitly selects the isolated IBext_v8 plugin; no Socket fallback")
    p.add_argument("--ib-hca", default="shca_0,shca_1,shca_2,shca_3")
    p.add_argument("--profiler", default="/opt/dtk/hip/bin/hipprof")
    p.add_argument("--profile-devices", default="0,1,2,3,4,5,6,7",
                   help="Local GPU event filter, NOT a model GPU visibility mask")
    p.add_argument("--profile-seconds", type=float,
                   help="Offline GPU window from first recorded kernel; no live stop, requests finish normally")
    p.add_argument("--session", type=int, default=52700, help="Node1 uses session+1")
    p.add_argument("--startup-seconds", type=int, default=2400)
    p.add_argument("--phase-seconds", type=int, default=1200)
    p.add_argument("--shutdown-seconds", type=int, default=90,
                   help="SIGINT grace per node for service/profiler drain, 1..1800 seconds")
    p.add_argument("--mla-shape-census", action="store_true",
                   help="Input/shape diagnostic only: no hipprof or performance claims")
    p.add_argument("--moe-shape-census", action="store_true",
                   help="MoE routes/shapes/input diagnostic only; timing invalid")
    p.add_argument("--benchmark-only", action="store_true", help="Same two-node lifecycle without profiler or input capture")
    p.add_argument("--mla-kernel", choices=("flaggems", "native_flashmla"), default="flaggems")
    p.add_argument("--no-async-scheduling", action="store_true",
                   help="Opt-in serialized scheduling candidate; keeps HIP Graph enabled")
    p.add_argument("--split-graph-comm", action="store_true",
                   help="Hygon TP candidate: separate captured/eager RCCL communicators; keeps async scheduling")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--flaggems-root", type=Path, default=os.environ.get("GLM52_FLAGGEMS_ROOT"))
    p.add_argument("--compressor-root", type=Path, default=os.environ.get("GLM52_COMPRESSOR_ROOT"))
    p.add_argument("--kv-cache-dtype", choices=("auto", "bfloat16", "fp8_ds_mla"), default="auto")
    p.add_argument("--fp8-mla-impl", choices=("vendor", "triton_grouped"), default="vendor")
    p.add_argument("--fp8-reuse-metadata", action="store_true")
    a = p.parse_args()
    if a.fp8_mla_impl != "vendor" and a.kv_cache_dtype != "fp8_ds_mla":
        p.error("triton_grouped requires --kv-cache-dtype fp8_ds_mla")
    if not 0 < a.shutdown_seconds <= 1800:
        p.error("shutdown seconds must be in 1..1800")
    if sum((a.benchmark_only, a.mla_shape_census, a.moe_shape_census)) > 1:
        p.error("benchmark and census modes are mutually exclusive")
    use_census = a.mla_shape_census or a.moe_shape_census
    if a.network_backend == "shca" and not a.rdma_library_path:
        p.error("shca requires explicit --rdma-library-path")
    use_profiler = not (a.benchmark_only or use_census)
    try:
        profile_devices = [int(x) for x in a.profile_devices.split(",")]
    except ValueError:
        p.error("profile devices must be comma-separated integers")
    if (not profile_devices or len(set(profile_devices)) != len(profile_devices)
            or min(profile_devices) < 0 or max(profile_devices) > 7):
        p.error("profile devices must be unique local IDs in 0..7")
    if a.profile_seconds is not None and (
            not use_profiler or not 0 < a.profile_seconds < a.phase_seconds):
        p.error("profile seconds must be positive, below phase budget, and used only with profiler")
    if a.profile_seconds is not None and len(profile_devices) != 1:
        p.error("offline window requires exactly one selected GPU per node")
    project, root = a.project.resolve(), a.run_root.resolve()
    runner = Path(__file__).with_name("dual_node_smoke.py").resolve()
    plugin = runner.parents[2]
    python = project / "repos/vllm-v0.24.0/venv-vllm-v0_24_0-empty/bin/python"
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "-o", "StrictHostKeyChecking=yes", "-p", str(a.ssh_port),
           "-l", a.ssh_user, a.ssh_host]
    common = ["env", "-u", "NCCL_NET_PLUGIN", "-u", "NCCL_NET",
              "NCCL_IB_DISABLE=0", "NCCL_IB_HCA=" + a.ib_hca,
              "GLM52_RDMA_LIBRARY_PATH=" + a.rdma_library_path,
              "VLLM_FL_HYGON_SPLIT_GRAPH_COMM=" + ("1" if a.split_graph_comm else "0"),
              *(["IBV_DRIVERS=shca"] if a.rdma_library_path else []),
              *(["NCCL_NET_PLUGIN=shca", "NCCL_NET=IBext_v8"] if a.network_backend == "shca" else []),
              "python3", str(runner), "--project", str(project),
              "--master-addr", a.node0, "--master-port", str(a.master_port),
              "--interface", a.interface, "--model", a.model,
              "--gpu-memory-utilization", str(a.memory), "--port", str(a.port),
              "--served-model-name", a.alias, "--num-prompts", str(a.num_prompts),
              "--output-len", str(a.output_len), "--max-model-len", str(a.max_model_len), "--live-log"]
    common += ["--mla-kernel", a.mla_kernel, "--kv-cache-dtype", a.kv_cache_dtype,
               "--fp8-mla-impl", a.fp8_mla_impl]
    if a.flaggems_root is not None:
        common += ["--flaggems-root", str(a.flaggems_root.resolve())]
    if a.compressor_root is not None:
        common += ["--compressor-root", str(a.compressor_root.resolve())]
    if a.fp8_reuse_metadata:
        common.append("--fp8-reuse-metadata")
    if a.no_async_scheduling:
        common.append("--no-async-scheduling")
    if a.benchmark_only:
        common.append("--graph-proof")
    servers = []
    for rank, ip in enumerate((a.node0, a.node1)):
        servers.append([a.profiler, "--session", str(a.session + rank),
            "--hip-trace", "--follow-fork", "--show-pid", "--trace-args",
            "--devices", a.profile_devices, "--output-type", "0",
            "--group-stream", "--stats", "--trace-off",
            "-d", str(root / "tmp"), "-o", str(root / "tmp/trace"),
            *common, "--mode", "serve", "--node-rank", str(rank),
            "--node-ip", ip, "--run-root", str(root / "runtime")])
        if a.kv_cache_memory_bytes is not None:
            servers[-1] += ["--kv-cache-memory-bytes", str(a.kv_cache_memory_bytes)]
        if a.reuse_server_cache is not None:
            servers[-1] += ["--cache-root", str(a.reuse_server_cache)]
        if not use_profiler:
            servers[-1] = servers[-1][servers[-1].index("env"):]
        if a.mla_shape_census:
            servers[-1].append("--mla-shape-census")
        if a.moe_shape_census:
            servers[-1].append("--moe-shape-census")
    benches = {phase: [*common, "--mode", "bench", "--node-rank", "0",
        "--node-ip", a.node0, "--run-root", str(root / phase)]
        for phase in ("warmup", "measure")}
    exports = [a.profiler, "--db", str(root / "tmp/trace.db"), "--hip-trace",
        "--devices", a.profile_devices,
        "--show-pid", "--output-type", "0", "--segment-size", "1000000000",
        "--group-stream", "-d", str(root / "tmp/export"),
        "-o", str(root / "tmp/export/trace_stream")]
    plan = {"args": vars(a), "devices_each": list(range(8)), "TP": 16, "PP": 1, "DP": 1,
        "profile_devices_each": profile_devices, "profile_window_seconds": a.profile_seconds,
        "profile_window_policy": "natural_exit_then_offline_index_range" if a.profile_seconds is not None else None,
        "input_tokens": 4096, "concurrency": 64, "max_model_len": a.max_model_len,
        "kv_cache_memory_bytes": a.kv_cache_memory_bytes,
        "max_num_batched_tokens": 8192, "prefix_cache": False,
        "graph": "FULL_AND_PIECEWISE sizes 1,2,4,8,16,32,64", "warmup_rounds": 1,
        "EOS": "ignore", "servers": servers, "benches": benches, "export": exports,
        "flow": ["resource/port check", "setsid hipprof trace-off both nodes",
                 "health", "warmup", "start both sessions", "measure",
                 "stop then flush both sessions", "SIGINT exact recorded groups",
                 "DB quick_check/event check", "offline export"],
        "scope": "trim10 diagnostic hotspots, not full-model performance or accuracy"}
    if a.profile_seconds is not None:
        plan["flow"] = ["resource/port check", "trace-off launch", "health",
                        "same-workload warmup", "start both sessions", "complete all requests",
                        "SIGINT exact groups; profiler exits naturally",
                        "read-only launch-index window", "bounded offline export and count check"]
    if use_census:
        plan.update(export=None, scope="real input/shape census; timing invalid",
            flow=["resource/port check", "setsid native mp both nodes", "health",
                  "warmup", "enable TP0 census", "measure requests and save inputs",
                  "SIGINT exact recorded groups", "check census evidence"])
    elif a.benchmark_only:
        plan.update(export=None, scope="trim10 exploratory E2E performance; not full-model or accuracy qualification",
            flow=["resource/port check", "setsid native mp both nodes", "health",
                  "same-workload warmup", "measure", "SIGINT exact recorded groups", "Graph/request/resource checks"])
    if a.dry_run:
        print(json.dumps(plan, default=str, indent=2))
        return
    if root.exists():
        p.error("Refusing to overwrite run root")
    root.mkdir(parents=True)
    (root / "tmp").mkdir()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    metadata = dict(plan, start=dt.datetime.now(dt.timezone.utc).isoformat(),
        status="RUNNING", stage="prepare", groups={}, events=[],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        plugin_head=subprocess.check_output(["git", "-C", str(plugin), "rev-parse", "HEAD"], text=True).strip(),
        model_config_sha256=hashlib.sha256((Path(a.model) / "config.json").read_bytes()).hexdigest(),
        model_index_sha256=hashlib.sha256((Path(a.model) / "model.safetensors.index.json").read_bytes()).hexdigest())
    if use_census:
        census_file = "moe_shape_census.py" if a.moe_shape_census else "mla_shape_census.py"
        metadata["census_sha256"] = hashlib.sha256(runner.with_name(census_file).read_bytes()).hexdigest()
    def save(stage):
        metadata["stage"] = stage
        metadata["events"].append([dt.datetime.now(dt.timezone.utc).isoformat(), stage])
        (root / "run_metadata.txt").write_text(json.dumps(metadata, default=str, indent=2))
        print("STAGE", stage, flush=True)
    def node(rank, command, timeout=60):
        cmd = command if rank == 0 else [*ssh,
            shlex.join(["docker", "exec", a.container1, *command])]
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    local = None
    tracing_started = False
    try:
        save("prepare")
        for rank in (0, 1):
            gpu = node(rank, ["/opt/hyhal/bin/hy-smi", "--showmeminfo", "vram", "--showpids"])
            used = [int(v) for v in re.findall(r"Total Used Memory \(MiB\):\s*(\d+)", gpu)]
            assert len(used) == 8 and max(used) < 16 and "No KFD PIDs" in gpu, gpu
            metadata.setdefault("resource_preflight", {})[str(rank)] = gpu
            check = "import socket; s=socket.socket(); s.bind(('0.0.0.0'," + str(a.master_port) + ")); s.close(); s=socket.socket(); s.bind(('0.0.0.0'," + str(a.port) + ")); s.close()"
            node(rank, ["python3", "-c", check])
            if use_profiler:
                print(node(rank, [a.profiler, "-h"])[:100], flush=True)
        node(1, ["python3", "-c",
            "from pathlib import Path; p=Path(" + repr(str(root)) + "); p.mkdir(); (p/'tmp').mkdir()"])
        save("launch")
        with open(root / "server.log", "w") as log:
            local = subprocess.Popen(servers[0], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        metadata["groups"]["0"] = local.pid
        launch = ("import subprocess; f=open(" + repr(str(root / "server.log")) + ",'w'); "
                  "p=subprocess.Popen(" + repr(servers[1]) +
                  ",stdout=f,stderr=subprocess.STDOUT,start_new_session=True); print(p.pid)")
        metadata["groups"]["1"] = int(node(1, ["python3", "-c", launch]).strip())
        save("health_wait")
        deadline = time.monotonic() + a.startup_seconds
        while time.monotonic() < deadline:
            if local.poll() is not None:
                raise RuntimeError(f"node0 server exited {local.returncode}")
            try:
                with opener.open(f"http://{a.node0}:{a.port}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            time.sleep(10)
        else:
            raise TimeoutError("startup budget exhausted")
        assert "1" in metadata["groups"]
        for phase in ("warmup", "measure"):
            if phase == "measure" and use_census:
                save("census_start")
                census = root / "runtime/serve/census"
                census.mkdir(parents=True, exist_ok=True)
                (census / "ENABLE").touch(exist_ok=False)
            elif phase == "measure" and use_profiler:
                save("profile_start")
                tracing_started = True
                for rank in (0, 1):
                    print(node(rank, [a.profiler, "--session-client", str(a.session + rank), "--start"]))
            save(phase)
            with open(root / ("warmup.log" if phase == "warmup" else "bench.log"), "w") as log:
                proc = subprocess.Popen(benches[phase], stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
                metadata[phase + "_pgid"] = proc.pid
                def tee():
                    for line in proc.stdout:
                        log.write(line); log.flush()
                        print(line, end="", flush=True)
                reader = threading.Thread(target=tee, daemon=True)
                reader.start()
                try:
                    if a.profile_seconds is None:
                        rc = proc.wait(timeout=a.phase_seconds)
                    else:
                        deadline = time.monotonic() + a.phase_seconds
                        active_at = None
                        with open(root / (phase + "_scheduler.jsonl"), "w") as metrics_log:
                            while proc.poll() is None:
                                now = time.monotonic()
                                if now >= deadline:
                                    raise TimeoutError(f"{phase} budget exhausted")
                                with opener.open(f"http://{a.node0}:{a.port}/metrics", timeout=5) as r:
                                    sample = scheduler_metrics(r.read().decode())
                                sample.update(utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                                              monotonic=time.monotonic())
                                metrics_log.write(json.dumps(sample) + "\n")
                                metrics_log.flush()
                                if phase == "measure":
                                    if active_at is None and sample["running"] > 0:
                                        active_at = sample["monotonic"]
                                        metadata["window_first_activity"] = sample
                                try:
                                    proc.wait(timeout=1)
                                except subprocess.TimeoutExpired:
                                    pass
                            rc = proc.returncode
                except BaseException:
                    os.killpg(proc.pid, signal.SIGINT)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=10)
                    raise
                finally:
                    reader.join(timeout=10)
                if rc:
                    raise RuntimeError(f"{phase} exit {rc}")
            result = json.loads((root / phase / "bench/results.json").read_text())
            assert result["completed"] == a.num_prompts
            assert result["total_output_tokens"] == a.num_prompts * a.output_len
            assert result["total_input_tokens"] == a.num_prompts * 4096
            metadata[phase + "_result"] = {k: v for k, v in result.items() if not isinstance(v, list)}
        if a.profile_seconds is not None and "window_first_activity" not in metadata:
            raise RuntimeError("INCONCLUSIVE: no observed active requests")
        metadata["status"] = "MEASURE_PASS"
    except BaseException as exc:
        metadata.update(status="FAIL", error=repr(exc))
        raise
    finally:
        save("stop_flush")
        for action in ("--stop", "--flush"):
            for rank in (0, 1):
                if tracing_started and a.profile_seconds is None:
                    try:
                        print(node(rank, [a.profiler, "--session-client", str(a.session + rank), action]))
                    except Exception as exc:
                        metadata.setdefault("cleanup_errors", []).append(repr(exc))
        save("service_shutdown")
        for rank, pgid in list(metadata["groups"].items()):
            command, timeout = shutdown_group_command(pgid, a.shutdown_seconds)
            try:
                cleanup = node(int(rank), command, timeout=timeout)
                print(cleanup, flush=True)
                metadata.setdefault("group_cleanup", {})[rank] = cleanup
            except Exception as exc:
                metadata.setdefault("cleanup_errors", []).append(repr(exc))
        if local:
            try:
                metadata["local_exit"] = local.wait(timeout=120)
            except subprocess.TimeoutExpired:
                metadata.setdefault("cleanup_errors", []).append("node0 group did not exit in 120s")
        save("shutdown_recorded")
    if use_census:
        records = [json.loads(line) for line in (census / "shapes.jsonl").read_text().splitlines()]
        ops = [r for r in records if r["event"] == "op"]
        graphs = [r for r in records if r["event"] == "graph_replay"]
        snapshots = [r for r in records if r["event"] == "saved"]
        assert ops and graphs and snapshots, (len(ops), len(graphs), len(snapshots))
        assert all((census / r["file"]).stat().st_size > 0 for r in snapshots)
        assert not metadata.get("cleanup_errors"), metadata.get("cleanup_errors")
        metadata.update(status="CENSUS_COLLECTED", census_counts=dict(ops=len(ops),
            graph_replays=len(graphs), snapshots=len(snapshots)),
            end=dt.datetime.now(dt.timezone.utc).isoformat())
        save("analysis_ready")
        print("OUTPUT", root, "real inputs only; no performance claim", flush=True)
        return
    if a.benchmark_only:
        for rank in (0, 1):
            proof = node(rank, ["python3", "-c",
                "from pathlib import Path; import re,sys; s=Path(sys.argv[1]).read_text(); "
                "p=set(re.findall(r'GLM_GRAPH_REPLAY pid=(\\d+) count=128',s)); "
                "assert len(p)==8,p; print('GRAPH_REPLAY_PASS',sorted(p)); "
                "print('native_routes',s.count('GLM_MLA_ROUTE native_flashmla'))",
                str(root / "server.log")])
            metadata.setdefault("graph_proof", {})[str(rank)] = proof
            print(proof, flush=True)
            gpu = node(rank, ["/opt/hyhal/bin/hy-smi", "--showmeminfo", "vram", "--showpids"])
            used = [int(v) for v in re.findall(r"Total Used Memory \(MiB\):\s*(\d+)", gpu)]
            assert len(used) == 8 and max(used) < 16 and "No KFD PIDs" in gpu, gpu
            metadata.setdefault("resource_final", {})[str(rank)] = gpu
        assert not metadata.get("cleanup_errors"), metadata.get("cleanup_errors")
        metadata.update(status="BENCHMARK_PASS", end=dt.datetime.now(dt.timezone.utc).isoformat())
        save("analysis_ready")
        print("OUTPUT", root, "unprofiled exploratory performance", flush=True)
        return
    save("export")
    for rank in (0, 1):
        check = """import os,sqlite3,sys
from pathlib import Path
p=Path(sys.argv[1]); g=int(sys.argv[2])
for d in Path('/proc').iterdir():
 if d.name.isdigit():
  try:
   if os.getpgid(int(d.name))==g: raise RuntimeError('service group still live: '+d.name)
  except ProcessLookupError: pass
assert not (p/'tmp/trace.db-journal').exists()
c=sqlite3.connect('file:'+str(p/'tmp/trace.db')+'?mode=ro',uri=True)
assert c.execute('pragma quick_check').fetchone()[0]=='ok'
ts=[x[0] for x in c.execute("select name from sqlite_master where type='table'") if x[0].startswith(('HIP_','HIPOPS_'))]
ns={t:c.execute('select count(*) from "'+t+'"').fetchone()[0] for t in ts}
assert sum(ns.values())>0,ns
print(ns)
(p/'tmp/export').mkdir()
"""
        print(node(rank, ["python3", "-c", check, str(root), str(metadata["groups"][str(rank)])]))
        node_exports = list(exports)
        window = None
        if a.profile_seconds is not None:
            window = json.loads(node(rank, ["python3", str(Path(__file__).resolve()),
                "--index-window", str(root / "tmp/trace.db"),
                str(a.profile_seconds), str(profile_devices[0])]))
            metadata.setdefault("offline_windows", {})[str(rank)] = window
            node_exports += ["--index-range", window["index_range"]]
            save("export_window_node" + str(rank))
        print(node(rank, node_exports, timeout=600)[-2000:])
        rename = """from pathlib import Path
import sys
p=Path(sys.argv[1]); e=p/'tmp/export'
for src,dst in [('trace_stream_stream.json','trace_stream.json'),('trace_stream-trace.db.hipkernel.csv','trace.hipkernel.csv'),('trace_stream-trace.db.hiptrace.csv','trace.hiptrace.csv')]:
 s=e/src; d=p/'tmp'/dst; assert s.stat().st_size>0,s
 if d.exists():
  old=p/'tmp/shutdown_stats'; old.mkdir(exist_ok=True)
  backup=old/dst; assert not backup.exists(),backup; d.rename(backup)
 s.rename(d)
"""
        node(rank, ["python3", "-c", rename, str(root)])
        if window is not None:
            verify = """import csv,json,sys
from pathlib import Path
p=Path(sys.argv[1]); w=json.loads(sys.argv[2])
def rows(path):
 pid=None
 for line in path.read_text().splitlines():
  if line.startswith('# PID:'): pid=int(line.split(':',1)[1])
  elif pid==w['pid'] and line.startswith('"'):
   r=next(csv.reader([line]))
   if r[0] not in ('Name','Total'): yield r
ks=list(rows(p/'tmp/trace.hipkernel.csv'))
assert sum(int(r[1]) for r in ks)==w['kernel_calls']
assert sum(int(r[2]) for r in ks)==w['kernel_total_ns']
graphs=sum(int(r[1]) for r in rows(p/'tmp/trace.hiptrace.csv') if r[0]=='hipGraphLaunch')
assert graphs>0, 'window has no graph replay'
print('WINDOW_EXPORT_PASS',w['pid'],w['kernel_calls'],graphs,w['actual_span_ns'])
"""
            print(node(rank, ["python3", "-c", verify, str(root), json.dumps(window)]))
    metadata.update(status="PROFILE_EXPORTED", end=dt.datetime.now(dt.timezone.utc).isoformat())
    save("analysis_ready")
    print("OUTPUT", root, "on both nodes", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--index-window":
        print(json.dumps(profile_index_window(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))))
    elif len(sys.argv) == 3 and sys.argv[1] == "--analyze-run":
        print(json.dumps(analyze_node(sys.argv[2]), indent=2))
    else:
        main()
