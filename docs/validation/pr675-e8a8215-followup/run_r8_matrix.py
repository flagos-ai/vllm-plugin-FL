import contextlib
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

repo = Path(__file__).resolve().parents[3]
root = Path(os.environ.get("OUTPUT_DIR", "/tmp")) / (
    "r8-warmed-"
    + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
)
root.mkdir()
print("OUTPUT_ROOT=" + str(root), flush=True)
env = os.environ.copy()
env.update(
    TOPS_VISIBLE_DEVICES="2,3",
    VLLM_PLUGINS="fl",
    PYTHONPATH=str(repo),
    TORCH_GCU_ENABLE_INT64_AND_UINT64="1",
    ENABLE_I64_CHECK="0",
    TORCHGCU_INDUCTOR_ENABLE="1",
    NO_PROXY="localhost,127.0.0.1",
    no_proxy="localhost,127.0.0.1",
    PYTHONUNBUFFERED="1",
)
for model, tp in [("Qwen3-0.6B", 1), ("Qwen3.6-27B", 2)]:
    out = root / model
    out.mkdir()
    port = 18581
    url = f"http://127.0.0.1:{port}/v1"
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        "/models/" + model,
        "--served-model-name",
        "qwen",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp),
        "--enforce-eager",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "32",
        "--gpu-memory-utilization",
        "0.80",
    ]
    (out / "command.json").write_text(json.dumps(cmd, indent=2))
    (out / "head.txt").write_text(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode()
    )
    log = (out / "server.log").open("w")
    server = subprocess.Popen(
        cmd,
        cwd=repo,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        ready = False
        for trial in range(600):
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode}: {out}")
            with contextlib.suppress(requests.RequestException):
                ready = requests.get(url + "/models", timeout=2).status_code == 200
            if ready:
                break
            if trial % 12 == 0:
                print(model, "loading", trial * 5, "seconds", flush=True)
            time.sleep(5)
        if not ready:
            raise TimeoutError(model)
        measure = [
            sys.executable,
            str(Path(__file__).with_name("benchmark_logprobs_service.py")),
            "--base-url",
            url,
            "--output",
            str(out / "measurements"),
            "--server-pid",
            str(server.pid),
            "--max-tokens",
            "32",
            "--concurrency",
            "1",
            "4",
            "16",
        ]
        subprocess.run(measure, check=True, env=env, timeout=2400)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGINT)
            try:
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        log.close()
print("MATRIX_DONE", root, flush=True)
