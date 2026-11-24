import concurrent.futures as cf
import datetime
import json
import os
import signal
import socket
import subprocess
from pathlib import Path

REPO = Path("/workspace/pr675-remove-added-tests")
SRC = REPO / "tools/adaptation-gate-cases"
ROOT = Path("/workspace/adaptation-gate-results") / (
    "review-rms-consistency-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
)
ROOT.mkdir(parents=True)
COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
).strip()


def run(model, mode, devices, port):
    root = ROOT / (model.split("/")[-1] + "-" + mode)
    root.mkdir()
    env = dict(
        os.environ,
        TOPS_VISIBLE_DEVICES=devices,
        VLLM_PLUGINS="fl",
        PYTHONPATH=str(REPO) + ":/workspace/s60-adaptation/src/vllm",
        PYTHONUNBUFFERED="1",
        TENSOR_PARALLEL_SIZE="2",
        MAX_MODEL_LEN="32768",
        PORT=str(port),
        SERVICE_TIMEOUT="7200",
        REQUEST_TIMEOUT="600",
        MODEL_PATH=model,
        SERVED_MODEL_NAME="qwen",
        RESULTS_DIR=str(root),
        SERVER_PID_FILE=str(root / "server.pid"),
        NO_PROXY="localhost,127.0.0.1",
        no_proxy="localhost,127.0.0.1",
        TORCH_GCU_ENABLE_INT64_AND_UINT64="1",
        ENABLE_I64_CHECK="0",
        TORCHGCU_INDUCTOR_ENABLE="1",
    )
    for key in (
        "BASE_URL",
        "GRAPH_COMPILATION_CONFIG",
        "MAX_NUM_SEQS",
        "TORCHINDUCTOR_EMULATE_PRECISION_CASTS",
    ):
        env.pop(key, None)
    if mode == "graph" and "35B" in model:
        env.update(
            MAX_NUM_SEQS="128",
            GRAPH_COMPILATION_CONFIG='{"custom_ops":["none","+SiluAndMul"]}',
        )
    for key in ("VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        env[key] = str(root / key.lower())
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError("occupied port " + str(port))
    manifest = dict(
        commit=COMMIT,
        model=model,
        mode=mode,
        devices=devices,
        port=port,
        expected_scenarios=5,
        expected_requests=26,
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("START", root, flush=True)
    with (root / "serve.log").open("w") as sl, (root / "pytest.log").open("w") as tl:
        server = subprocess.Popen(
            ["bash", "run_serve_" + mode + ".sh"],
            cwd=SRC,
            env=env,
            stdout=sl,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        client = subprocess.Popen(
            ["bash", "run_test.sh"],
            cwd=SRC,
            env=env,
            stdout=tl,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                try:
                    client.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    if server.poll() is not None:
                        os.killpg(client.pid, signal.SIGTERM)
                        client.wait(timeout=30)
                        break
        finally:
            for proc in (client, server):
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGINT)
                    try:
                        proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
        manifest.update(
            test_exit_code=client.returncode, server_exit_code=server.returncode
        )
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("END", json.dumps(manifest), flush=True)
    return manifest


if __name__ == "__main__":
    print("ROOT", ROOT, flush=True)
    results = []
    for mode, base_port in [("graph", 18410), ("eager", 18420)]:
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(run, "/models/Qwen3.6-27B", mode, "0,1", base_port),
                pool.submit(run, "/models/Qwen3.6-35B-A3B", mode, "2,3", base_port + 2),
            ]
            results.extend(future.result() for future in futures)
    (ROOT / "matrix.json").write_text(json.dumps(results, indent=2))
    print("FINISHED", ROOT, flush=True)
    raise SystemExit(0 if all(r["test_exit_code"] == 0 for r in results) else 1)
