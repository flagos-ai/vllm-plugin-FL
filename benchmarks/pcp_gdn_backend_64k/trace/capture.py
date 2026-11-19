"""Sequential Nsight capture; refuse to launch when any GPU is occupied."""

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from gpu_guard import idle_gpu_lock, require_idle

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / "vllm-plugin-FL-pcp-gdn-backend"
NSYS = (
    ROOT.parent
    / "profiling_tools/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys"
)


def main():
    with idle_gpu_lock():
        for backend in ("flashinfer", "triton"):
            occupancy = require_idle()
            out = ROOT / backend
            out.mkdir(exist_ok=False)
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("VLLM_FL_", "NCCL_"))
            }
            env.update(
                CUDA_VISIBLE_DEVICES="0,1,2,3",
                VLLM_PLUGINS="fl_qwen_cp",
                VLLM_FL_QWEN_CP="1",
                VLLM_ENABLE_V1_MULTIPROCESSING="0",
                VLLM_WORKER_MULTIPROC_METHOD="spawn",
                PYTHONPATH=f"{ROOT}:{SOURCE}",
                PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            command = [
                str(NSYS),
                "profile",
                "--trace=cuda,nvtx,osrt",
                "--sample=none",
                "--cpuctxsw=none",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--wait=all",
                "--force-overwrite=false",
                "--osrt-threshold=10000",
                "--output",
                str(out / "prefill"),
                sys.executable,
                str(ROOT / "trace_case.py"),
                backend,
                str(out / "result.json"),
            ]
            snapshot = out / "source"
            snapshot.mkdir()
            hashes = {}
            for file in (SOURCE / "vllm_fl/cp").glob("*.py"):
                hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
                (snapshot / file.name).write_bytes(file.read_bytes())
            for file in ROOT.glob("*.py"):
                (out / file.name).write_bytes(file.read_bytes())
            manifest = {
                "command": command,
                "gpu_before": occupancy,
                "hashes": hashes,
                "revision": subprocess.check_output(
                    ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
                ).strip(),
                "environment": {
                    k: v
                    for k, v in env.items()
                    if k.startswith(("CUDA_", "VLLM_", "NCCL_"))
                    or k in {"PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}
                },
            }
            (out / "launch.json").write_text(json.dumps(manifest, indent=2))
            with (out / "run.log").open("x") as log:
                process = subprocess.Popen(
                    command,
                    env=env,
                    cwd=out,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                print("START", backend, process.pid, flush=True)
                try:
                    code = process.wait(timeout=900)
                    if code:
                        raise RuntimeError(f"{backend}: capture exited {code}")
                finally:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
            assert (out / "prefill.nsys-rep").exists()
            assert all(
                hashlib.sha256((SOURCE / "vllm_fl/cp" / name).read_bytes()).hexdigest()
                == digest
                for name, digest in hashes.items()
            )
            print("DONE", backend, flush=True)
        (ROOT / "gpu_after.json").write_text(json.dumps(require_idle(), indent=2))
    # Export after releasing the GPUs and cooperative lock.
    for backend in ("flashinfer", "triton"):
        out = ROOT / backend
        with (out / "export.log").open("x") as log:
            subprocess.run(
                [
                    str(NSYS),
                    "export",
                    "--type=sqlite",
                    "--output",
                    str(out / "prefill.sqlite"),
                    str(out / "prefill.nsys-rep"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print("EXPORTED", backend, flush=True)


if __name__ == "__main__":
    main()
