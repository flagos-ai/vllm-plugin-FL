"""Counterbalanced TP/PCP campaign with occupancy checks before every process."""

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
SCHEDULE = [
    (
        "r1_tp",
        "tp",
        ["native_flashinfer", "fused_flashinfer", "fused_triton", "native_triton"],
    ),
    ("r1_pcp", "pcp", ["fused_flashinfer", "fused_triton"]),
    ("r2_pcp", "pcp", ["fused_triton", "fused_flashinfer"]),
    (
        "r2_tp",
        "tp",
        ["native_triton", "fused_triton", "fused_flashinfer", "native_flashinfer"],
    ),
]


def main():
    with idle_gpu_lock():
        for tag, mode, variants in SCHEDULE:
            occupancy = require_idle()
            out = ROOT / tag
            out.mkdir(exist_ok=False)
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("VLLM_FL_", "NCCL_"))
            }
            env.update(
                CUDA_VISIBLE_DEVICES="0,1,2,3",
                VLLM_PLUGINS="fl_qwen_cp",
                VLLM_FL_QWEN_CP="0" if mode == "tp" else "1",
                VLLM_ENABLE_V1_MULTIPROCESSING="0",
                VLLM_WORKER_MULTIPROC_METHOD="spawn",
                PYTHONPATH=f"{ROOT}:{SOURCE}",
                PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            command = [
                sys.executable,
                str(ROOT / "timing.py"),
                "--mode",
                mode,
                "--variants",
                *variants,
                "--output",
                str(out / "result.json"),
            ]
            snapshot = out / "source"
            snapshot.mkdir()
            hashes = {}
            for file in (SOURCE / "vllm_fl/cp").glob("*.py"):
                data = file.read_bytes()
                hashes[file.name] = hashlib.sha256(data).hexdigest()
                (snapshot / file.name).write_bytes(data)
            for file in ROOT.glob("*.py"):
                (out / file.name).write_bytes(file.read_bytes())
            manifest = {
                "command": command,
                "source": str(SOURCE),
                "hashes": hashes,
                "gpu_before": occupancy,
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
            (out / "source.diff").write_text(
                subprocess.check_output(["git", "-C", str(SOURCE), "diff"], text=True)
            )
            with (out / "run.log").open("x") as log:
                proc = subprocess.Popen(
                    command,
                    env=env,
                    cwd=out,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                print("START", tag, proc.pid, flush=True)
                try:
                    status = proc.wait(timeout=900)
                    if status:
                        raise RuntimeError(f"{tag} exit {status}: {out / 'run.log'}")
                finally:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGTERM)
            after = {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                for f in (SOURCE / "vllm_fl/cp").glob("*.py")
            }
            assert after == hashes, "Implementation changed during benchmark"
            print("DONE", tag, flush=True)
        (ROOT / "timing_gpu_after.json").write_text(
            json.dumps(require_idle(), indent=2)
        )


if __name__ == "__main__":
    main()
